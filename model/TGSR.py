import functools
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
from inspect import isfunction
from thop import clever_format,profile

import math

from utils.dcls_utils import get_uperleft_denominator
from utils.module_util import initialize_weights, ResidualBlock_noBN, make_layer


def exists(val):
    return val is not None


def is_empty(t):
    return t.nelement() == 0


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x


def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def center_iter(x, means, buckets=None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q, torch.flip(q[:, N - pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d", ng=ng, h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2, 2 * gs, gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2, 2 * gs, gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)

        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)

        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]

        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)

        return out


class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k, v, x_global.detach()


class TAB(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay=0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.irca_attn = IRCA(dim, qk_dim, heads)
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c', cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

        k_global, v_global, x_means = self.irca_attn(x, x_means)

        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j',
                                    F.normalize(x, dim=-1),
                                    F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)

            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)

        y = self.iasa_attn(x, idx_last, k_global, v_global)
        y = rearrange(y, 'b (h w) c->b c h w', h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x

        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)

        return rearrange(x, 'b (h w) c->b c h w', h=h)


def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, mlp_dim, heads=1):
        super().__init__()

        self.layer = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)

        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)

        return x


class DGAB(nn.Module):
    setting = dict(dim=72, block_num=10, qk_dim=72, mlp_dim=192, heads=6,
                   patch_size=[16, 20, 24, 28, 16, 20, 24, 28, 16, 20])

    def __init__(self, in_chans=3, n_iters=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 num_tokens=[32, 64, 128, 256, 32, 64, 128, 256, 32, 64],
                 group_size=[256, 128, 64, 32, 256, 128, 64, 32, 256, 128],
                 upscale: int = 4):
        super().__init__()

        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']

        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size
        self.reduction = 4
        nf2 = self.dim // self.reduction

        self.cls = CLS(self.dim)
        basic_block = functools.partial(ResidualBlock_noBN, nf=self.dim)
        self.feature_block = make_layer(basic_block, 3)
        self.head1 = nn.Conv2d(self.dim, nf2, 3, 1, 1)
        self.reshapex = nn.Conv2d(self.dim + nf2, self.dim, 3, 1, 1)

        # -----------1 shallow--------------
        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        # ----------2 deep--------------
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()

        for i in range(self.block_num):
            self.blocks.append(nn.ModuleList([TAB(self.dim, self.qk_dim, self.mlp_dim,
                                                  self.heads, self.n_iters[i],
                                                  self.num_tokens[i], self.group_size[i]),
                                              LRSA(self.dim, self.qk_dim,
                                                   self.mlp_dim, self.heads)]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))

        # ----------3 reconstruction---------

        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(self.block_num):
            residual = x

            global_attn, local_attn = self.blocks[i]

            x = global_attn(x)

            x = local_attn(x, self.patch_size[i])

            x = residual + self.mid_convs[i](x)
        return x

    def forward(self, x, kernel):

        if self.upscale != 1:
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else:
            base = x

        x = self.first_conv(x)

        feature = self.feature_block(x)

        f1 = self.head1(feature)

        f2 = self.cls(x, kernel)

        x = torch.cat([f1, f2], dim=1)

        x = self.reshapex(x)

        x = self.forward_features(x) + x

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        out = self.last_conv(out) + base

        return out

    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [K]'.format(self._get_name(),
                                                   num_parameters / 10 ** 3)


class Estimator(nn.Module):
    def __init__(
            self, in_nc=1, nf=32, para_len=10, num_blocks=1, kernel_size=4, filter_structures=[]
    ):
        super(Estimator, self).__init__()

        self.filter_structures = filter_structures
        self.ksize = kernel_size
        self.G_chan = 8
        self.in_nc = in_nc
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)

        self.head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3)
        )

        self.body = nn.Sequential(
            make_layer(basic_block, num_blocks)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(nf, nf, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, para_len, 1),
            nn.Flatten(),
        )

        self.dec = nn.ModuleList()
        for i, f_size in enumerate(self.filter_structures):
            if i == 0:
                in_chan = in_nc
            elif i == len(self.filter_structures) - 1:
                in_chan = in_nc
            else:
                in_chan = self.G_chan
            self.dec.append(nn.Linear(para_len, self.G_chan * in_chan * f_size ** 2))

        self.apply(initialize_weights)

    def calc_curr_k(self, kernels, batch):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.ones([1, batch * self.in_nc]).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(kernels):
            curr_k = F.conv2d(delta, w, padding=self.ksize - 1, groups=batch) if ind == 0 else F.conv2d(curr_k, w,
                                                                                                        groups=batch)
        curr_k = curr_k.reshape(batch, self.in_nc, self.ksize, self.ksize).flip([2, 3])
        return curr_k

    def forward(self, LR):

        batch, channel = LR.shape[0:2]
        f1 = self.head(LR)
        f = self.body(f1) + f1

        latent_kernel = self.tail(f)

        kernels = [self.dec[0](latent_kernel).reshape(
            batch * self.G_chan,
            channel,
            self.filter_structures[0],
            self.filter_structures[0])]

        for i in range(1, len(self.filter_structures) - 1):
            kernels.append(self.dec[i](latent_kernel).reshape(
                batch * self.G_chan,
                self.G_chan,
                self.filter_structures[i],
                self.filter_structures[i]))

        kernels.append(self.dec[-1](latent_kernel).reshape(
            batch * channel,
            self.G_chan,
            self.filter_structures[-1],
            self.filter_structures[-1]))

        K = self.calc_curr_k(kernels, batch).mean(dim=1, keepdim=True)

        # for anisox2
        # K = F.softmax(K.flatten(start_dim=1), dim=1)
        # K = K.view(batch, 1, self.ksize, self.ksize)

        K = K / torch.sum(K, dim=(2, 3), keepdim=True)

        return K


class CLS(nn.Module):
    def __init__(self, nf, reduction=4):
        super().__init__()

        self.reduce_feature = nn.Conv2d(nf, nf // reduction, 1, 1, 0)

        self.grad_filter = nn.Sequential(
            nn.Conv2d(nf // reduction, nf // reduction, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf // reduction, nf // reduction, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf // reduction, nf // reduction, 3),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(nf // reduction, nf // reduction, 1),
        )

        self.expand_feature = nn.Conv2d(nf // reduction, nf, 1, 1, 0)

    def forward(self, x, kernel):
        cls_feats = self.reduce_feature(x)
        kernel_P = torch.exp(self.grad_filter(cls_feats))
        kernel_P = kernel_P - kernel_P.mean(dim=(2, 3), keepdim=True)
        clear_features = torch.zeros(cls_feats.size()).to(x.device)
        ks = kernel.shape[-1]
        dim = (ks, ks, ks, ks)
        feature_pad = F.pad(cls_feats, dim, "replicate")
        for i in range(feature_pad.shape[1]):
            feature_ch = feature_pad[:, i:i + 1, :, :]
            clear_feature_ch = get_uperleft_denominator(feature_ch, kernel, kernel_P[:, i:i + 1, :, :])
            clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]

        x = self.expand_feature(clear_features)

        return x


class TGSR(nn.Module):
    def __init__(
            self,
            nf=32,
            nb=16,
            ng=5,
            in_nc=3,
            reduction=4,
            upscale=4,
            input_para=64,
            kernel_size=21,
            pca_matrix_path=None,
    ):
        super(TGSR, self).__init__()

        self.ksize = kernel_size
        self.scale = upscale

        if kernel_size == 21:
            filter_structures = [11, 7, 5, 1]  # for iso kernels all
        elif kernel_size == 11:
            filter_structures = [7, 3, 3, 1]  # for aniso kernels x2
        elif kernel_size == 31:
            filter_structures = [11, 9, 7, 5, 3]  # for aniso kernels x4
        else:
            print("Please check your kernel size, or reset a group filters for DDLK")

        self.Restorer = DGAB(
            upscale=upscale
        )
        self.Estimator = Estimator(
            kernel_size=kernel_size, para_len=input_para, in_nc=in_nc, nf=nf, filter_structures=filter_structures
        )

    def forward(self, lr):

        kernel = self.Estimator(lr)
        sr = self.Restorer(lr, kernel.detach())

        return sr, kernel


# 定义计算模型参数量的函数
def count_parameters(model, verbose=True):
    """
    计算模型的参数量
    Args:
        model: PyTorch模型
        verbose: 是否打印详细信息
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
        non_trainable_params: 不可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # 格式化输出（K/M/G）
    def format_num(num):
        for unit in ['', 'K', 'M', 'G']:
            if abs(num) < 1024.0:
                return f"{num:.2f} {unit}"
            num /= 1024.0
        return f"{num:.2f} T"

    if verbose:
        print("=" * 50)
        print(f"模型参数量统计")
        print("=" * 50)
        print(f"总参数量: {format_num(total_params)} ({total_params:,})")
        print(f"可训练参数: {format_num(trainable_params)} ({trainable_params:,})")
        print(f"不可训练参数: {format_num(non_trainable_params)} ({non_trainable_params:,})")
        print(f"可训练参数占比: {trainable_params / total_params * 100:.2f}%")
        print("=" * 50)

    return total_params, trainable_params, non_trainable_params


if __name__ == '__main__':
    # 创建模型实例
    model = TGSR( upscale=2)

    # 如果有GPU则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 计算并打印参数量
    xx = torch.randn(1, 3, 180, 320).to(device)

    # 计算并打印参数量
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    flops, _ = profile(model, inputs=(xx,), verbose=False)
    flops_formatted = clever_format(flops, "%.2f")
    print(f"flops: {flops_formatted} ")

    # 测试模型前向传播
    x = torch.randn(1, 3, 16, 16).to(device)
    sr, kl = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出超分图像形状: {sr.shape}")
    print(f"输出核形状: {kl.shape}")


