import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
from inspect import isfunction

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


class TokenwiseDegradationEstimator(nn.Module):
    """Estimate descriptors, adaptive degradation states, and assignments once."""

    def __init__(self, in_chans=3, dim=72, num_states=16, hidden_dim=None):
        super().__init__()
        hidden_dim = default(hidden_dim, dim)
        self.num_states = num_states
        self.dim = dim

        def branch():
            return nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            )

        self.lr_branch = branch()
        self.res2_branch = branch()
        self.res4_branch = branch()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

        self.base_queries = nn.Parameter(torch.randn(num_states, dim) * 0.02)
        self.state_q = nn.Linear(dim, dim, bias=False)
        self.state_k = nn.Linear(dim, dim, bias=False)
        self.state_v = nn.Linear(dim, dim, bias=False)
        self.assignment_d = nn.Linear(dim, dim, bias=False)
        self.assignment_s = nn.Linear(dim, dim, bias=False)
        self.state_norm = nn.LayerNorm(dim)

    @staticmethod
    def residual_observation(x, scale):
        _, _, h, w = x.shape
        dh, dw = max(1, h // scale), max(1, w // scale)
        low = F.interpolate(x, size=(dh, dw), mode='bilinear', align_corners=False)
        up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
        return x - up

    def forward(self, lr, tau=1.0, hard=False):
        if tau <= 0:
            raise ValueError("TDE temperature tau must be positive.")

        _, _, h, w = lr.shape
        r2 = self.residual_observation(lr, 2)
        r4 = self.residual_observation(lr, 4)

        d_map = self.fuse(torch.cat([
            self.lr_branch(lr),
            self.res2_branch(r2),
            self.res4_branch(r4),
        ], dim=1))
        descriptors = rearrange(d_map, 'b c h w -> b (h w) c')

        base_queries = self.base_queries.unsqueeze(0).expand(lr.shape[0], -1, -1)
        state_scores = torch.matmul(
            self.state_q(base_queries),
            self.state_k(descriptors).transpose(-1, -2),
        ) / math.sqrt(self.dim)
        states = self.state_norm(
            base_queries + torch.softmax(state_scores, dim=-1) @ self.state_v(descriptors)
        )

        desc_embed = F.normalize(self.assignment_d(descriptors), dim=-1)
        state_embed = F.normalize(self.assignment_s(states), dim=-1)
        assignment_logits = torch.matmul(desc_embed, state_embed.transpose(-1, -2))
        if self.training:
            assignments = F.gumbel_softmax(assignment_logits, tau=tau, hard=hard, dim=-1)
        else:
            assignments = torch.softmax(assignment_logits / tau, dim=-1)

        return {
            'descriptors': descriptors,
            'assignments': assignments,
            'states': states,
            'descriptor_map': d_map,
            'assignment_logits': assignment_logits,
            'hw': (h, w),
        }


class IntraGroupSelfAttention(nn.Module):
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

        if k_global.dim() == 3:
            k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
            v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)
        elif k_global.dim() == 4:
            k_global = k_global.unsqueeze(1).expand(-1, ng, -1, -1, -1)
            v_global = v_global.unsqueeze(1).expand(-1, ng, -1, -1, -1)
        else:
            raise ValueError(f"Unsupported global token shape: {k_global.shape}")

        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]

        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)

        return out


class InterGroupCrossAttention(nn.Module):
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


class GroupedMultiAttention(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay=0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        self.heads = heads

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.igca = InterGroupCrossAttention(dim, qk_dim, heads)
        self.igsa = IntraGroupSelfAttention(dim, qk_dim, heads, group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.last_vis = None

    def forward(self, x, assignments=None, states=None, return_vis=False):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        if states is not None:
            k_global = self.igca.to_k(states)
            v_global = self.igca.to_v(states)
            k_global = rearrange(k_global, 'b n (h dim_head)->b h n dim_head', h=self.heads)
            v_global = rearrange(v_global, 'b n (h dim_head)->b h n dim_head', h=self.heads)
            x_means = None
        else:
            if not self.initted:
                pad_n = self.num_tokens - N % self.num_tokens
                paded_x = torch.cat((x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
                x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c', cnt=self.num_tokens), dim=-2).detach()
            else:
                x_means = self.means.detach()

        if self.training and x_means is not None:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

            k_global, v_global, x_means = self.igca(x, x_means)
        elif x_means is not None:
            k_global, v_global, x_means = self.igca(x, x_means)

        with torch.no_grad():
            if assignments is not None:
                if assignments.shape[1] != N:
                    raise ValueError(
                        f"TDE assignment length {assignments.shape[1]} does not match feature length {N}."
                    )
                assignment_for_vis = assignments.detach()
                x_belong_idx = torch.argmax(assignments, dim=-1)
            else:
                x_scores = torch.einsum('b i c,j c->b i j',
                                        F.normalize(x, dim=-1),
                                        F.normalize(x_means, dim=-1))
                assignment_for_vis = torch.softmax(x_scores, dim=-1).detach()
                x_belong_idx = torch.argmax(x_scores, dim=-1)

            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)

        if return_vis:
            self.last_vis = {
                'assignments': assignment_for_vis,
                'group': x_belong_idx.detach(),
                'hw': (h, w),
            }

        y = self.igsa(x, idx_last, k_global, v_global)
        y = rearrange(y, 'b (h w) c->b c h w', h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x

        if self.training and x_means is not None:
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


class LightweightWindowSelfAttention(nn.Module):
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


class DegradationAwareGroupedAttentionBlock(nn.Module):
    """DGAB: grouped global interaction followed by local refinement."""

    def __init__(self, dim, qk_dim, mlp_dim, heads, patch_size, n_iter, num_tokens, group_size):
        super().__init__()
        self.patch_size = patch_size
        self.grouped_attention = GroupedMultiAttention(
            dim,
            qk_dim,
            mlp_dim,
            heads,
            n_iter,
            num_tokens,
            group_size,
        )
        self.local_attention = LightweightWindowSelfAttention(dim, qk_dim, mlp_dim, heads)
        self.fusion = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, assignments, states, return_vis=False):
        residual = x
        x = self.grouped_attention(
            x,
            assignments=assignments,
            states=states,
            return_vis=return_vis,
        )
        x = self.local_attention(x, self.patch_size)
        x = residual + self.fusion(x)
        vis = self.grouped_attention.last_vis if return_vis else None
        return x, vis


# Short public name used in the paper and configuration files.
DGAB = DegradationAwareGroupedAttentionBlock


class DegradationAwareReconstruction(nn.Module):
    setting = dict(dim=72, block_num=10, qk_dim=72, mlp_dim=192, heads=6,
                   patch_size=[16, 20, 24, 28, 16, 20, 24, 28, 16, 20])

    def __init__(self, in_chans=3, n_iters=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 num_tokens=[32, 64, 128, 256, 32, 64, 128, 256, 32, 64],
                 group_size=[256, 128, 64, 32, 256, 128, 64, 32, 256, 128],
                 upscale: int = 4,
                 num_states: int = 16,
                 assignment_temperature: float = 1.0,
                 enable_tde: bool = True):
        super().__init__()

        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        self.enable_tde = enable_tde
        self.assignment_temperature = assignment_temperature

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

        for i in range(self.block_num):
            self.blocks.append(DGAB(
                self.dim,
                self.qk_dim,
                self.mlp_dim,
                self.heads,
                self.patch_size[i],
                self.n_iters[i],
                self.num_tokens[i],
                self.group_size[i],
            ))

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

        self.tde = TokenwiseDegradationEstimator(
            in_chans=in_chans,
            dim=self.dim,
            num_states=num_states,
        ) if enable_tde else None
        self.last_tde_vis = None

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

    def forward_features(self, x, tde_out=None, return_vis=False):
        assignments = tde_out['assignments'] if tde_out is not None else None
        states = tde_out['states'] if tde_out is not None else None
        self.last_tde_vis = None

        for i, block in enumerate(self.blocks):
            x, block_vis = block(
                x,
                assignments=assignments,
                states=states,
                return_vis=return_vis and i == 0,
            )
            if return_vis and i == 0:
                self.last_tde_vis = block_vis
        return x

    def forward(self, x, kernel, return_vis=False):
        lr = x
        tde_out = self.tde(
            lr,
            tau=self.assignment_temperature,
        ) if self.tde is not None else None

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

        x = self.forward_features(x, tde_out=tde_out, return_vis=return_vis) + x

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        out = self.last_conv(out) + base

        if return_vis:
            tde_vis = None
            if tde_out is not None:
                tde_vis = {
                    'descriptors': tde_out['descriptors'].detach(),
                    'assignments': tde_out['assignments'].detach(),
                    'states': tde_out['states'].detach(),
                    'descriptor_map': tde_out['descriptor_map'].detach(),
                    'assignment_logits': tde_out['assignment_logits'].detach(),
                    'group': torch.argmax(tde_out['assignments'], dim=-1).detach(),
                    'hw': tde_out['hw'],
                    'dgab': self.last_tde_vis,
                }
            return out, tde_vis

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
        delta = torch.ones(
            [1, batch * self.in_nc],
            device=kernels[0].device,
            dtype=kernels[0].dtype,
        ).unsqueeze(-1).unsqueeze(-1)
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
            clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks].real

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
            num_states=16,
            assignment_temperature=1.0,
            enable_tde=True,
    ):
        super().__init__()

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

        self.reconstruction = DegradationAwareReconstruction(
            in_chans=in_nc,
            upscale=upscale,
            num_states=num_states,
            assignment_temperature=assignment_temperature,
            enable_tde=enable_tde,
        )
        self.kernel_estimator = Estimator(
            kernel_size=kernel_size, para_len=input_para, in_nc=in_nc, nf=nf, filter_structures=filter_structures
        )

    def forward(self, lr, return_aux=False, return_vis=False):

        kernel = self.kernel_estimator(lr)
        if return_vis:
            sr, tde_vis = self.reconstruction(lr, kernel.detach(), return_vis=True)
            return sr, {'kernel': kernel, 'tde': tde_vis}

        sr = self.reconstruction(lr, kernel.detach())

        if return_aux:
            return sr, {'kernel': kernel}
        return sr

    def export_tde_visualization_inputs(self, lr, out_dir, prefix='tde', sample_idx=0):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, aux = self.forward(lr, return_vis=True)
            paths = save_tde_visualization_inputs(lr, aux['tde'], out_dir, prefix, sample_idx)
        if was_training:
            self.train()
        return paths


def _npy_shape_repr(shape):
    shape = tuple(int(v) for v in shape)
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(v) for v in shape) + ")"


def _write_npy_without_numpy(path, tensor, dtype='float32'):
    import struct
    from array import array

    tensor = tensor.detach().cpu().contiguous()
    if dtype == 'float32':
        tensor = tensor.float()
        descr = '<f4'
        values = array('f', tensor.reshape(-1).tolist())
    elif dtype == 'int64':
        tensor = tensor.long()
        descr = '<i8'
        values = array('q', tensor.reshape(-1).tolist())
    else:
        raise ValueError(f"Unsupported fallback npy dtype: {dtype}")

    header = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
        descr,
        _npy_shape_repr(tensor.shape),
    )
    header_bytes = header.encode('latin1')
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b' ' * padding + b'\n'

    with open(path, 'wb') as f:
        f.write(b'\x93NUMPY')
        f.write(bytes([1, 0]))
        f.write(struct.pack('<H', len(header_bytes)))
        f.write(header_bytes)
        values.tofile(f)


def _save_tensor_npy(path, tensor, dtype='float32'):
    try:
        import numpy as np

        array = tensor.detach().cpu().contiguous()
        if dtype == 'float32':
            array = array.float().numpy().astype(np.float32, copy=False)
        elif dtype == 'int64':
            array = array.long().numpy().astype(np.int64, copy=False)
        else:
            raise ValueError(f"Unsupported npy dtype: {dtype}")
        np.save(str(path), array)
    except Exception:
        _write_npy_without_numpy(path, tensor, dtype)


def _save_lr_tensor_image(path, lr, sample_idx=0):
    from PIL import Image

    image = lr.detach().cpu()[sample_idx].clamp(0, 1)
    c, h, w = image.shape
    if c == 1:
        image = image.expand(3, -1, -1)
    elif c == 2:
        image = torch.cat([image, image[:1]], dim=0)
    elif c > 3:
        image = image[:3]

    pixels = []
    for y in range(h):
        for x in range(w):
            pixels.append(tuple(int(round(float(image[ch, y, x]) * 255)) for ch in range(3)))

    out = Image.new('RGB', (w, h))
    out.putdata(pixels)
    out.save(path)


def save_tde_visualization_inputs(lr, tde_vis, out_dir, prefix='tde', sample_idx=0):
    """Save LR image, TDE descriptors, and assignment matrix for visualization."""
    if tde_vis is None:
        raise ValueError("TDE visualization is unavailable. Set enable_tde=True and call return_vis=True.")

    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = tde_vis['hw']

    lr_path = out_dir / f'{prefix}_lr.png'
    descriptor_path = out_dir / f'{prefix}_descriptors.npy'
    assignment_path = out_dir / f'{prefix}_assignments.npy'
    group_path = out_dir / f'{prefix}_group.npy'

    descriptor = tde_vis['descriptors'][sample_idx].reshape(h, w, -1)
    assignments = tde_vis['assignments'][sample_idx].reshape(h, w, -1)
    group = tde_vis['group'][sample_idx].reshape(h, w)

    _save_lr_tensor_image(lr_path, lr, sample_idx)
    _save_tensor_npy(descriptor_path, descriptor, 'float32')
    _save_tensor_npy(assignment_path, assignments, 'float32')
    _save_tensor_npy(group_path, group, 'int64')

    return {
        'lr': str(lr_path),
        'descriptor': str(descriptor_path),
        'assignment': str(assignment_path),
        'group': str(group_path),
        'height': h,
        'width': w,
    }

