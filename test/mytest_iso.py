from model.catanet_arch import CATANet
from option.option_vqdam_test import args

from testmodel.cdcl.blindsr import BlindSR
from utils import utility, degradation
from data.srdataset import SRDataset
from torch.utils.data import DataLoader
import torch
import random
import os


def load_model(model, model_path, model_name):
    if os.path.isfile(model_path):
        print("Loading model", model_name, "from", model_path)
        checkpoint = torch.load(model_path,map_location=None)
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Model path does not exist:", model_path)


def main():
    if args.seed is not None:
        random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_paths = [
        '/home/wlxy/VQ-DAM-main/testPT/catanet/x4.pth',
    ]
    model_names = [
        'ccata'
    ]

    models = [CATANet(upscale=4).cuda() for _ in range(len(model_paths))]
    # model_path = '/home/wlxy/VQ-DAM-main/experiment/catadcls-5.75isox4/setting1_x4.pt'
    # model = CATANet(args)
    # model.load_state_dict(torch.load("/home/wlxy/VQ-DAM-main/experiment/catadcls-5.75isox4/x4.pth"))
    # model.load_state_dict(torch.load(model_paths), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model.eval()

    for model, model_path, model_name in zip(models, model_paths, model_names):
        load_model(model, model_path, model_name)

    # creat test dataset and load
    Test_List = ["Set5", 'Set14', "B100", "Urban100"]
    sigmas = [ 0,1.2,2.4,3.6]  # x4
    # sigmas = [1.2]  # x4

    for name in Test_List:
        dataset_test = SRDataset(args, name=name, train=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                     drop_last=False)

        for i in range(0, len(sigmas)):
            sigma = sigmas[i]
            if sigma == 0:
                degrade = degradation.BicubicDegradation(args.scale)
            else:
                degrade = degradation.StableIsoDegradation(args.scale, sigma)
            print(f"Degradation parameters:")
            print("Sigma=", sigma)

            # model_list = models[:]

            batch_deg_test(dataloader_test, model, args, degrade)


def batch_deg_test(test_loader, model, args, degrade):
    with torch.no_grad():
        test_psnr_list = 0
        test_ssim_list = 0

        for batch, (hr, _) in enumerate(test_loader):
            hr = hr.cuda(non_blocking=True)


            hr = crop_border_test(hr, args.scale)

            hr = hr.unsqueeze(1)
            lr = degrade(hr)

            hr = hr[:, 0, ...]
            lr = lr[:, 0, ...]

            hr = utility.quantize(hr, args.rgb_range)
            # lr = lr / 255.
            # for i, model in enumerate(model_list):
            model.eval()

            sr = model(lr)
            # sr = sr * 255.

            sr = utility.quantize(sr, args.rgb_range)

            test_psnr_list = utility.calc_psnr(sr, hr, args.scale, args.rgb_range, benchmark=True)
            test_ssim_list = utility.calc_ssim(sr, hr, args.scale, benchmark=True)

        # for i in range(len(model_list)):
            # print("Model: {}, PSNR: {}, SSIM: {}".format(model_name_list[i], test_psnr_list[i] / len(test_loader),
            #                                              test_ssim_list[i] / len(test_loader)))
        print("{:.2f}/{:.4f}".format(test_psnr_list,test_ssim_list))



def crop_border_test(img, scale):
    b, c, h, w = img.size()

    img = img[:, :, :int(h // scale * scale), :int(w // scale * scale)]

    return img


if __name__ == '__main__':
    main()
