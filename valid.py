import torch
from torchvision.transforms import functional as F
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
from metrics import ssim
# from data_for_NHHAZE import valid_dataloader
from data import valid_dataloader

def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sots = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('Start SOTS Evaluation')
        for idx, data in enumerate(sots):
            input_img, label_img = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            pred = model(input_img)

            pred_clip = torch.clamp(pred[2], 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            SSIM = ssim(pred[2],label_img).item()
            psnr_adder(psnr)
            ssim_adder(SSIM)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average(),ssim_adder.average()
