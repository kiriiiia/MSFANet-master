import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from metrics import ssim
import torchvision.transforms as transforms

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            tm = time.time() #计算时间

            pred = model(input_img)[2]

            elapsed = time.time() - tm
            adder(elapsed)

            SSIM = ssim(pred,label_img).item()
            ssim_adder(SSIM)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image: #save the dehazed image
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            PSNR = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(PSNR)

            print('%d iter PSNR: %.2fdB  SSIM: %.4f time: %f' % (iter_idx + 1, PSNR , SSIM , elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB,The average SSIM is %.4f' % (psnr_adder.average(),ssim_adder.average()))
        print("Average time: %f" % adder.average())
