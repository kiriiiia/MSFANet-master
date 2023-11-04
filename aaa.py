import cv2
import os
import numpy
import math
from PIL import Image as Image
from statistics import mean
from skimage.metrics import structural_similarity

def psnr(img1, img2):
    mse = numpy.mean((img1- img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
image_dir = "/home/ubuntu/wjjexperiemnt/dataset/NHAZE2020/test"

mypath = "/home/ubuntu/wjjexperiemnt/dataset/NHAZE2020/test"
hazy = os.listdir("/home/ubuntu/wjjexperiemnt/dataset/NHAZE2020/test/hazy") #一个列表里面全是图片
clear = os.listdir("/home/ubuntu/wjjexperiemnt/dataset/NHAZE2020/test/clear")


my_result = [os.path.join(image_dir,"hazy",img) for img in hazy]
print(my_result)

# clear_idx = img.split("/")[-1].split("_")[0]
# clear_idx = img.split("\\")[-1].split("_")[0] #在windows分割“\”符号要用共两个”\\“
# print(clear_idx)

clear_imgs= [os.path.join(image_dir,"clear",img) for img in clear]
print(clear_imgs)
psnr_block = []
ssim_block = []
for i in range (0,5):
    a = psnr(cv2.imread(my_result[i]),cv2.imread(clear_imgs[i]))
    b = structural_similarity(cv2.imread(clear_imgs[i]),cv2.imread(my_result[i]),multichannel=True)
    print(my_result[i])
    print(clear_imgs[int(i)])
    print(a)
    psnr_block.append(a)
    ssim_block.append(b)

print(mean(psnr_block),mean(ssim_block))