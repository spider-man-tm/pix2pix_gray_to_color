import math
import os
import random

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def path_conf(path):
    if os.path.exists(path)==False:
        os.makedirs(path)


def df_to_log_unet(df_path):
    log = []
    df = pd.read_csv(df_path)
    for i in range(df.shape[0]):
        log_epoch = {
            'epoch': df.iloc[i][0],
            'train_l1_loss': df.iloc[i][1],
            'val_l1_loss': df.iloc[i][2],
            'train_psnr': df.iloc[i][3],
            'val_psnr': df.iloc[i][4],
        }
        log.append(log_epoch)
        
    return log


def df_to_log_pix2pix(df_path):
    log = []
    df = pd.read_csv(df_path)
    for i in range(df.shape[0]):
        log_epoch = {
            'epoch': df.iloc[i][0],
            'train_D_adv_loss': df.iloc[i][1],
            'train_G_adv_loss': df.iloc[i][2],
            'train_G_l1_loss': df.iloc[i][3],
            'val_G_l1_loss': df.iloc[i][4],
            'train_psnr': df.iloc[i][5],
            'val_psnr': df.iloc[i][6],
        }
        log.append(log_epoch)
        
    return log


def psnr(inp, real, r=255.):
    mse = ((inp - real) ** 2).mean()
    psnr = 10 * math.log10(r**2 / mse)
    return psnr


def picture_resize(img, size=320):
    h, w, c = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    if h<w:
        top = (w-h) // 2
        bottom = (w-h) - top
        img = np.vstack([img[-top:, :, :], img, img[:bottom, :, :]])
    elif h>w:
        left = (h-w) // 2
        right = (h-w) - left
        img = np.hstack([img[:, -right:, :], img, img[:, :left, :]])

    img = cv2.resize(img, (size, size))

    return img, (h, w)


def set_aspect(img, org_size, size=320):
    h, w = org_size
    if h<w:
        x = size - int(size * h/w)
        top =  x // 2
        bottom = x - top
        img = img[top:-bottom, :, :]
    else:
        x = size - int(size * w/h)
        left = x // 2
        right = x - left
        img = img[:, left:-right, :]

    return img


def preprocess_generator(rgb_tensor, device):
    RGB2YCrCb = np.array([[0.299, 0.587, 0.114],
                        [0.5, -0.418688, -0.081312],
                        [-0.168736, -0.331264, 0.5]], np.float32)
    RGB2YCrCb = torch.as_tensor(RGB2YCrCb.reshape(3, 3, 1, 1)).to(device)
    x = nn.functional.conv2d(rgb_tensor, RGB2YCrCb)  # Y:0~1 CbCr:-0.5~0.5
    x *= 2.0
    x[:, 0,:,:] -= 1.0  # YCbCr:-1~1
    return x


def deprocess_generator(ycrcb_tensor, device):
    YCrCb2RGB = np.array([[1, 1.402, 0],
                        [1, -0.714136, -0.344136],
                        [1, 0, 1.772]], np.float32)
    YCrCb2RGB = torch.as_tensor(YCrCb2RGB.reshape(3, 3, 1, 1)).to(device)
    x = ycrcb_tensor / 2.0
    x[:, 0,:,:] += 0.5  # Y:0~1 CbCr:-0.5~0.5
    x = nn.functional.conv2d(x, YCrCb2RGB)  # Change RGB (0~1)
    return x