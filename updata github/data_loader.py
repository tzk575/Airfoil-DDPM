
import numpy as np
import torch
import torch.nn as nn
from airfoil_draw import airfoil_plot
from ddpm import GaussianDiffusion
from unet_1D import UNet


def get_x_axis(num):
    x_n = num
    theta = np.zeros([x_n + 1])
    x_newdata = np.zeros(x_n + 1)
    for i in range(1, x_n + 1):
        theta[i] = np.pi * (i - 1) / x_n
        x_newdata[i] = 1 - np.cos(theta[i])
    x_new = x_newdata[1:] / 2
    return x_new




def new_airfoil(air, interval_num):
    new_air = air[::interval_num]
    return new_air


def new_dataset(data, num_interval):
    # 对整个数据集应用下采样
    return data[:, ::num_interval]

