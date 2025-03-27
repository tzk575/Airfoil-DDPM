import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ddpm import GaussianDiffusion
from tqdm import tqdm
from unet_1D_test import timestep_embedding, UNetBlock, UNet, attention_draw
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_x_axis(num):
    x_n = num
    theta = np.zeros([x_n + 1])
    x_newdata = np.zeros(x_n + 1)
    for i in range(1, x_n + 1):
        theta[i] = np.pi * (i - 1) / x_n
        x_newdata[i] = 1 - np.cos(theta[i])
        x_new = x_newdata[1:] / 2
    return x_new

def airfoil_plot(airfoil, n_points, filename):
    if not isinstance(airfoil, np.ndarray):
        airfoil = np.array(airfoil)

    y_up = airfoil[:, 0]
    y_low = airfoil[:, 1]

    y_up = y_up.reshape(-1, 1)
    y_low = y_low.reshape(-1, 1)

    air_x = get_x_axis(n_points)
    x = air_x

    plt.figure()
    plt.plot(x, y_up, '-o', markersize=3)
    plt.plot(x, y_low, '-o', markersize=3)
    plt.ylim(-0.2, 0.2)
    plt.legend().set_visible(False)
    plt.title("Airfoil")
    plt.grid(True)
    plt.show()
    plt.close()

def read_and_process_dat(file_path, num):
    data = np.loadtxt(file_path, skiprows=1)
    x_coords = data[:, 0]
    y_coords = data[:, 1]

    x_up = x_coords[:num]
    x_low = x_coords[num-1:num * 2]

    y_up = y_coords[:num]
    y_low = y_coords[num-1:num * 2]

    x_up = x_up[::-1]
    y_up = y_up[::-1]

    x = np.column_stack((x_up, x_low))
    y = np.column_stack((y_up, y_low))

    return x, y

path1 = 'airfoil_23.dat'
x1, y = read_and_process_dat(path1, 128)

# 超参数
device = 'cpu'
timesteps = 666
noise_factor = 0.05
noise_factor_front = noise_factor / 10
airfoil = torch.tensor(y, device=device, dtype=torch.float32)

gaussian = GaussianDiffusion(noise_factor, noise_factor_front, timesteps=timesteps)

#t = torch.full((1,), 10, device=device).long()
t = torch.randint(0, timesteps, (1,), device=device).long()
print(t)

# 定义掩码
boundary_point = 15
mask = torch.zeros_like(airfoil)
mask[:boundary_point] = 1


# 生成噪声
noise_front = torch.randn_like(airfoil) * noise_factor_front
noise_middle = torch.randn_like(airfoil) * noise_factor

# 应用掩码进行加噪
air_noise = noise_front * mask + noise_middle * (1 - mask)


# 应用高斯扩散模型
airfoil_noisy = gaussian.q_sample(airfoil, t, noise=air_noise)

# 绘制结果
airfoil_plot(airfoil_noisy, airfoil_noisy.shape[0], 'noisy_airfoil.png')


