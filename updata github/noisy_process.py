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


def airfoil_plot_double(y, air_noise, n_points, name):
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if not isinstance(air_noise, np.ndarray):
        air_noise = np.array(air_noise)

    y_up = y[:, 0]
    y_low = y[:, 1]

    noise_up = air_noise[:, 0]
    noise_low = air_noise[:, 1]

    y_up = y_up.reshape(-1, 1)
    y_low = y_low.reshape(-1, 1)
    noise_up = noise_up.reshape(-1, 1)
    noise_low = noise_low.reshape(-1, 1)

    # 生成x坐标
    air_x = get_x_axis(n_points)
    x = air_x

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制原始翼型图像
    axs[0].plot(x, y_up, linewidth=3, label='Original Upper Surface')
    axs[0].plot(x, y_low, linewidth=3, label='Original Lower Surface')
    axs[0].set_title("Original Airfoil")
    axs[0].set_ylim(-0.15, 0.15)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    # 绘制噪声翼型图像
    axs[1].plot(x, noise_up, linewidth=3, label='Noisy Upper Surface')
    axs[1].plot(x, noise_low, linewidth=3, label='Noisy Lower Surface')
    axs[1].set_title("Noisy Airfoil")
    axs[1].set_ylim(-0.15, 0.15)
    axs[1].legend(fontsize=15)
    axs[1].grid(True)


    plt.tight_layout()
    #plt.show()
    plt.savefig(name, dpi=300)


def airfoil_plot(airfoil, n_points, filename):

    if not isinstance(airfoil, np.ndarray):
        airfoil = np.array(airfoil)

    y_up = airfoil[:, 0]
    y_low = airfoil[:, 1]

    y_up = y_up.reshape(-1, 1)
    y_low = y_low.reshape(-1, 1)


    # 生成x坐标
    air_x = get_x_axis(n_points)
    x = air_x

    # 绘制翼型图像
    plt.figure()
    #plt.axis("equal")

    plt.plot(x, y_up, '-o', markersize=3)
    plt.plot(x, y_low, '-o', markersize=3)
    #print(plt)

    plt.ylim(-0.2, 0.2)
    plt.legend().set_visible(False)
    plt.title("Airfoil")
    plt.grid(True)
    #plt.axis('off')

    #plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

def airfoil_plot1(airfoil, n_points):

    if not isinstance(airfoil, np.ndarray):
        airfoil = np.array(airfoil)

    y_up = airfoil[:, 0]
    y_low = airfoil[:, 1]

    y_up = y_up.reshape(-1, 1)
    y_low = y_low.reshape(-1, 1)


    # 生成x坐标
    air_x = get_x_axis(n_points)
    x = air_x

    # 绘制翼型图像
    plt.figure()
    plt.axis("equal")

    plt.plot(x, y_up, color='C0', linewidth=3)
    plt.plot(x, y_low, color='C0', linewidth=3)
    #print(plt)

    plt.legend().set_visible(False)
    plt.title("Noisy Airfoil")
    plt.axis('off')

    #plt.savefig('noisy_air1.png', dpi=300)
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


    # 将两组数据拼接成 (128, 2) 数组
    x = np.column_stack((x_up, x_low))
    y = np.column_stack((y_up, y_low))

    return x, y



path1 = 'airfoil_23.dat'
x1, y = read_and_process_dat(path1, 128)





# 超参数
device = 'cpu'
timesteps = 666  #这里不能小于20  会nan
noise_factor1 = 0.05
noise_factor2 = noise_factor1 / 3
noise_factor3 = noise_factor2 / 5

airfoil = torch.tensor(y, device=device, dtype=torch.float32)
airfoil_plot(airfoil, airfoil.shape[0], '1')


gaussian = GaussianDiffusion(noise_factor1, noise_factor2, timesteps=timesteps)



t = torch.full((1,), 100, device=device).long()
print(t)


x_start = airfoil
boundary_point1 = 5
boundary_point2 = 15


mask1 = torch.zeros_like(x_start)
mask2 = torch.zeros_like(x_start)
mask3 = torch.zeros_like(x_start)

mask1[:boundary_point1] = 1
mask2[boundary_point1:boundary_point2] = 1
mask3[boundary_point2:] = 1

# random noise ~ N(0, 1)
noise_middle = torch.randn_like(x_start) * noise_factor1
noise_front = torch.randn_like(x_start) * noise_factor2
noise_head = torch.randn_like(x_start) * noise_factor3

noise = noise_head * mask1 + noise_front * mask2 + noise_middle * mask3
x_noisy = gaussian.q_sample(x_start, t, noise)  # x_t ~ q(x_t | x_0)



airfoil_plot(x_noisy, x_noisy.shape[0], '1')


