
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ddpm import GaussianDiffusion
from unet_1D_test import UNet
from data_loader import new_airfoil, get_x_axis, new_dataset
import os



def get_x_axis(num):
    x_n = num
    theta = np.zeros([x_n + 1])
    x_newdata = np.zeros(x_n + 1)
    for i in range(1, x_n + 1):
        theta[i] = np.pi * (i - 1) / x_n
        x_newdata[i] = 1 - np.cos(theta[i])
    x_new = x_newdata[1:] / 2
    return x_new


def airfoil_plot(airfoil, n_points, index=1, save_path='.'):

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

    plt.plot(x, y_up, '-o', label="Upper Surface", markersize=4)
    plt.plot(x, y_low, '-o', label="Lower Surface", markersize=4)

    plt.ylim(-0.25, 0.25)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil")
    plt.legend()
    plt.grid(True)

    filename = f"{save_path}/{index}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(filename, dpi=300)

    #plt.show()
    plt.close()


device = 'cpu'
timesteps = 666  #这里不能小于20  会nan
batch_size = 2048
noise_factor = 0.05
noise_factor_front = noise_factor / 3
noise_factor_head = noise_factor_front / 5



gaussian = GaussianDiffusion(noise_factor, noise_factor_front, noise_factor_head, timesteps=timesteps)
unet = UNet(2, 2, 16).to(device)
unet.load_state_dict(torch.load('unet_epochs_500_npoint_128_0.05_mask.pth', map_location=device))
unet.eval()



#以下测试从纯噪声开始生成翼型的效果
shape = (batch_size, 128, 2)
noise = torch.randn(shape) * noise_factor
x_t = noise
new_air_list = []


res = gaussian.sample(unet, batch_size, 128, device)[-1]


np.save('generate_airfoil_500.npy', res)
for i in range(res.shape[0]):
    airfoil = res[i]
    airfoil_plot(airfoil, airfoil.shape[0], i, './sample/501')


