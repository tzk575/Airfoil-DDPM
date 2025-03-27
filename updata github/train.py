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
from unet_1D_test import timestep_embedding, UNetBlock, UNet
from data_loader import new_airfoil, get_x_axis, new_dataset
from airfoil_draw import airfoil_plot



# 超参数
device = 'cuda:0'
path = 'train_data.npy'
timesteps = 666  #这里不能小于20  会nan
batch_size = 32
learning_rate = 5e-4
epochs = 501
num_interval = 1
noise_factor = 0.05
noise_factor_front = noise_factor / 3
noise_factor_head = noise_factor_front / 5
regularity_coefficient = 2



data = np.load(path)
data = torch.tensor(data, dtype=torch.float32)
newdata = new_dataset(data, num_interval)
dataset = TensorDataset(newdata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


gaussian = GaussianDiffusion(noise_factor, noise_factor_front, noise_factor_head, timesteps=timesteps)
unet = UNet(2, 2, 16).to(device)


gaussian.train(unet, learning_rate, epochs, dataloader, device, batch_size, regularity_coefficient)



