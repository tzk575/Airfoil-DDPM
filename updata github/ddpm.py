import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from airfoil_draw import airfoil_plot
from unet_1D_test import attention_draw
from geometric_constraints import laplacian_smooth_loss



def losses_plot(epoch, losses):
    # 将所有GPU上的tensor移到CPU上并转换为numpy数组
    cpu_losses = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses]

    epochs = np.arange(epoch)

    plt.figure(figsize=(10, 5))
    plt.plot(cpu_losses, label='diffusion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses at Epoch {epoch}')
    plt.legend()
    plt.savefig(f'loss_plot_epoch_{epoch}.png', dpi=300)
    plt.close()

#β的变化规律
def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)



class GaussianDiffusion:
    def __init__(self, noise_factor, noise_factor_front, noise_factor_head, timesteps=1000, beta_schedule="linear"):
        self.timesteps = timesteps

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")
        self.betas = betas

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        self.noise_factor = noise_factor
        self.noise_factor_front = noise_factor_front
        self.noise_factor_head = noise_factor_head

    def _extract(self, a, t, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float() #取到a列表中的第t个数，  其实也就实现了前t个alpha相乘
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    #前向加噪
    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start) * self.noise_factor

        #self.sqrt_alphas_cumprod 是一个关于alpha的列表 也就是i=0到T
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)   #sqrt{alpha_t}
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)   #sqrt{1 - alpha_t}

        #print(1, x_start.shape)
        #print(2, sqrt_alphas_cumprod_t.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    #计算给定时间步的条件分布的均值
    def q_mean_variance(self, x_start, t):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    #计算给定时间步的条件分布的方差
    def q_posterior_mean_variance(self, x_start, x_t, t):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    #根据给定的噪声和噪声数据，推断原始数据
    def predict_start_from_noise(self, x_t, t, noise):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    #预测均值和方差
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    #采样去噪后的数据
    @torch.no_grad()
    def p_sample(self, model, x_t, t, boundary_point1=5, boundary_point2=15, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)

        mask = torch.zeros_like(x_t)
        mask1 = mask[:boundary_point1] = 1
        mask2 = mask[boundary_point1:boundary_point2] = 1
        mask3 = mask[boundary_point2:] = 1

        # random noise ~ N(0, 1)
        noise_middle = torch.randn_like(x_t) * self.noise_factor
        noise_front = torch.randn_like(x_t) * self.noise_factor_front
        noise_head = torch.randn_like(x_t) * self.noise_factor_head

        noise = noise_head * mask1 + noise_front * mask2 + noise_middle * mask3

        #noise = torch.randn_like(x_t) * self.noise_factor
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # compute x_{t-1}
        pred_air = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_air


    #生成实例
    @torch.no_grad()
    def sample(self, model, batch_size, n_points, device, channels=2):
        # denoise: reverse diffusion
        shape = (batch_size, n_points, channels)
        # start from pure noise (for each example in the batch)
        res_air = torch.randn(shape, device=device) * self.noise_factor # x_T ~ N(0, 1)
        airs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc="sampling loop time step", total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            res_air = self.p_sample(model, res_air, t)
            # print(i)
            # temp = res_air.cpu()
            # airfoil_plot(temp[0], temp[0].shape[0], i)
            # if i % 50 == 0 or i < 6:
            #     print('in the sample', res_air.shape)
            #     airfoil_plot(res_air[0], res_air[0].shape[0])

            airs.append(res_air.cpu().numpy())
        return airs



    def train_losses(self, model, x_start, t, noise_factor1, noise_factor2, noise_factor3, boundary_point1=5, boundary_point2=15):
        # compute train losses
        mask = torch.zeros_like(x_start)
        mask1 = mask[:boundary_point1] = 1
        mask2 = mask[boundary_point1:boundary_point2] = 1
        mask3 = mask[boundary_point2:] = 1

        # random noise ~ N(0, 1)
        noise_middle = torch.randn_like(x_start) * noise_factor1
        noise_front = torch.randn_like(x_start) * noise_factor2
        noise_head = torch.randn_like(x_start) * noise_factor3

        noise = noise_head * mask1 + noise_front * mask2 + noise_middle * mask3
        x_noisy = self.q_sample(x_start, t, noise)  # x_t ~ q(x_t | x_0)
        predicted_noise = model(x_noisy, t)  # predict noise from noise
        # predicted_noise = model(x_noisy, t) # predict noise from noise

        loss = F.mse_loss(noise, predicted_noise)
        return loss


    def train(self, model, learning_rate, epochs, train_loader, device, batch_size, lamuda):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        all_epoch_losses = []
        for epoch in range(epochs):
            epoch_losses = 0.0
            print(epoch, '##############')
            for step, data in enumerate(train_loader):
                optimizer.zero_grad()
                airfoil = data[0].to(device)

                t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

                loss_model = self.train_losses(model, airfoil, t, self.noise_factor,
                                               self.noise_factor_front, self.noise_factor_head)
                loss_model = torch.log1p(loss_model) * 1000


                if epoch % 5 == 0:
                    n_points = airfoil.shape[1]
                    res_air = self.sample(model, batch_size, n_points, device)[-1]
                    res_air = torch.tensor(res_air, dtype=torch.float32).to(device)


                    loss_smooth = laplacian_smooth_loss(res_air)
                    loss = loss_model + torch.mean(loss_smooth) * lamuda


                else:
                    loss = loss_model

                epoch_losses += loss
                #epoch_losses += loss.item() * 100

                loss.backward()
                optimizer.step()

            avg_loss = epoch_losses / (step + 1)
            print(type(avg_loss))
            print(f"Epoch [{epoch+1}/{epochs}]", avg_loss)
            all_epoch_losses.append(avg_loss)
        # all_epoch_losses_cpu = [loss.detach().cpu() for loss in all_epoch_losses]
        # all_epoch_losses_array = np.array([loss.numpy() for loss in all_epoch_losses_cpu])
        # np.save('epoch_losses.npy', all_epoch_losses_array)
        # torch.save(model.state_dict(),
        #            f'unet_epochs_{epochs}_npoint_{n_points}_{self.noise_factor}_attention.pth')

        # # sample
        # res_air = self.sample(model, batch_size, n_points, device)[-1]
        #
        # for i in range(res_air.shape[0]):
        #     airfoil_plot(res_air[i], n_points, i, f'./sample/{epochs}')


            save_epochs = [1, 200, 300, 500]
            if epoch in save_epochs:
                losses_plot(epoch, all_epoch_losses)
                torch.save(model.state_dict(),
                           f'unet_epochs_{epoch}_npoint_{n_points}_{self.noise_factor}_mask.pth')
                #attention_draw(attention_list, 0, 0)

                # sample
                res_air = self.sample(model, batch_size, n_points, device)[-1]

                for i in range(res_air.shape[0]):
                    airfoil_plot(res_air[i], n_points, i, f'./sample/{epoch}')





