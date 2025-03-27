import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)



class SpatialAttention(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channel, channel, kernel_size, padding=padding, bias=False)  # 这里假设输入是1D数据
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        return x * attention.expand_as(x), attention



def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# 两个一维卷积+时间信息嵌入
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super(UNetBlock, self).__init__()
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        # Spatial attention
        self.attention = SpatialAttention(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        # Linear layer for time embedding
        self.time_fc = nn.Linear(time_embed_dim, out_channels)

    def forward(self, x, t_emb):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        t = self.time_fc(t_emb)  # [batch_size, out_channels]
        t = t[:, :, None]  # Add a fake spatial dimension: [batch_size, out_channels, 1]

        x = x + t

        x, attention_weights = self.attention(x)
        x = self.conv3(x)

        return x, attention_weights


class Downsample(nn.Module):
    # 下采样模块
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.op = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    # 上采样模块
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.upsample(x)
        #return x



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super(UNet, self).__init__()
        self.time_embed_dim = time_embed_dim

        # 下采样
        self.down_block = nn.Sequential(
            UNetBlock(in_channels, 64, time_embed_dim),
            Downsample(64),
            UNetBlock(64, 128, time_embed_dim),
            Downsample(128),
        )

        # 中间块（不进行下采样或上采样）
        self.middle_block = UNetBlock(128, 128, time_embed_dim)

        # 上采样
        self.up_block = nn.Sequential(
            Upsample(256),
            UNetBlock(256, 64, time_embed_dim),
            Upsample(128),
            UNetBlock(128, 64, time_embed_dim),
        )


        # 输出层
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        # 时间嵌入
        t_emb = timestep_embedding(t, self.time_embed_dim)
        layer_list = []
        attention_weights_list = []
        l = x.permute(0, 2, 1)


    # 下采样
        for module in self.down_block:
            if isinstance(module, UNetBlock):
                l, att_weights = module(l, t_emb)
            else:
                l = module(l)
            layer_list.append(l)

        # 中间层
        l, att_weights = self.middle_block(l, t_emb)

        # 上采样   //附带跳跃连接
        for module in self.up_block:
            cat_in = torch.cat([l, layer_list.pop()], dim=1)
            if isinstance(module, UNetBlock):
                l, att_weights = module(cat_in, t_emb)
            else:
                l = module(cat_in)


        # 输出层
        result = self.final_conv(l)

        return result.permute(0, 2, 1)


def attention_draw(attention_weights_list, sample_idx, attention_layer_idx):
    # 输入的时候是2通道的  经过卷积层之后通道数改变  所以选择其中一个通道即可
    # Extracting attention weights for the specified sample and layer
    attention_weights = attention_weights_list[attention_layer_idx][sample_idx]


    # Detaching and moving the tensor to CPU for plotting
    attention_weights = attention_weights.detach().cpu().numpy().squeeze()

    # Plotting
    plt.figure(figsize=(10, 4))  # Optionally setting the figure size
    plt.plot(attention_weights[:, 2], label='Attention Weights')
    plt.title('Attention Weights across the Airfoil at Layer {}'.format(attention_layer_idx + 1))
    plt.xlabel('Position along the airfoil')
    plt.ylabel('Attention intensity')
    plt.legend()
    plt.grid(True)  # Optionally adding a grid for easier visualization
    plt.show()




