# Airfoil-DDPM
 
 环境依赖
Python >= 3.8

PyTorch >= 1.10

NumPy, Matplotlib, SciPy, tqdm 等


# Airfoil-DDPM

本项目实现了一种基于 **去噪扩散概率模型（DDPM）** 的翼型生成方法，可用于生成几何结构连续、气动性能良好的翼型轮廓。该模型无需参数化函数支持，直接在二维坐标点空间中进行建模与采样，适用于翼型智能设计任务。

## 项目亮点

- 基于 DDPM 的逐步去噪生成机制
- 引入一维 UNet 架构，支持时间步嵌入与注意力机制
- 针对翼型上下表面设计结构对称的生成流程
- 生成结果具有良好的几何连续性和气动可行性

## 项目结构说明

| 文件名 | 功能描述 |
|--------|----------|
| `train.py` | 训练主脚本，加载训练数据并训练 UNet-DDPM 模型 |
| `generate_test.py` | 用于从纯噪声中采样生成翼型，并可视化生成结果 |
| `ddpm.py` | DDPM 核心模块，实现前向加噪与反向去噪过程 |
| `unet_1D_test.py` | 一维UNet结构及注意力模块定义 |
| `noisy_process.py` | 测试前向加噪过程，并可视化原始/加噪翼型对比图 |
| `airfoil_draw.py` / `data_loader.py` | 用于绘图与数据处理的辅助工具模块 |

## 使用说明

### 训练模型

```bash
python train.py

请确保已有训练数据 train_data.npy 文件，包含 shape 为 (样本数, 128, 2) 的翼型坐标点数据


### 测试
python generate_test.py   做相应修改


