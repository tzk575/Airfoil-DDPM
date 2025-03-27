import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载数据
data = np.load('train_data.npy')  # 替换为你的文件路径
N, L, C = data.shape
data = data.transpose(0, 2, 1).astype(np.float32)

# 划分训练集和验证集
X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)


# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * L, 128)
        self.fc2 = nn.Linear(128, 2)  # 二分类任务

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    inputs = torch.from_numpy(X_train)
    targets = torch.from_numpy(np.random.randint(0, 2, size=(X_train.shape[0],))).long()  # 转换为 Long 类型

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


# 可视化卷积层的特征图，聚焦于前缘部分
def plot_front_edge_features(conv_output, title, num_points=10):
    num_filters = conv_output.shape[1]
    fig, axs = plt.subplots(num_filters // 4, 4, figsize=(15, 15))
    for i in range(num_filters):
        ax = axs[i // 4, i % 4]
        ax.plot(conv_output[0, i, :num_points].detach().numpy())
        ax.set_title(f'Filter {i + 1}')
    plt.suptitle(title)
    plt.show()


# 获取前缘部分特征
model.eval()
with torch.no_grad():
    val_inputs = torch.from_numpy(X_val)
    conv1_output = model.conv1(val_inputs)
    plot_front_edge_features(conv1_output, "Conv1 Front Edge Features")

    conv2_output = model.conv2(F.relu(conv1_output))
    plot_front_edge_features(conv2_output, "Conv2 Front Edge Features")


# 计算前缘部分的统计指标（例如均值和方差）
def compute_statistics(conv_output, num_points=10):
    mean = torch.mean(conv_output[:, :, :num_points], dim=[0, 2]).detach().numpy()
    std_dev = torch.std(conv_output[:, :, :num_points], dim=[0, 2]).detach().numpy()
    return mean, std_dev


conv1_mean, conv1_std = compute_statistics(conv1_output)
conv2_mean, conv2_std = compute_statistics(conv2_output)

print("Conv1 Front Edge Mean:", conv1_mean)
print("Conv1 Front Edge Std Dev:", conv1_std)
print("Conv2 Front Edge Mean:", conv2_mean)
print("Conv2 Front Edge Std Dev:", conv2_std)
