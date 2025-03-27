
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def airfoil_double_plot(airfoil1, airfoil2, air_x):

    device = airfoil1.device
    if device != 'cpu':
        airfoil1 = airfoil1.cpu()
        airfoil2 = airfoil2.cpu()


    if not isinstance(airfoil1, np.ndarray):
        airfoil1 = np.array(airfoil1)
    if not isinstance(airfoil2, np.ndarray):
        airfoil2 = np.array(airfoil2)




    #airfoil = airfoil.to('cpu')
    y1_up = airfoil1[:, 0]
    y1_low = airfoil1[:, 1]

    y1_up = y1_up.reshape(-1, 1)
    y1_low = y1_low.reshape(-1, 1)


    y2_up = airfoil2[:, 0]
    y2_low = airfoil2[:, 1]

    y2_up = y2_up.reshape(-1, 1)
    y2_low = y2_low.reshape(-1, 1)


    # 生成x坐标
    x = air_x


    # 绘制翼型图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(x, y1_up, label="Upper Surface")
    ax1.plot(x, y1_low, label="Lower Surface")
    ax1.set_title('Airfoil 1')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, y2_up, label="Upper Surface")
    ax2.plot(x, y2_low, label="Lower Surface")
    ax2.set_title('Airfoil 2')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True)

    # #plt.ylim(-0.3, 0.3)
    # plt.xlabel("x/c")
    # plt.ylabel("y/c")
    # plt.title("Generated Airfoil")
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close()












