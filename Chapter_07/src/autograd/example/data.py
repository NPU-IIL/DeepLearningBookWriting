from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.image

import autograd.numpy as np
import autograd.numpy.random as npr
import data_mnist

def load_mnist():
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))  # 定义一个函数，用于将数组进行部分扁平化
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)  # 定义一个函数，用于进行one-hot编码
    train_images, train_labels, test_images, test_labels = data_mnist.mnist()  # 调用data_mnist模块中的mnist函数加载MNIST数据集
    train_images = partial_flatten(train_images) / 255.0  # 对训练图像数据进行部分扁平化并归一化
    test_images  = partial_flatten(test_images)  / 255.0  # 对测试图像数据进行部分扁平化并归一化
    train_labels = one_hot(train_labels, 10)  # 对训练标签进行one-hot编码
    test_labels = one_hot(test_labels, 10)  # 对测试标签进行one-hot编码
    N_data = train_images.shape[0]  # 获取训练图像数据的样本数

    return N_data, train_images, train_labels, test_images, test_labels # 返回加载和处理后的数据


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """图片格式应该设置为(图片数量N_images x 像素pixels)的矩阵."""
    N_images = images.shape[0]  # 获取图像数据的样本数
    N_rows = (N_images - 1) // ims_per_row + 1  # 计算需要的行数
    pad_value = np.min(images.ravel())  # 获取图像数据中的最小值作为填充值
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)  # 创建一个填充值为pad_value的拼接图像
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)  # 将一维图像数据转换为二维图像
        row_ix = i // ims_per_row  # 计算当前图像所在的行索引
        col_ix = i % ims_per_row  # 计算当前图像所在的列索引
        row_start = padding + (padding + digit_dimensions[0]) * row_ix  # 计算当前图像在拼接图像中的行起始位置
        col_start = padding + (padding + digit_dimensions[1]) * col_ix  # 计算当前图像在拼接图像中的列起始位置
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image  # 将当前图像复制到拼接图像中的相应位置
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)  # 在指定的Axes对象上绘制拼接图像
    plt.xticks(np.array([]))  # 设置X轴刻度为空
    plt.yticks(np.array([]))  # 设置Y轴刻度为空
    return cax  # 返回图像的Colorbar对象

def save_images(images, filename, **kwargs):
    fig = plt.figure(1)  # 创建一个新的Figure对象
    fig.clf()  # 清空Figure对象中的所有内容
    ax = fig.add_subplot(111)  # 在Figure对象上添加一个Axes对象
    plot_images(images, ax, **kwargs)  # 调用plot_images函数绘制拼接图像
    fig.patch.set_visible(False)  # 设置Figure对象的背景不可见
    ax.patch.set_visible(False)  # 设置Axes对象的背景不可见
    plt.savefig(filename)  # 将Figure对象保存为图像文件


def make_pinwheel(radial_std, tangential_std, num_classes, num_per_class, rate,
                  rs=npr.RandomState(0)):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)  # 在0到2π之间均匀生成num_classes个角度值

    features = rs.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])  # 生成符合正态分布的特征向量
    features[:, 0] += 1  # 对特征向量的第一个维度进行偏移
    labels = np.repeat(np.arange(num_classes), num_per_class)  # 生成重复的标签

    angles = rads[labels] + rate * np.exp(features[:,0])  # 根据标签和特征计算角度
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])  # 构建旋转矩阵
    rotations = np.reshape(rotations.T, (-1, 2, 2))  # 调整旋转矩阵的形状

    return np.einsum('ti,tij->tj', features, rotations)  # 返回生成的数据
