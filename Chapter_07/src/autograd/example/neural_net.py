"""一个用于对MNIST手写数字进行分类的多层感知器."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam
from data import load_mnist


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    # 初始化参数矩阵和偏置向量，并返回一个参数列表
    return [(scale * rs.randn(m, n),   # 权重矩阵
             scale * rs.randn(n))      # 偏置向量
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    # 前向传播，计算神经网络的输出
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def l2_norm(params):
    # 将参数展平为向量，计算其L2范数
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    # 计算对数后验概率，包括先验项和似然项
    log_prior = -L2_reg * l2_norm(params)  # 先验项
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)  # 似然项
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    # 计算模型在给定数据上的准确率
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == '__main__':
    # 模型参数
    layer_sizes = [784, 200, 100, 10]  # 每层的神经元数量
    L2_reg = 1.0  # L2正则化系数

    # 训练参数
    param_scale = 0.1  # 参数初始缩放系数
    batch_size = 256  # 批量大小
    num_epochs = 20  # 训练迭代次数
    step_size = 0.001  # 学习率

    print("加载训练数据...")
    # 加载MNIST数据集
    N, train_images, train_labels, test_images, test_labels = load_mnist()
    # 初始化每层参数
    init_params = init_random_params(param_scale, layer_sizes)
    # 设置每批次数据大小
    num_batches = int(np.ceil(len(train_images) / batch_size))

    def batch_indices(iter):
        # 根据迭代次数返回批次索引
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # 定义训练目标
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)

    # 使用自动微分计算目标函数的梯度
    objective_grad = grad(objective)

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")
    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, train_images, train_labels)
            test_acc = accuracy(params, test_images, test_labels)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))

    # 使用Adam优化算法优化参数
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)
