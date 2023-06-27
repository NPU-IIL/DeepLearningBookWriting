"""基于“LeNet-5”模型,在MNIST数据集上实现了一个卷积神经网络."""
from __future__ import absolute_import
from __future__ import print_function
from builtins import range

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.signal
from autograd import grad
import data_mnist

convolve = autograd.scipy.signal.convolve

class WeightsParser(object):
    """定义用于索引参数向量的辅助类."""
    def __init__(self):
        self.idxs_and_shapes = {} # 存储参数索引和形状的字典
        self.N = 0 # 参数向量的总长度

    def add_weights(self, name, shape):
        start = self.N # 当前参数在参数向量中的起始索引
        self.N += np.prod(shape) # 更新参数向量的总长度
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)  # 将参数名称、切片索引和形状存储在字典中

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape) # 根据切片索引和形状重塑参数向量

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))  # 使用切片索引表示每个批次的起始和结束索引
        start += N_batch  # 更新起始索引，移动到下一个批次的起始位置
    return batches

def logsumexp(X, axis, keepdims=False):
    max_X = np.max(X)  # 找到输入数组的最大值
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=keepdims))  # 计算对数和的指数

def make_nn_funs(input_shape, layer_specs, L2_reg):
    parser = WeightsParser()  # 创建参数解析器对象
    cur_shape = input_shape  # 当前层的输入形状
    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)  # 构建当前层的权重字典并获取权重数量和输出形状
        parser.add_weights(layer, (N_weights,))  # 将权重字典添加到参数解析器中

    def predictions(W_vect, inputs):
        """输出归一化的对数概率。
        输入的形状为：[数据,颜色,y,x]"""
        cur_units = inputs  # 当前层的输出
        for layer in layer_specs:
            cur_weights = parser.get(W_vect, layer)  # 从参数解析器中获取当前层的权重
            cur_units = layer.forward_pass(cur_units, cur_weights)  # 前向传播计算当前层的输出
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)  # 计算L2正则化的先验项
        log_lik = np.sum(predictions(W_vect, X) * T)  # 计算对数似然项
        return - log_prior - log_lik  # 返回损失函数值

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(pred_fun(W_vect, X), axis=1))  # 计算错误率

    return parser.N, predictions, loss, frac_err

class conv_layer(object):
    def __init__(self, kernel_shape, num_filters):
        self.kernel_shape = kernel_shape  # 卷积核形状
        self.num_filters = num_filters  # 卷积核数量

    def forward_pass(self, inputs, param_vector):
        # 输入维度：[数据, 输入通道数, y, x]
        # 参数维度：[输入通道数, 输出通道数, y, x]
        # 输出维度：[数据, 输出通道数, y, x]
        params = self.parser.get(param_vector, 'params')  # 获取参数向量中的卷积参数
        biases = self.parser.get(param_vector, 'biases')  # 获取参数向量中的偏置项
        conv = convolve(inputs, params, axes=([2, 3], [2, 3]), dot_axes=([1], [0]), mode='valid')  # 执行卷积操作
        return conv + biases  # 返回卷积结果加上偏置项

    def build_weights_dict(self, input_shape):
        # 输入形状：[输入通道数, y, x]（无需知道数据量）
        self.parser = WeightsParser()  # 创建参数解析器对象
        self.parser.add_weights('params', (input_shape[0], self.num_filters) + self.kernel_shape)  # 添加卷积参数到参数解析器
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))  # 添加偏置项到参数解析器
        output_shape = (self.num_filters,) + self.conv_output_shape(input_shape[1:], self.kernel_shape)  # 计算输出形状
        return self.parser.N, output_shape  # 返回参数数量和输出形状

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)  # 计算卷积输出的形状

class maxpool_layer(object):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape  # 池化窗口形状

    def build_weights_dict(self, input_shape):
        # input_shape 维度：[输入通道数, y, x]
        output_shape = list(input_shape)
        for i in [0, 1]:
            assert input_shape[i + 1] % self.pool_shape[i] == 0, "maxpool shape should tile input exactly"
            output_shape[i + 1] = input_shape[i + 1] // self.pool_shape[i]
        return 0, output_shape  # 返回参数数量为0，输出形状

    def forward_pass(self, inputs, param_vector):
        new_shape = inputs.shape[:2]
        for i in [0, 1]:
            pool_width = self.pool_shape[i]
            img_width = inputs.shape[i + 2]
            new_shape += (img_width // pool_width, pool_width)
        result = inputs.reshape(new_shape)  # 重塑输入以进行池化
        return np.max(np.max(result, axis=3), axis=4)  # 执行最大池化操作

class full_layer(object):
    def __init__(self, size):
        self.size = size  # 全连接层的大小

    def build_weights_dict(self, input_shape):
        # 输入形状是任意的（已被展平）
        input_size = np.prod(input_shape, dtype=int)  # 输入的大小（展平后）
        self.parser = WeightsParser()  # 创建参数解析器对象
        self.parser.add_weights('params', (input_size, self.size))  # 添加全连接参数到参数解析器
        self.parser.add_weights('biases', (self.size,))  # 添加偏置项到参数解析器
        return self.parser.N, (self.size,)  # 返回参数数量和输出形状

    def forward_pass(self, inputs, param_vector):
        params = self.parser.get(param_vector, 'params')  # 获取参数向量中的全连接参数
        biases = self.parser.get(param_vector, 'biases')  # 获取参数向量中的偏置项
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))  # 如果输入维度大于2，则展平输入
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)  # 执行全连接层的前向传播操作

class tanh_layer(full_layer):
    def nonlinearity(self, x):
        return np.tanh(x)  # 双曲正切激活函数

class softmax_layer(full_layer):
    def nonlinearity(self, x):
        return x - logsumexp(x, axis=1, keepdims=True)  # Softmax激活函数


if __name__ == '__main__':
    # 网络参数
    L2_reg = 1.0  # L2正则化系数
    input_shape = (1, 28, 28)  # 输入形状
    layer_specs = [conv_layer((5, 5), 6),  # 卷积层1
                   maxpool_layer((2, 2)),  # 最大池化层1
                   conv_layer((5, 5), 16),  # 卷积层2
                   maxpool_layer((2, 2)),  # 最大池化层2
                   tanh_layer(120),  # 双曲正切层
                   tanh_layer(84),  # 双曲正切层
                   softmax_layer(10)]  # Softmax层

    # 训练参数
    param_scale = 0.1  # 参数初始缩放系数
    learning_rate = 1e-3  # 学习率
    momentum = 0.9  # 动量
    batch_size = 256  # 批量大小
    num_epochs = 50  # 训练迭代次数

    # 加载和处理MNIST数据集
    print("加载训练数据...")
    add_color_channel = lambda x: x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))  # 添加颜色通道
    one_hot = lambda x, K: np.array(x[:, None] == np.arange(K)[None, :], dtype=int)  # one-hot编码
    train_images, train_labels, test_images, test_labels = data_mnist.mnist()  # 加载MNIST数据集
    train_images = add_color_channel(train_images) / 255.0  # 添加颜色通道并进行归一化
    test_images = add_color_channel(test_images) / 255.0  # 添加颜色通道并进行归一化
    train_labels = one_hot(train_labels, 10)  # 对训练标签进行one-hot编码
    test_labels = one_hot(test_labels, 10)  # 对测试标签进行one-hot编码
    N_data = train_images.shape[0]  # 训练样本数量

    # 创建神经网络函数
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
    loss_grad = grad(loss_fun)

    # 初始化权重
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale  # 初始化权重

    # 数值上检查梯度，以确保安全
    # quick_grad_check(loss_fun, W, (train_images[:50], train_labels[:50]))

    print("    Epoch      |    Train err  |   Test error  ")
    def print_perf(epoch, W):
        test_perf = frac_err(W, test_images, test_labels)  # 计算测试误差
        train_perf = frac_err(W, train_images, train_labels)  # 计算训练误差
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # 使用随机梯度下降进行训练
    batch_idxs = make_batches(N_data, batch_size)  # 创建批次索引
    cur_dir = np.zeros(N_weights)  # 当前方向

    for epoch in range(num_epochs):
        print_perf(epoch, W)  # 打印性能指标
        for idxs in batch_idxs:
            grad_W = loss_grad(W, train_images[idxs], train_labels[idxs])  # 计算梯度
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W  # 计算动量方向
            W -= learning_rate * cur_dir  # 更新权重

