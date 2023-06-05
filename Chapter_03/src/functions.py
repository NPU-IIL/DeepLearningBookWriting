import os.path

import numpy as np


def load_mnist(data_path):
    # 加载训练数据
    with open(os.path.join(data_path, 'train-images.idx3-ubyte'), 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape((-1, 1, 28, 28))

    with open(os.path.join(data_path, 'train-labels.idx1-ubyte'), 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # 加载测试数据
    with open(os.path.join(data_path, 't10k-images.idx3-ubyte'), 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape((-1, 1, 28, 28))

    with open(os.path.join(data_path, 't10k-labels.idx1-ubyte'), 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # 添加填充
    train_images = np.pad(train_images, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
    test_images = np.pad(test_images, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')

    # labels转换为one-hot
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    return (train_images, train_labels), (test_images, test_labels)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def relu_backward(grad_output, input):
    return grad_output * (input > 0)


def rot180(kernel):
    return np.rot90(kernel, 2, axes=(2, 3))


def cross_entropy_loss(y, t):
    # 批量平均损失
    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]


def softmax_backward(y, t):
    return (y - t) / y.shape[0]


def conv2d(input, kernel, *, padding=0, stride=1):
    batch_size, num_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape

    out_height = int((in_height - kernel_height + 2 * padding) / stride) + 1
    out_width = int((in_width - kernel_width + 2 * padding) / stride) + 1

    output = np.zeros((batch_size, out_channels, out_height, out_width))

    padded_input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

    for b in range(batch_size):
        for c in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    h_start = h_out * stride
                    h_end = h_start + kernel_height
                    w_start = w_out * stride
                    w_end = w_start + kernel_width

                    receptive_field = padded_input[b, :, h_start:h_end, w_start:w_end]
                    output[b, c, h_out, w_out] = np.sum(receptive_field * kernel[c])

    return output


def full_conv(input, kernel):
    batch_size, num_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape

    out_height = in_height + kernel_height - 1
    out_width = in_width + kernel_width - 1

    output = np.zeros((batch_size, out_channels, out_height, out_width))

    for b in range(batch_size):
        for c in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    # h_start: (in_height - 1) -> (in_height - out_height)
                    h_start = in_height - h_out - 1
                    h_end = h_start + kernel_height
                    # w_start: (in_width - 1) -> (in_width - out_width)
                    w_start = in_width - w_out - 1
                    w_end = w_start + kernel_width

                    top_padding = None
                    down_padding = None
                    left_padding = None
                    right_padding = None

                    if h_start < 0:
                        top_padding = -h_start
                        h_start = 0
                    if h_end > in_height:
                        down_padding = h_end - in_height
                        h_end = in_height
                    if w_start < 0:
                        left_padding = -w_start
                        w_start = 0
                    if w_end > in_width:
                        right_padding = w_end - in_width
                        w_end = in_width

                    receptive_field = input[b, :, h_start:h_end, w_start:w_end]
                    # pad receptive field to kernel size: [1, IC, KS, KS]
                    if top_padding is not None:
                        receptive_field = np.pad(receptive_field, ((0, 0), (top_padding, 0), (0, 0)), 'constant')
                    if down_padding is not None:
                        receptive_field = np.pad(receptive_field, ((0, 0), (0, down_padding), (0, 0)), 'constant')
                    if left_padding is not None:
                        receptive_field = np.pad(receptive_field, ((0, 0), (0, 0), (left_padding, 0)), 'constant')
                    if right_padding is not None:
                        receptive_field = np.pad(receptive_field, ((0, 0), (0, 0), (0, right_padding)), 'constant')

                    output[b, c, h_out, w_out] = np.sum(receptive_field * kernel[c])

    return output

