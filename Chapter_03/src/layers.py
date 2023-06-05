from functions import *


# 卷积层
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
            2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros((out_channels, 1))

        self.input = None
        self.grad_weights = None
        self.grad_bias = None
        self.conv_output = None

    def forward(self, input):
        self.input = input
        self.conv_output = conv2d(input, self.weights, stride=self.stride, padding=self.padding)
        # conv_output: (B, OC, OH, OW)
        # bias: (OC, 1) -> (1, OC, 1, 1)
        output = relu(self.conv_output + self.bias.reshape((1, -1, 1, 1)))
        return output

    def backward(self, grad_output):
        # batch_size, num_channels, in_height, in_width = grad_output.shape

        grad_output = relu_backward(grad_output, self.conv_output)

        # rotated_kernel: [OC, IC, KS, KS] -> [IC, OC, KS, KS]
        # grad_output: [B, OC, OH, OW]
        # grad_input: [IC, B, H, W] -> [B, IC, H, W]
        rotated_kernel = rot180(self.weights)
        rotated_kernel = np.transpose(rotated_kernel, (1, 0, 2, 3))
        grad_input = full_conv(rotated_kernel, grad_output)
        grad_input = np.transpose(grad_input, (1, 0, 2, 3))

        # input: [B, IC, H, W] -> [IC, B, H, W]
        # grad_output: [B, OC, OH, OW] -> [OC, B, OH, OW]
        # grad_weights: [IC, OC, KS, KS] -> [OC, IC, KS, KS]
        self.grad_weights = conv2d(np.transpose(self.input, (1, 0, 2, 3)), np.transpose(grad_output, (1, 0, 2, 3)),
                                   padding=self.padding)
        self.grad_weights = np.transpose(self.grad_weights, (1, 0, 2, 3))

        # grad_output: [B, OC, OH, OW] -> [OC, 1]
        self.grad_bias = np.sum(grad_output, axis=(0, 2, 3)).reshape(self.out_channels, 1)
        return grad_input


# 平均汇聚层
class AvgPool:
    def __init__(self, kernel_size, stride=1):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, num_channels, in_height, in_width = x.shape
        out_height = int((in_height - self.kernel_size) / self.stride) + 1
        out_width = int((in_width - self.kernel_size) / self.stride) + 1

        output = np.zeros((batch_size, num_channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(num_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size

                        receptive_field = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h_out, w_out] = np.mean(receptive_field)

        return output

    # average pooling layer backward
    def backward(self, grad_output):
        batch_size, num_channels, out_height, out_width = grad_output.shape
        in_height = int((out_height - 1) * self.stride + self.kernel_size)
        in_width = int((out_width - 1) * self.stride + self.kernel_size)

        grad_input = np.zeros((batch_size, num_channels, in_height, in_width))

        for b in range(batch_size):
            for c in range(num_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size

                        grad_input[b, c, h_start:h_end, w_start:w_end] = grad_output[b, c, h_out, w_out] / (
                                self.kernel_size * self.kernel_size)

        return grad_input


# 展平
class Flatten:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input.shape)


# 全连接层
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
        self.bias = np.zeros((1, out_features))

        self.input = None
        self.grad_weights = None
        self.grad_bias = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward(self, grad_output):
        # grad_weights = input.T * grad_output
        # input: [B, I] -> [I, B]
        # grad_output: [B, O]
        self.grad_weights = np.dot(self.input.T, grad_output)

        # grad_bias
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # grad_input = grad_output * weights.T
        # grad_output: [B, O]
        # weights: [I, O] -> [O, I]
        # grad_input: [B, I]
        return np.dot(grad_output, self.weights.T)
