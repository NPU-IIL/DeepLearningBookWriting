from layers import *


class LeNet:
    def __init__(self):
        self.conv1 = ConvLayer(in_channels=1, out_channels=6, kernel_size=5)
        self.avgpool1 = AvgPool(kernel_size=2, stride=2)
        self.conv2 = ConvLayer(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = AvgPool(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.linear1 = Linear(16 * 5 * 5, 120)
        self.linear2 = Linear(120, 84)
        self.linear3 = Linear(84, 10)

        self.loss = []

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.avgpool1.forward(x)
        x = self.conv2.forward(x)
        x = self.avgpool2.forward(x)
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = relu(x)
        x = self.linear2.forward(x)
        x = relu(x)
        x = self.linear3.forward(x)
        x = softmax(x)
        return x

    def backward(self, grad_output):
        # softmax的反向传播直接通过y-t传入即可，必须使用交叉熵损失函数
        grad_output = self.linear3.backward(grad_output)
        grad_output = relu_backward(grad_output, self.linear2.output)
        grad_output = self.linear2.backward(grad_output)
        grad_output = relu_backward(grad_output, self.linear1.output)
        grad_output = self.linear1.backward(grad_output)
        grad_output = self.flatten.backward(grad_output)
        grad_output = self.avgpool2.backward(grad_output)
        grad_output = self.conv2.backward(grad_output)
        grad_output = self.avgpool1.backward(grad_output)
        grad_output = self.conv1.backward(grad_output)
        return grad_output