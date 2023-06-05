import os.path
from functions import load_mnist
from lenet import LeNet
from train import train
from test import test
from visualize import visualize_results


# 超参数设置
batch_size = 128
learning_rate = 0.005

# 加载数据
data_path = os.path.join('..', 'MNIST') # 这里改为你的数据集所在路径
print('Loading dataset...')
(train_images, train_labels), (test_images, test_labels) = load_mnist(data_path)

# 模型实例化
lenet = LeNet()

# 训练
epochs = 1

print('\nTraining...')
train(lenet, train_images, train_labels, epochs, learning_rate, batch_size)

# 测试
print('\nTesting...')
test(lenet, test_images, test_labels)

# 随机挑选9个数据进行预测，并可视化结果
visualize_results(lenet, test_images, test_labels)