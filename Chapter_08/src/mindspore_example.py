# 导入模块
import os
# os.environ['DEVICE_ID'] = '0'
import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
context.set_context(mode=context.GRAPH_MODE,
                    device_target='Ascend')  # Ascend, CPU, GPU


def create_dataset(data_dir, training=True,
                   batch_size=32, resize=(32, 32),
                   rescale=1/(255*0.3081), shift=-0.1307/0.3081,
                   buffer_size=64):
    '''
    数据处理
    '''
    data_train = os.path.join(data_dir, 'train')  # train set
    data_test = os.path.join(data_dir, 'test')  # test set
    ds = ms.dataset.MnistDataset(
        data_train if training else data_test)
    ds = ds.map(input_columns=["image"],
                operations=[CV.Resize(resize),
                            CV.Rescale(rescale, shift),
                            CV.HWC2CHW()])
    ds = ds.map(input_columns=["label"],
                operations=C.TypeCast(ms.int32))
    ds = ds.shuffle(buffer_size=buffer_size).batch(
        batch_size, drop_remainder=True)
    return ds


class LeNet5(nn.Cell):
    '''
    定义模型
    '''

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, pad_mode='valid')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(data_dir, lr=0.01, momentum=0.9, num_epochs=3):
    '''
    训练
    '''
    ds_train = create_dataset(data_dir)
    ds_eval = create_dataset(data_dir, training=False)
    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(
        sparse=True, reduction='mean')
    opt = nn.Momentum(
        net.trainable_params(), lr, momentum)
    loss_cb = LossMonitor(
        per_print_times=ds_train.get_dataset_size())
    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    # dataset_sink_mode can be True when using Ascend
    model.train(num_epochs, ds_train,
                callbacks=[loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)


# 此处修改为数据的挂载路径
train('data/lenet/lenet5/MNIST/')
