import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import optuna
from optuna.trial import TrialState

class CNN(nn.Module):
    def __init__(self,dropout):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 输入(1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1,padding=1),  # output shape (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 14, 14)
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  #  (32, 7, 7)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 将数据由（32,7,7)这样的空间数据拉成一个列向量，也就是32*7*7
        y = self.fc(x)
        return y  # return x for visualization


DOWNLOAD_MNIST = False

# 下载并且加载数据集
if not os.path.exists('./mnist'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为图像张量
    transforms.Normalize((0.1307,), (0.3081,))  # 像素归一化，变成0-1分布的数据
])
train_data = datasets.MNIST(root='./mnist',
                            train=True,  # 表示训练数据
                            download=DOWNLOAD_MNIST,
                            transform=transform)
test_data = datasets.MNIST(root='./mnist',
                           train=False,  # 非训练集，即测试集
                           download=False,
                           transform=transform)  # 按照预先转变的设置对图像进行转变
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)  # 加载数据
# test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)  # 加载数据
# print(len(train_loader),len(test_loader))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test(model,test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad(): #不用计算梯度
        for (x_data,y_data) in test_loader:
            x_data = x_data.cuda()
            y_pred = model(x_data)
            _, predicted = torch.max(y_pred.cpu().data, dim=1) #预测矩阵中每一行的最大值及对应索引
            total += y_data.size(0) #y_data是一个bacch_size*10的概率结果矩阵 .size(0)就取到第一个维度的长度N
            correct += (predicted == y_data).sum().item()  #张量perdicted和y_data中对应索引上值相等时，新的张量correct_t对应位置就是1，否则为0，
    acc = 100*correct/total
    print('Accuracy on test set:%d %%' % (100*correct/total))
    return acc

def train(model,epochs,optimizer,criterion,train_loader,test_loader):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, (input,label) in enumerate(train_loader):  # 数据总量/每批训练量=最终step的值
            input = input.cuda()
            label = label.cuda()
            # print(input.size())
            output = model(input)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()  # 神经网络反向传播
            optimizer.step()  # 更新梯度，或者更新参数

            running_loss += loss.item() #当张量只有一个值时，.item()就能获取到该值，是个标量
            if batch_idx %200 == 199: #每300次训练打印一次
                print('[%d,%5d] loss: %.5f' % (epoch+1,batch_idx+1,running_loss/300))
                running_loss = 0.0
        accuracy = test(model,test_loader)
    return accuracy

def objective(trial):
    epochs = 10
    batch_size = trial.suggest_int('batch_size', 16, 32, step=16)  # batch_size开始与结束区间为16、32，步长16
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)  # 加载数据
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)  # 加载数据

    model = CNN(dropout).to(DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, (input, label) in enumerate(train_loader):  # 数据总量/每批训练量=最终step的值
            input = input.cuda()
            label = label.cuda()
            # print(input.size())
            output = model(input)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()  # 神经网络反向传播
            optimizer.step()  # 更新梯度，或者更新参数

            running_loss += loss.item()  # 当张量只有一个值时，.item()就能获取到该值，是个标量
            if batch_idx % 500 == 499:  # 每300次训练打印一次
                print('[%d,%5d] loss: %.5f' % (epoch + 1, batch_idx + 1, running_loss / 300))
                running_loss = 0.0

        accuracy = test(model, test_loader)

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == '__main__':
    print("Optuna Starting!!")
    study = optuna.create_study(direction='minimize')  # minimize
    study.optimize(objective, n_trials=1)  # n_trials 设置试验次数

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))