# 导入依赖包
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 构建训练集和测试集


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx])
        y = torch.Tensor(self.labels[idx])
        return x, y


data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, 100)

train_data = CustomDataset(data[:80], labels[:80])
test_data = CustomDataset(data[80:], labels[80:])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# 误差分析

# 假设有 10 个样本和 3 个类别
labels = np.random.randint(0, 3, size=(10,))
predictions = np.random.randint(0, 3, size=(10,))

# 转换为 PyTorch Tensor
labels_tensor = torch.from_numpy(labels)
predictions_tensor = torch.from_numpy(predictions)

# 计算混淆矩阵
confusion_matrix = torch.zeros(3, 3)
for t, p in zip(labels_tensor, predictions_tensor):
    confusion_matrix[t, p] += 1

# 计算性能指标
accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()
precision = confusion_matrix.diag() / confusion_matrix.sum(dim=0)
recall = confusion_matrix.diag() / confusion_matrix.sum(dim=1)
f1_score = 2 * precision * recall / (precision + recall)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
