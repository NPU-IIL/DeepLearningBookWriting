# 导入依赖包
import mindspore
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.py_transforms as P

# 构建训练集和测试集


class CustomDataset():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, 100)

train_data = CustomDataset(data[:80], labels[:80])
test_data = CustomDataset(data[80:], labels[80:])

train_ds = ds.GeneratorDataset(train_data, ["data", "label"])
test_ds = ds.GeneratorDataset(test_data, ["data", "label"])

train_ds = train_ds.map(input_columns="data",
                        operations=C.TypeCast(mstype.float32))
train_ds = train_ds.map(input_columns="label",
                        operations=C.TypeCast(mstype.int32))

test_ds = test_ds.map(input_columns="data",
                      operations=C.TypeCast(mstype.float32))
test_ds = test_ds.map(input_columns="label",
                      operations=C.TypeCast(mstype.int32))

train_loader = train_ds.batch(batch_size=32, drop_remainder=True)
test_loader = test_ds.batch(batch_size=32, drop_remainder=True)


# 误差分析

# 假设有 10 个样本和 3 个类别
labels = np.random.randint(0, 3, size=(10,))
predictions = np.random.randint(0, 3, size=(10,))

# 转换为 MindSpore Tensor
labels_tensor = mindspore.Tensor(labels, mindspore.int32)
predictions_tensor = mindspore.Tensor(predictions, mindspore.int32)

# 计算混淆矩阵
confusion_matrix = mindspore.ops.operations.ConfusionMatrix(num_classes=3)
confusion_matrix.update(labels_tensor, predictions_tensor)
confusion_matrix = confusion_matrix.eval()

# 计算性能指标
accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()
precision = confusion_matrix.diag() / confusion_matrix.sum(axis=1)
recall = confusion_matrix.diag() / confusion_matrix.sum(axis=0)
f1_score = 2 * precision * recall / (precision + recall)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score}")
