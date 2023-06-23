import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, datasets
# 加载数据集
# 加载minist数据集，分成训练集和测试集，每个样本包含图像和标签
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets', x.shape, y.shape, x.min(), y.min())
# 处理数据
# 训练集图像数据归一化到0-1之前
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
# 构建数据集对象
db = tf.data.Dataset.from_tensor_slices((x, y))
# 批量训练，并行计算一次32个样本、所有数据集迭代20次
db = db.batch(32).repeat(10)
# 构建网络模型
# 构建Sequential窗口，共3层网络，前一个网络的输出作为后一个网络的输入
model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])
# 指定输入大小
model.build(input_shape=(None, 28*28))
# 打印出网络的结构和参数量
model.summary()
# optimizers用于更新梯度下降算法参数，0.01为学习率
optimizer = optimizers.SGD(lr=0.01)
# acc_meter用于计算正确率
acc_meter = keras.metrics.Accuracy()
# 创建参数文件
summary_writer = tf.summary.create_file_writer('./tf_log')
# 进行训练测试
# 循环数据集
for step, (xx, yy) in enumerate(db):
    with tf.GradientTape() as tape:
        #图像样本大小重置(-1, 28*28)
        xx = tf.reshape(xx, (-1, 28*28))
        out = model(xx)
        y_onehot = tf.one_hot(yy, depth=10)
        loss = tf.square(out-y_onehot)
        loss = tf.reduce_sum(loss/xx.shape[0])
    acc_meter.update_state(tf.argmax(out, axis=1), yy)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('train-loss', float(loss), step=step)
        tf.summary.scalar('test-acc', acc_meter.result().numpy(), step=step)
    if step % 1000 == 0:
        print(step, 'loss:', float(loss), end=' ')
        print('acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
