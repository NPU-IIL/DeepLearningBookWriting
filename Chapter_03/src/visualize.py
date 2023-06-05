import matplotlib.pyplot as plt
import numpy as np


# 随机从测试集中选择9张图片测试，以九宫格形式打印输出
def visualize_results(model, test_images, test_labels):
    fig = plt.figure(figsize=(8, 8))
    for i in range(9):
        index = np.random.randint(0, len(test_images))
        image = test_images[index]
        label = test_labels[index]
        image = image.reshape(1, 1, 32, 32)
        output = model.forward(image)
        pred = np.argmax(output)
        ax = fig.add_subplot(3, 3, i + 1)
        plt.axis('off')
        ax.imshow(image[0][0], cmap='gray')
        ax.set_title(f'pred: {pred}, label: {np.argmax(label)}')
    plt.show()