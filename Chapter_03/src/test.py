import numpy as np


def test(model, test_images, test_labels):
    correct = 0
    for image, label in zip(test_images, test_labels):
        image = image.reshape(1, 1, 32, 32)
        output = model.forward(image)
        if np.argmax(output) == np.argmax(label):
            correct += 1
    print('Test accuracy: %.3f' % (correct / len(test_images)))
