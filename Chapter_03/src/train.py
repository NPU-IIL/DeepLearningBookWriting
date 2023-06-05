import numpy as np
from functions import cross_entropy_loss, softmax_backward


def train(model, train_images, train_labels, epochs, learning_rate, batch_size):
    for epoch in range(epochs):
        correct = 0
        loss_list = []
        batch_num = 0
        for i in range(0, len(train_images), batch_size):
            batch_num += 1
            batch_images = train_images[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # forward
            output = model.forward(batch_images)
            loss = cross_entropy_loss(output, batch_labels)
            loss_list.append(loss)

            # backward
            grad_output = softmax_backward(output, batch_labels)
            model.backward(grad_output)

            # update parameters
            for layer in [model.conv1, model.conv2, model.linear1, model.linear2, model.linear3]:
                layer.weights -= learning_rate * layer.grad_weights
                layer.bias -= learning_rate * layer.grad_bias

            # calculate accuracy
            pred = np.argmax(output, axis=1)
            correct += np.sum(pred == np.argmax(batch_labels, axis=1))
            print(f'iter {i / batch_size}, loss: {loss}')
        model.loss.append(loss_list)
        avg_loss = np.mean(np.array(loss_list)).item()
        print('Epoch %d/%d, avg_loss: %.3f, train accuracy: %.3f' % (
            epoch + 1, epochs, avg_loss, correct / len(train_images)))

