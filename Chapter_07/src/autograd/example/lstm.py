"""实现长短期记忆字符模型,对多个示例进行矢量化，但每个字符串有固定长度."""

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from os.path import dirname, join
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.special import logsumexp

from autograd.misc.optimizers import adam
from rnn import string_to_one_hot, one_hot_to_string,\
                build_dataset, sigmoid, concat_and_multiply


def init_lstm_params(input_size, state_size, output_size,
                     param_scale=0.01, rs=npr.RandomState(0)):
    # 初始化LSTM模型的参数
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'init cells':   rp(1, state_size),  # 初始单元状态的参数
            'init hiddens': rp(1, state_size),  # 初始隐藏状态的参数
            'change':       rp(input_size + state_size + 1, state_size),  # 输入门权重参数
            'forget':       rp(input_size + state_size + 1, state_size),  # 遗忘门权重参数
            'ingate':       rp(input_size + state_size + 1, state_size),  # 输入门权重参数
            'outgate':      rp(input_size + state_size + 1, state_size),  # 输出门权重参数
            'predict':      rp(state_size + 1, output_size)}  # 预测权重参数

def lstm_predict(params, inputs):
    # 使用给定参数预测LSTM模型的输出
    def update_lstm(input, hiddens, cells):
        change  = np.tanh(concat_and_multiply(params['change'], input, hiddens))  # 计算输入门的值
        forget  = sigmoid(concat_and_multiply(params['forget'], input, hiddens))  # 计算遗忘门的值
        ingate  = sigmoid(concat_and_multiply(params['ingate'], input, hiddens))  # 计算输入门的值
        outgate = sigmoid(concat_and_multiply(params['outgate'], input, hiddens))  # 计算输出门的值
        cells   = cells * forget + ingate * change  # 更新单元状态
        hiddens = outgate * np.tanh(cells)  # 更新隐藏状态
        return hiddens, cells

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)  # 计算输出概率的对数
        return output - logsumexp(output, axis=1, keepdims=True)  # 归一化对数概率

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)  # 初始化隐藏状态
    cells   = np.repeat(params['init cells'],   num_sequences, axis=0)  # 初始化单元状态

    output = [hiddens_to_output_probs(hiddens)]  # 存储每个时间步的输出概率
    for input in inputs:  # 对于每个时间步的输入
        hiddens, cells = update_lstm(input, hiddens, cells)  # 更新LSTM状态
        output.append(hiddens_to_output_probs(hiddens))  # 存储输出概率
    return output

def lstm_log_likelihood(params, inputs, targets):
    # 计算LSTM模型的对数似然
    logprobs = lstm_predict(params, inputs)  # 预测输出概率
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])  # 计算对数似然
    return loglik / (num_time_steps * num_examples)  # 平均对数似然



if __name__ == '__main__':
    num_chars = 128

    # 学习预测我们自己的源代码
    text_filename = join(dirname(__file__), 'lstm.py')
    train_inputs = build_dataset(text_filename, sequence_length=30,
                                 alphabet_size=num_chars, max_lines=60)

    init_params = init_lstm_params(input_size=128, output_size=128,
                                   state_size=40, param_scale=0.01)

    def print_training_prediction(weights):
        # 打印训练文本和预测文本
        print("Training text                         Predicted text")
        logprobs = np.asarray(lstm_predict(weights, train_inputs))
        for t in range(logprobs.shape[1]):
            training_text  = one_hot_to_string(train_inputs[:,t,:])
            predicted_text = one_hot_to_string(logprobs[:,t,:])
            print(training_text.replace('\n', ' ') + "|" +
                  predicted_text.replace('\n', ' '))

    def training_loss(params, iter):
        # 计算训练损失
        return -lstm_log_likelihood(params, train_inputs, train_inputs)

    def callback(weights, iter, gradient):
        # 迭代过程中的回调函数
        if iter % 10 == 0:
            print("Iteration", iter, "Train loss:", training_loss(weights, 0))
            # print_training_prediction(weights)

    # 使用自动微分构建损失函数的梯度
    training_loss_grad = grad(training_loss)

    print("Training LSTM...")
    trained_params = adam(training_loss_grad, init_params, step_size=0.1,
                          num_iters=1000, callback=callback)

    print()
    print("Generating text from LSTM...")
    num_letters = 30
    for t in range(20):
        text = ""
        for i in range(num_letters):
            seqs = string_to_one_hot(text, num_chars)[:, np.newaxis, :]
            logprobs = lstm_predict(trained_params, seqs)[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print(text)
