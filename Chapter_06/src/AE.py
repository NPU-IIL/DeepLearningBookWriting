import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = (x_train / 255).reshape(-1, 784)
x_test = (x_test / 255).reshape(-1, 784)

def sigmoid(z, d=False):
    return sigmoid(z) * (1 - sigmoid(z)) + 1e-12 if d else 1 / (1 + np.exp(-z))

def relu(z, d=False):
    return (z > 0)+1e-12 if d else  z * (z > 0)

layers = [
    {"act":relu, "shape":(1024,784)},
    {"act":relu, "shape":(50,1024)},
    {"act":relu, "shape":(1024,50)},
    {"act":sigmoid, "shape":(784,1024)}
]

# 全局变量
l, errors, epochs = len(layers), [], 30

lr, b1, b2 = 0.002, 0.9, 0.999
rw,mw,rb,mb = {},{},{},{} 
a,w,b,f, = {},{},{},{}
for i, layer in zip(range(1,l+1), layers):
    n_out, n_in = layer["shape"]
    f[i] = layer["act"]
# Xavier初始化方法
    w[i] = np.random.randn(n_out, n_in) / n_in**0.5
    b[i], rb[i], mb[i] = [np.zeros((n_out,1)) for i in [1,2,3]]
    rw[i], mw[i] = [np.zeros((n_out, n_in)) for i in [1,2]]

for t in range(1, epochs+1):
    # 训练
    for batch in np.split(x_train, 30):
        a[0] = batch.T
    for i in range(1,l+1):
        a[i] = f[i]((w[i] @ a[i-1]) + b[i])
    # 反向传播
    dz,dw,db = {},{},{}
    for i in range(1,l+1)[::-1]:
        d = w[i+1].T @ dz[i+1] if l-i else 0.5*(a[l]-a[0])
        dz[i] = d * f[i](a[i],d=1)
        dw[i] = dz[i] @ a[i-1].T
        db[i] = np.sum(dz[i], 1, keepdims=True)
    
    def adam(m, r, z, dz, i):
        m[i] = b1 * m[i] + (1 - b1) * dz[i]
        r[i] = b2 * r[i] + (1 - b2) * dz[i]**2
        m_hat = m[i] / (1. - b1**t)
        r_hat = r[i] / (1. - b2**t) 
        z[i] -= lr * m_hat / (r_hat**0.5 + 1e-12)
    for i in range(1,l+1):
        adam(mw, rw, w, dw, i)
        adam(mb, rb, b, db, i)
    # 验证
    a[0] = x_test.T
    for i in range(1,l+1):
    a[i] = f[i]((w[i] @ a[i-1]) + b[i])
    errors += [np.mean((a[l]-a[0])**2)]
    print("Val loss - ", errors[-1])

import matplotlib.pyplot as plt
y_pred = []
a[0] = x_train[:20].T
#forward pass
for i in range(1,l+1):
    a[i] = f[i](w[i] @ a[i-1] + b[i])
y_pred = a[l]

plt.figure(figsize=(20,5))

for i in range(20):
    plt.subplot(3, 20, i + 1)
    plt.imshow(x_train[i].reshape(28,28), cmap="gray")
    plt.axis("off")
    plt.grid(b=False)

for i in range(20):
    plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(a[l-2].T[i].reshape(5,-1), cmap="gray")
    plt.axis("off")
    plt.grid(b=False)
    
for i in range(20):
    plt.subplot(3, 20, i + 1 + 40)
    plt.imshow(y_pred.T[i].reshape(28,28), cmap="gray")
    plt.axis("off")
    plt.grid(b=False)

plt.show()

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))
class dA(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, \
        W=None, hbias=None, vbias=None, numpy_rng=None):

        self.n_visible = n_visible  # 可见层（输入层）的单元数量
        self.n_hidden = n_hidden    # 隐藏层的单元数量

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
            
        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # 初始化权重矩阵W
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # 初始化隐藏层偏置为0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # 初始化可见层偏置为0

        self.numpy_rng = numpy_rng
        self.x = input
        self.W = W
        self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias

        # self.params = [self.W, self.hbias, self.vbias]

def get_corrupted_input(self, input, corruption_level):
    assert corruption_level < 1
    return self.numpy_rng.binomial(size=input.shape,n=1,p=1-corruption_level) * input

def get_hidden_values(self, input):
    return sigmoid(numpy.dot(input, self.W) + self.hbias)

def get_reconstructed_input(self, hidden):
    return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)

def train(self, lr=0.1, corruption_level=0.3, input=None):
    if input is not None:
        self.x = input
    x = self.x
    tilde_x = self.get_corrupted_input(x, corruption_level)
    y = self.get_hidden_values(tilde_x)
    z = self.get_reconstructed_input(y)
    L_h2 = x - z
    L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)
    L_vbias = L_h2
    L_hbias = L_h1
    L_W =  numpy.dot(tilde_x.T, L_h1) + numpy.dot(L_h2.T, y)
    self.W += lr * L_W
    self.hbias += lr * numpy.mean(L_hbias, axis=0)
    self.vbias += lr * numpy.mean(L_vbias, axis=0)

def negative_log_likelihood(self, corruption_level=0.3):
    tilde_x = self.get_corrupted_input(self.x, corruption_level)
    y = self.get_hidden_values(tilde_x)
    z = self.get_reconstructed_input(y)
    cross_entropy = - numpy.mean(numpy.sum(self.x * numpy.log(z) +(1 - self.x) * numpy.log(1 - z),axis=1))
    return cross_entropy

def reconstruct(self, x):
    y = self.get_hidden_values(x)
    z = self.get_reconstructed_input(y)
    return z




   

