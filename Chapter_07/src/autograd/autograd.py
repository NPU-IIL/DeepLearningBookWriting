import numpy as np
import matplotlib.pyplot as plt

"""构建向后传播（backward）上下文的辅助函数"""
def build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values):
    """for binary operator"""
    requires_grad = ts1.requires_grad or ts2.requires_grad
    dependency = []
    if ts1.requires_grad:
        dependency.append(dict(tensor=ts1, grad_fn=grad_fn_ts1))
    if ts2.requires_grad:
        dependency.append(dict(tensor=ts2, grad_fn=grad_fn_ts2))
    tensor_cls = ts1.__class__
    return tensor_cls(values, requires_grad, dependency)


def build_unary_ops_tensor(ts, grad_fn, values):
    """对于一元运算符"""
    requires_grad = ts.requires_grad
    dependency = []
    if ts.requires_grad:
        dependency.append(dict(tensor=ts, grad_fn=grad_fn))
    tensor_cls = ts.__class__
    return tensor_cls(values, requires_grad, dependency)

""" 定义Tensor类

- 需要定义数值运算符(numerical operators)
- 存储它的依赖张量(dependent tensors) 
- 存储梯度函数 w.r.t 它的依赖张量
- 重载运算符"""
def as_tensor(obj):
    if not isinstance(obj, Tensor):
        obj = Tensor(obj)
    return obj


class Tensor:
    
    def __init__(self, values, requires_grad=False, dependency=None):
        self._values = np.array(values)
        self.shape = self.values.shape
        
        self.grad = None
        if requires_grad:
            self.zero_grad()
        self.requires_grad = requires_grad
        
        if dependency is None:
            dependency = []
        self.dependency = dependency
            
    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, new_values):
        self._values = np.array(new_values)
        self.grad = None
        
    def zero_grad(self):
        self.grad = np.zeros(self.shape)
        
    def __matmul__(self, other):
        """ self @ other """
        return _matmul(self, as_tensor(other))
        
    def __rmatmul__(self, other):
        """ other @ self """
        return _matmul(as_tensor(other), self)
    
    def __imatmul__(self, other):
        """ self @= other """
        self.values = self.values @ as_tensor(other).values
        return self
    
    def __add__(self, other):
        """ self + other """
        return _add(self, as_tensor(other))
    
    def __radd__(self, other):
        """ other + self """
        return _add(as_tensor(other), self)
    
    def __iadd__(self, other):
        """ self += other """
        self.values = self.values + as_tensor(other).values
        return self
       
    def __sub__(self, other):
        """ self - other """
        return _sub(self, as_tensor(other))
    
    def __rsub__(self, other):
        """ other - self """
        return _add(as_tensor(other), self)
    
    def __isub__(self, other):
        """ self -= other """
        self.values = self.values - as_tensor(other).values
        return self
        
    def __mul__(self, other):
        """ self * other """
        return _mul(self, as_tensor(other))
    
    def __rmul(self, other):
        """ other * self """
        return _mul(as_tensor(other), self)
    
    def __imul(self, other):
        """ self *= other """
        self.values = self.values * as_tensor(other).values
        return self
    
    def __neg__(self):
        """ -self """
        return _neg(self)
    
    def sum(self, axis=None):
        return _sum(self, axis=axis)
    
    
    def backward(self, grad=None):
        assert self.requires_grad, "Call backward() on a non-requires-grad tensor."
        grad = 1.0 if grad is None else grad
        grad = np.array(grad)

        # 计算梯度
        self.grad += grad

        # 将梯度传播到它的依赖项
        for dep in self.dependency:
            grad_for_dep = dep["grad_fn"](grad)
            dep["tensor"].backward(grad_for_dep)
            
            
def _matmul(ts1, ts2):
    values = ts1.values @ ts2.values

    # c = a @ b
    # D_c / D_a = grad @ b.T
    # D_c / D_b = a.T @ grad
    def grad_fn_ts1(grad):
        return grad @ ts2.values.T

    def grad_fn_ts2(grad):
        return ts1.values.T @ grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def _add(ts1, ts2):
    values = ts1.values + ts2.values

    # c = a + b
    # D_c / D_a = 1.0
    # D_c / D_b = 1.0
    def grad_fn_ts1(grad):
        # 广播机制 (5, 3) + (3,) -> (5, 3)
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        # 广播机制 (5, 3) + (1, 3) -> (5, 3)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_ts2(grad):
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def _sub(ts1, ts2):
    return ts1 + (-ts2)


def _mul(ts1, ts2):
    values = ts1.values * ts2.values

    # c = a * b
    # D_c / D_a = b
    # D_c / D_b = a
    def grad_fn_ts1(grad):
        grad = grad * ts2.values
        for _ in range(grad.ndim - ts1.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def grad_fn_ts2(grad):
        grad = grad * ts1.values
        for _ in range(grad.ndim - ts2.values.ndim):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(ts2.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def _neg(ts):
    values = -ts.values

    def grad_fn(grad):
        return -grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def _sum(ts, axis):
    values = ts.values.sum(axis=axis)
    if axis is not None:
        repeat = ts.values.shape[axis]

    def grad_fn(grad):
        if axis is None:
            grad = grad * np.ones_like(ts.values)
        else:
            grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat, axis)
        return grad

    return build_unary_ops_tensor(ts, grad_fn, values)

"""线性回归样例"""
# 训练数据
x = Tensor(np.random.normal(0, 1.0, (100, 3)))
coef = Tensor(np.random.randint(0, 10, (3,)))
y = x * coef - 3 

params = {
    "w": Tensor(np.random.normal(0, 1.0, (3, 3)), requires_grad=True),
    "b": Tensor(np.random.normal(0, 1.0, 3), requires_grad=True)
}

learng_rate = 3e-4
loss_list = []
for e in range(101):
    # 梯度置零
    for param in params.values():
        param.zero_grad()
    
    # 前向传播
    predicted = x @ params["w"] + params["b"]
    err = predicted - y
    loss = (err * err).sum()
    
    # 自动反向传播
    loss.backward()
    
    # 更新参数
    for param in params.values():
        param -= learng_rate * param.grad
        
    loss_list.append(loss.values)
    if e % 10 == 0:
        print("epoch-%i \tloss: %.4f" % (e, loss.values))

plt.figure(figsize=(8, 5))
plt.plot(loss_list)
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()