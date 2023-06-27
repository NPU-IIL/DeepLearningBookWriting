import numpy as np

class Operation:
    """表示执行计算的图形节点。

    `Operation` 是 `Graph` 中的一个节点，它取零或更多对象作为输入，
    并产生零个或多个对象作为输出。
    """

    def __init__(self, input_nodes=[]):
        """构造父类算子 Operation
        """
        self.input_nodes = input_nodes

        # 初始化consumer列表（即接收此操作输出作为输入的节点）
        self.consumers = []

        # 将此操作附加到所有输入节点的consumer列表
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # 将此操作附加到当前活动的默认图中的操作列表
        _default_graph.operations.append(self)

    def compute(self):
        """
        Operation 具体实现
        """
        pass


class add(Operation):
    """返回 x + y 按元素相加.
    """

    def __init__(self, x, y):
        """构建 add 节点

        Args:
          x: 第一个加数节点
          y: 第二个加数节点
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """计算 add operation 的结果

        Args:
          x_value: 第一个加数节点
          y_value: 第二个加数节点
        """
        return x_value + y_value


class matmul(Operation):
    """将矩阵 a 乘以矩阵 b, 生成 a * b。
    """

    def __init__(self, a, b):
        """构建 matmul 节点

        Args:
          a: 第一个矩阵
          b: 第二个矩阵
        """
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        """计算 matmul operation 的输出

        Args:
          a_value: 第一个矩阵值
          b_value: 第二个矩阵值
        """
        return a_value.dot(b_value)


class placeholder:
    """
       表示在计算计算图的输出时必须提供值的占位符节点
    """

    def __init__(self):
        """构建 placeholder
        """
        self.consumers = []

        # 将此占位符附加到当前活动的默认图中的占位符列表
        _default_graph.placeholders.append(self)


class Variable:
    """
    表示一个变量（即计算图的一个固有的、可变的参数）。
    """

    def __init__(self, initial_value=None):
        """构建 Variable

        Args:
          initial_value: 此变量的初始值
        """
        self.value = initial_value
        self.consumers = []

        # 将此变量附加到当前活动的默认图中的变量列表
        _default_graph.variables.append(self)


class Graph:
    """表示计算图
    """

    def __init__(self):
        """构建 Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self


# 创建 gragh,variables,placeholder,隐藏层节点y，输出节点z
Graph().as_default()

A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

x = placeholder()

y = matmul(A, x)

z = add(y, b)




class Session:
    """表示计算图的特定执行。
    """

    def run(self, operation, feed_dict={}):
        """计算 Operation 的输出

        Args:
          operation: 我们要计算其输出的operation。
          feed_dict: 将占位符映射到 Session 的值的字典
        """

        # 对图执行后序遍历以使节点按正确顺序排列
        nodes_postorder = traverse_postorder(operation)

        # 迭代所有节点以确定它们的值
        for node in nodes_postorder:

            if type(node) == placeholder:
                # 将节点值设置为来自 feed_dict 的占位符值
                node.output = feed_dict[node]
            elif type(node) == Variable:
                # 将节点值设置为变量的值属性
                node.output = node.value
            else:  # Operation
                # 从输入节点的输出值中获取此操作的输入值
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # 计算此操作的输出
                node.output = node.compute(*node.inputs)

            # 将列表转换为 numpy 数组
            if type(node.output) == list:
                node.output = np.array(node.output)

        # 返回请求的节点值
        return operation.output


def traverse_postorder(operation):
    """执行后序遍历，按照必须计算的顺序返回节点列表

    Args:
       operation: 开始遍历的 operation
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

session = Session()
output = session.run(z, {
    x: [5, 6]
})
print(output)