from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# 1. 创建初始tensor向量
# X: name 
# TensorProto.FLOAT: type
# [None, None]: shape
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# outputs, the shape is left undefined
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# nodes

# 2. 定义MatMUl的onnx操作结点
# 输入: X, A 输出：'XA'
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# 3. 从结点创建graph
# 图的构建包括，结点，名字，输入，输出
graph = make_graph([node1, node2], # nodes
                    'lr', # a name
                    [X, A, B], # inputs
                    [Y]) # outputs)

onnx_model = make_model(graph)

check_model(onnx_model)

print(onnx_model)


