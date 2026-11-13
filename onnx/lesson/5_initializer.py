# 1-4假设线性模型的系数是模型的输入，这不方便。
# 这些系数越高成为常量或随着onnx语义的初始化一起保存到模型本身。
# 本节将把这些系数纳入初始化中，这个包实现了两个函数去转换numpy数组到onnx数组的互相转换。

import numpy 
from onnx import numpy_helper, TensorProto
from onnx.helper import(
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
A = numpy_helper.from_array(value, name='A')
value = numpy.array([0.1, 0.2], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['AX'])
node2 = make_node('Add', ['AX', 'C'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
onnx_model = make_model(graph)
check_model(onnx_model)

print(onnx_model)

print('** initializers **')
for init in onnx_model.graph.initializer:
    print(init)

    