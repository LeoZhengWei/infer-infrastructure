# 部分算子需要转置。
# y = XA' + B 即 y = Add(MatMul(x, Transpose(A)) + B) 构建计算图
# 转置算子需要一个用于定义轴转换顺序的属性：perm=[1, 0]
# 该属性会作为命名参数添加到make_node函数中

from onnx import TensorProto
from onnx.helper import(
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# unchanged
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# added
node_transpose = make_node('Transpose', ['A'], ['tA'], perm=[1, 0])

# unchanged except A is replaced by tA
node1 = make_node('MatMul', ['X', 'tA'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# node_transpose is added to the list
graph = make_graph([node_transpose, node1, node2], 'lr',
                   [X, A, B], [Y])
onnx_model = make_model(graph)
with open('lr.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
check_model(onnx_model)

# the work is done, let's display 
print(onnx_model)