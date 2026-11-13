# 1. serialize a numpy array to a protobuf binary file
import numpy 
from onnx.numpy_helper import from_array

numpy_tensor = numpy.array([0, 1, 4, 5, 3], dtype=numpy.float32)
print(type(numpy_tensor))

onnx_tensor = from_array(numpy_tensor)
print(type(onnx_tensor))

serialized_tensor = onnx_tensor.SerializeToString()
print(type(serialized_tensor))

with open('saved_tensor.pb', 'wb') as f:
    f.write(serialized_tensor)

# 2. deserialize the protobuf binary file to a numpy array
from onnx import TensorProto
from onnx.numpy_helper import to_array

with open('saved_tensor.pb', 'rb') as f:
    serialized_tensor = f.read()
print(type(serialized_tensor))

onnx_tensor = TensorProto()
onnx_tensor.ParseFromString(serialized_tensor)
print(type(onnx_tensor))

numpy_tensor = to_array(onnx_tensor)
print(numpy_tensor)