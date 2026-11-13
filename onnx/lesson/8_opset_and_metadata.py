# 加载之前生成的模型，检查含有哪些元数据
from onnx import load

with open('lr.onnx', 'rb') as f:
    onnx_model = load(f)

for field in ['doc_string', 
              'domain',
              'functions',
              'ir_version',
              'metadata_props',
              'model_version',
              'opset_import',
              'producer_name',
              'producer_version',
              'training_info']:
    print(f"{field}:{getattr(onnx_model, field)}")

print(f"ir_version:{onnx_model.ir_version}")
for opset in onnx_model.opset_import:
    print(f"opset:opset_domain = {opset.domain} opset_version = {opset.version}")

# IR定义了ONNX语言的版本。算子集（Opset）定义了所使用算子的版本。
# 在未指定精度的情况下，ONNX会使用已安装包自带的最新版本，也可以选用其他版本

del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 14

for opset in onnx_model.opset_import:
    print(f"opset:opset_domain = {opset.domain} opset_version = {opset.version}")

onnx_model.model_version = 1
onnx_model.producer_name = 'someone'
onnx_model.producer_version = 'some other thing'
onnx_model.doc_string = 'documentation about this model'
prop = onnx_model.metadata_props

from onnx import helper, load
data = dict(key1="value1", key2="value2")
helper.set_model_props(onnx_model, data)
print(onnx_model)