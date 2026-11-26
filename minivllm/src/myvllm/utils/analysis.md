# utils
一步一步分析vllm/utils下代码的作用
## context.py
## loader.py
### def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str)
#### L14 - L56
1. 整体是查找输入的模型路径是否存在，如果存在就使用输入的路径，不存在就去huggingface_hub上下载对应的模型到该路径；
2. 准备好模型文件后，逐个读入模型文件到safetensor_files，按照顺序排序这个模型文件后，依次从safetensors文件中读入登记的张量名，存储到hf_weights中,这里收集的是hf对应的权重;多个分片共用同一个hf_weights，读完字典后才是整份Huggingface权重。后面要把q_proj, k_proj, v_proj拼成一快，才必须按名字把他们都收齐。
3. loaded_params是集合，记「已经抄进自定义模型」。一次成功的QKV拼接会放进四个字符串：
    · 自定义参数名: model.layers.0.self_attn.qkv_projection.weight
    · 以及被吃掉的三个HuggingFace名字q_proj, k_proj, v_proj
    该集合方便之后hf_name in loaded_params能直接判断。
4. skipped_params是列表，元素是(名字，原因)。用于最后的打印，不参加拷贝。原因：
    · 自定义模型没有同名参数
    · 张量已经并入qkv_projection或gate_up
    · 拷贝时抛了异常
#### L59 - L86 
处理hf_weights，将同一层的Q、K、V沿输出维拼成一块自定义模型中建好的**qkv_projection.weight**。
磁盘上的权重文件不动，hf_weights里原来的张量不变,torch.cat得到的是一块新张量。
```text
Qwen3是三个独立线性层
参数                形状
q_proj.weight       (2048, 1024)
k_proj.weight       (1024, 1024)
v_proj.weight       (1024, 1024)
```
自定义模型把它们收成一个QKVColumnParallelLinear。
forward里一次矩阵乘，再按[q_size, kv_size, kv_size]切开。
加载时按同样的顺序叠回去:torch.cat(..., dim=0)得到(4096, 1024)。
Linear的权重布局是(输出维， 输入维)，所以dim=0才是输出维。
