# S2 C++ 改造

产物都在 `engine/`，由学员从 L06 起新建。C++17。权重文件与 llama2.c 用同一份 stories15M.bin。

## 所有权

- 权重大数组和 RunState 用 `std::vector<float>` 或 `std::unique_ptr`，离开作用域自己释放。
- 热路径上不再出现裸 `malloc` / `free`。
- L06 只加载文件头和权重，打印 Config。打印值与 llama2.c 的 Config 一致。
- 建议验收：`cmake -DCMAKE_BUILD_TYPE=Debug` 加 `-fsanitize=address,undefined`，加载后正常退出。

文件头是 7 个 `int`：`dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len`。`vocab_size < 0` 表示分类器权重不与 embedding 共享，真正词表大小取绝对值。字段以学员克隆到的 `run.c` 为准，若上游改了，以文件为准并写进 PROGRESS。

## Tensor

L07 的最小接口：

```cpp
struct Tensor {
    float* data;
    std::vector<int> shape;
    std::vector<int> stride; // 元素为单位
    bool owned;
};
```

- 权重视图：`owned == false`，`data` 指向大数组内部。
- 学员要能取出第 0 层 `wq`，断言元素个数，并断言指针落在权重大数组的区间里。
- 这一课禁止把 `wq` 拷贝到第二块缓冲里再算。

## 无 cache 前向（L08）

行为：每一步用当前完整序列重算 Q、K、V 和注意力，不读、不写 KV cache。

对拍：

```text
同一 prompt
temperature = 0（greedy）
生成 16 个 token
与 third_party/llama2.c 的 ./run 逐个相同
```

分叉时按这个顺序查，不要同时改几处：

1. 权重偏移（最常见）
2. `n_kv_heads` 与重复次数
3. RoPE 的位置下标
4. 因果 mask 的方向
5. RMSNorm 的 eps、SiLU 门控是乘上去的

tokenizer 可以先调用 llama2.c 的 `tokenizer.bin` 逻辑；允许学员用一个小脚本把 prompt 编成 token id 再送进 C++。对拍的是 token id，不是字符串外观。

L08 结束时 PROGRESS 写下 16 个 token id。
