# M2 llama.cpp 核心子集

上游 https://github.com/ggerganov/llama.cpp ，克隆到 `third_party/llama.cpp`。

文件已经被拆开，而且还会再拆。不要按旧教程死找单一的 `ggml.c`。用下面的锚点让学员搜索，读到锚点就停。

明确跳过：CUDA / Metal / Vulkan / RPC、`server` 与 HTTP、完整量化 matmul 内核、所有新架构的模型文件。

## L17 张量、分配器、图

笔记 `notes/L17-ggml.md` 要有三步和对应的类型或函数名：

1. 张量结构（维度、类型、数据指针）
2. 内存从哪一次分配来（上下文 / 分配器 / 图的工作缓冲）
3. 建图与取结果

并写一句：这和 `engine/` 的 Tensor 差在「先描述计算，再分配、再执行」。

## L18 Llama 主路径

只追一条 decode：

- 上下文里 KV cache 的形状和生命周期
- 采样（先看 greedy / argmax 那条）
- 一步新增 token 时谁在写 cache

笔记要对比学员 L11 的 cache：工业实现多出来的是多序列、更多元数据，以及以后才会碰到的分页。这一课点到分页的目的（避免每条请求独占一整段最大长度），不实现分页。

## L19 量化块

找到 `block_q8_0` 与 `block_q4_0`（若改名，找注释里仍叫 Q8_0 / Q4_0 的结构）。

学员自己算：一个 block 覆盖多少个元素、结构体多少字节、同样元素的 fp32 多少字节、省下多少。

可选：用十行以内的伪代码说明反量化在做什么。不把量化 GEMV 接进 `engine/`。

历史布局供导师核对（以上游结构体为准，不一致就以上游为准）：

- Q8_0：一块 32 个元素，一个 fp16 scale，32 个 int8
- Q4_0：一块 32 个元素，一个 fp16 scale，16 个字节的 nibble

## L20

收齐 PROGRESS 基线表，闭卷回答 `learn/GOALS.md` 的 5 题。半页笔记只写这台机器上的数字：朴素 GEMV、AVX2、无 cache 生成、有 cache 生成。
