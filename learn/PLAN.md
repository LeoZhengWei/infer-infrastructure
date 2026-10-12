# 学习计划（按课推进）

原则：一次只上一课，过关再下一课。不要按「llama.cpp → CUDA → vLLM」当入门顺序。

节奏：每周 3 课，每课 60–90 分钟。主干 13 课（约 5 周），进阶 7 课（约 3 周）。总共约 **7 周 / 20 课 / 25–30 小时**。

**学完要成为什么样的人：** 见 `learn/GOALS.md`。终点是你自己的 C++ 推理器能和 llama2.c 对上 token，并且你能指出工业引擎里内存池、KV、量化块各在解决什么。

进度勾选：`learn/PROGRESS.md`。导师技能会读本文件的「当前课」。

建议日历（从 2026-10-12 起，每周一、三、五各一课；挤不出就顺延，不要并课）：

| 周 | 日期 | 课 |
|----|------|----|
| 1 | 10/12–10/18 | L01–L03 |
| 2 | 10/19–10/25 | L04–L06 |
| 3 | 10/26–11/01 | L07–L09 |
| 4 | 11/02–11/08 | L10–L12 |
| 5 | 11/09–11/15 | L13–L15 |
| 6 | 11/16–11/22 | L16–L18 |
| 7 | 11/23–11/29 | L19–L20 |

L12 结束时，计算、内存、对拍、KV cache 已经闭环。L13 做完才离开自己的引擎，去看别人的框架。

---

## 怎么上每一课

1. 用 Cursor 打开仓库目录 `infer-infrastructure`（不要只打开上一级 `zw`），开 Agent。
2. 把该课的 **开场白** 原样发给 Agent（下面每课都写好了）。
3. 只追求该课的 **过关标准**。做完就停，不要顺手开下一课。
4. 让 Agent 更新 `PROGRESS.md`。你自己在本文件课号前打勾。

卡超过 40 分钟仍无产物：缩小到「先打印 Config」或「先对拍 4 个 token」，记下报错，下一课先修这个坑。

---

## 当前课

**L01**（S0 心智与工具链）

---

## 主干（必须按序，L01–L13）

### L01 · S0 心智与工具链 · 60 分钟

- 做：`g++ --version`、`cmake --version`、`lscpu | grep -i avx`。能编译一个打印 `sizeof(float)` 的文件即可。
- 懂：llama2.c、自己的 `engine/`、tiny-cnn、llama.cpp 核心各解决哪一层（见 skill 的 `stack.md`）。
- 过关：能用自己的话回答「CPU 上单请求生成慢，在写 AVX2 之前先确认什么」。
- 开场白：`按计划上 L01。核验 g++/cmake/AVX2，讲清 llama2.c、tiny-cnn、llama.cpp 各解决什么。不要克隆仓库，不要写推理代码。`

### L02 · S1 跑通 llama2.c · 90 分钟

- 做：把 [karpathy/llama2.c](https://github.com/karpathy/llama2.c) 克隆到 `third_party/llama2.c`，按上游说明下载 stories15M.bin，编译运行并生成一段文本。
- 过关：PROGRESS 里有命令、权重路径、一段生成摘录。
- 开场白：`按计划上 L02。只克隆 llama2.c 并跑通 stories15M。不要改成 C++，不要看 tiny-cnn。`

### L03 · S1 前向调用链 · 90 分钟

- 做：通读 `run.c` 的 forward。在 `notes/L03-forward.md` 按调用顺序列出每一步的输入输出形状（用 Config 的名字）。
- 过关：笔记能从 token id 走到 logits，且含 RMSNorm、QKV、RoPE、softmax、残差、门控 FFN。
- 开场白：`按计划上 L03。只画 run.c 前向调用链和形状。不写 C++。`

### L04 · S1 内存布局 · 90 分钟

- 做：`notes/L04-memory.md`。三块：Config、权重大数组里各张量的偏移、RunState 里哪些缓冲每层复用、KV cache 的元素个数。
- 过关：能手算某一层 `wq` 的起始元素下标（写出公式，代入本模型 Config）。
- 开场白：`按计划上 L04。只画内存布局和偏移公式。不写 C++。`

### L05 · S1 算子口试 · 60 分钟

- 做：闭卷讲 matmul、RMSNorm、RoPE、attention 缩放各在哪。只对照 `run.c` 改口误。
- 过关：四段都能讲清公式和张量形状。
- 开场白：`按计划上 L05。口试四个算子。不开始 C++ 移植。`

### L06 · S2 加载权重 · 90 分钟

- 做：在 `engine/` 用 CMake 建 C++17 程序。`std::vector` / `unique_ptr` 读入 bin，打印 Config。Debug + ASan 能正常退出。
- 过关：打印的 dim、n_layers、n_heads、n_kv_heads、vocab、seq_len 与 llama2.c 的 Config 一致。
- 开场白：`按计划上 L06。只做权重加载和 RAII。不要写 forward。`

### L07 · S2 Tensor 视图 · 90 分钟

- 做：Tensor 持有连续内存、shape、stride。用视图切出第 0 层 `wq`，不额外拷贝。
- 过关：断言元素个数正确，且数据指针落在权重大数组内部。
- 开场白：`按计划上 L07。实现 Tensor 视图。不要写整网 forward。`

### L08 · S2 无 cache 前向对拍 · 90 分钟

- 做：每步重算整段序列的前向 + greedy。同一 prompt、temperature 0，与 `./run` 比 16 个 token。
- 过关：16 个 token id 完全一致，写入 PROGRESS。
- 开场白：`按计划上 L08。C++ 无 cache 前向与 llama2.c greedy 对拍。不要 SIMD，不要自己加 cache。`

### L09 · S3 朴素 GEMV 基线 · 60 分钟

- 做：拆出 `gemv_naive`。至少测本模型一层 `wq` 的尺寸，再测一组更大的 GEMV。
- 过关：PROGRESS 基线表有两行 naive（rows、cols、毫秒、CPU）。
- 开场白：`按计划上 L09。只测朴素 GEMV。不要写 AVX2。`

### L10 · S3 AVX2 · 90 分钟

- 做：`gemv_avx2`。与 naive 同一输入对拍，再比延迟。接回整网后 16 token 仍与 L08 一致。
- 过关：表里有 max abs diff 和两行延迟；token 仍然一致。
- 开场白：`按计划上 L10。AVX2 GEMV 对拍并填延迟。不要改数值算法去换速度。`

### L11 · S4 KV cache · 90 分钟

- 做：为每层保存 K、V。decode 只写当前位置。RoPE 用绝对位置下标。
- 过关：能生成 16 token。这一课不对拍。
- 开场白：`按计划上 L11。给 engine 加 KV cache。先求能跑，不要 batch。`

### L12 · S4 cache 对拍 · 60 分钟

- 做：同一 prompt，无 cache 与有 cache 的 greedy token。再把生成长度加到 64，比较时间。
- 过关：16 token 完全一致；64 token 的生成时间里，cache 更短。两行写入基线表。
- 开场白：`按计划上 L12。无 cache 与 KV greedy 对拍，并记下延迟。`

### L13 · S4 同长静态 batch · 90 分钟

- 做：batch=2，两条序列一样长。每条与自己的 batch=1 结果比较。
- 过关：两条 prompt 的 16 token 都与单条一致。
- 开场白：`按计划上 L13。同长度静态 batch。不要连续批处理，不要分页 KV。`

主干结束。下一课起才打开 tiny-cnn，再下一阶段才打开 llama.cpp。

---

## 进阶（L14–L20，顺序固定）

为何是 **框架 → llama.cpp 核心**：

- 自己的引擎是单文件式的计算。tiny-cnn 用来看算子如何变成可以注册的层。
- llama.cpp 用来对上工业代码里的三块：图与内存池、KV、量化布局。全仓库不是课文。

### L14 · M1 跑通 tiny-cnn · 60 分钟

- 做：克隆 [tiny-dnn/tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) 到 `third_party/tiny-dnn`，跑 MNIST 最小示例。
- 过关：PROGRESS 记下示例路径和它打印的精度或损失。
- 开场白：`按计划上 L14。只跑 tiny-cnn 的 MNIST 示例。不要改框架。`

### L15 · M1 精读结构 · 90 分钟

- 做：`notes/L15-framework.md`。张量、层接口、一条 SIMD 热循环、NCHW 与 NHWC、模板分发在哪。
- 过关：能指出一条热路径没有虚调用，并说明布局如何影响最内层访存。
- 开场白：`按计划上 L15。精读 tensor 与卷积实现，写笔记。不要加新算子。`

### L16 · M1 新增 RMSNorm · 90 分钟

- 做：按该框架的层写法新增 RMSNorm，加固定输入测试。与 `engine/` 的 RMSNorm 或手算对拍。
- 过关：测试通过，max abs diff 写入 PROGRESS。
- 开场白：`按计划上 L16。给 tiny-cnn 加 RMSNorm 并对拍。不要改 llama.cpp。`

### L17 · M2 ggml 张量、内存池、图 · 90 分钟

- 做：克隆 llama.cpp。只读张量结构、分配器、建图与执行。跳过 GPU 后端、server、量化内核。
- 过关：`notes/L17-ggml.md` 里有「建图 → 分配 → 计算」三步和对应符号名，并写明和自己 Tensor 的差别。
- 开场白：`按计划上 L17。只读 ggml 的张量、内存池、计算图。不要读完整个仓库，不要写后端。`

### L18 · M2 Llama 的 KV 与采样 · 90 分钟

- 做：沿一条 decode 看 KV 的形状、写回位置、greedy 采样。
- 过关：笔记写明和 L11 的 cache 差在哪。分页只说明它避免「每条请求占满最大长度」。
- 开场白：`按计划上 L18。只读 llama 的 KV、采样和推理主循环。不要做服务端。`

### L19 · M2 量化块布局 · 90 分钟

- 做：找到 Q8_0 与 Q4_0 的结构体，画一个 block 的字节。算出相对 fp32 省下多少字节。
- 过关：两个 block 的元素数、字节数、节省量写进 PROGRESS。
- 开场白：`按计划上 L19。只搞清 8bit/4bit 块布局和字节数。不要实现完整量化 matmul。`

### L20 · 复盘与口试 · 60 分钟

- 做：基线表收齐。闭卷回答 `learn/GOALS.md` 的 5 题。`notes/L20-recap.md` 只写这台机器上的四行数字：朴素 GEMV、AVX2、无 cache、有 cache。
- 过关：口试通过，表有数字。
- 开场白：`按计划上 L20。收齐基线表并做毕业口试。不要开新项目。`

---

## 明确不做（直到计划写到）

- 克隆 llama.cpp 当作 L02 的作业，或通读全部后端
- 下载 7B、编译 CUDA、`pip install vllm` 当完成标准
- 在 token 还没对齐时写 AVX2 或量化 GEMV
- 连续批处理、分页 KV 的生产实现（服务调度在 `llms-zth` 那条线，不在本仓库）

---

## 加课（全部 L20 之后，可选）

- 变长 batch 与 padding mask
- 手写一个 Q8 GEMV，与 fp32 对拍并记录误差
- 只服务 prefill 的分块 GEMM
- 对照阅读 llama.cpp 里一个 CPU 算子的实现，映射回自己的 `gemv_avx2`
