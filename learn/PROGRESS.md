# 学习进度

最后更新：2026-10-12 · 尚未开课

## 当前阶段

下一课 **L01：S0 心智与工具链**（详见 `learn/PLAN.md`）。毕业标准：`learn/GOALS.md`。

已有相邻经验：`llms-zth` 的 PyTorch 小 GPT 学到 L06（训练与 greedy 采样）。本仓库不因此免课。L03 的公式可以走快，内存布局和 C++ 对拍仍从 L01 按序做。

## 会话日志

- 10-12：仓库与课表建立。当前课 L01。

## 毕业（L20 才勾）

- [ ] 口试 5 题能闭卷答（见 GOALS.md）
- [ ] 产物清单齐全（笔记 / engine 对拍 / 基线表 / RMSNorm / 量化字节账）
- [ ] 无 AVX2 的机器已在表里写明实际 ISA，而不是把别的指令集标成 AVX2

## 主干检查点

- [ ] S0 能区分：llama2.c / engine / tiny-cnn / llama.cpp 核心各解决什么
- [ ] S1 stories15M 已生成文本；前向笔记和 `wq` 下标公式写完
- [ ] S2 `engine/` 加载 Config 正确；无 cache 的 16 token 与 llama2.c 一致
- [ ] S3 朴素 GEMV 与 AVX2：有 max abs diff，AVX2 更快，整网 token 仍一致
- [ ] S4 KV cache 与无 cache 的 token 一致，拉长生成后 cache 更短
- [ ] S4 同长 batch=2 与各自的单条一致

## 进阶检查点（S4 后再勾）

- [ ] M1 tiny-cnn MNIST 示例跑通；笔记指出热路径与布局
- [ ] M1 RMSNorm 对拍有 max abs diff
- [ ] M2 笔记写清 ggml 的建图、分配、计算
- [ ] M2 能对比工业 KV 与自己的 cache
- [ ] M2 Q8_0 / Q4_0 的字节账

## 主线实现

- 参照：`third_party/llama2.c`（L02 克隆）+ stories15M.bin
- 学员代码：`engine/`（L06 起）
- 框架阅读：`third_party/tiny-dnn`（L14）、`third_party/llama.cpp`（L17）
- Config：跑起来之后把 dim / n_layers / n_heads / n_kv_heads / vocab / seq_len 填到这里

## 基线表

| 项 | rows | cols 或 token | 时间 | max abs diff | 硬件 / 编译选项 | 备注 |
|----|------|----------------|------|--------------|-----------------|------|
| gemv naive（wq 尺寸） |  |  |  | — |  | L09 |
| gemv naive（更大） |  |  |  | — |  | L09 |
| gemv avx2（wq 尺寸） |  |  |  |  |  | L10 |
| 生成 16 token 无 cache | — | 16 |  | token 与 llama2.c |  | L08/L12 |
| 生成 64 token 无 cache | — | 64 |  |  |  | L12 |
| 生成 64 token 有 cache | — | 64 |  | 与无 cache 的 token |  | L12 |

## 环境笔记

- 系统：Ubuntu
- 编译器 / cmake / AVX2：L01 填写
- 权重路径：L02 填写
