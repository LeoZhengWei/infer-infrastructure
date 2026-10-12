# 路线（专家版）

主干是一条因果链：先看清计算和内存，再换成自己的 C++，再用数字证明优化，最后才读别人的框架。

```
S0  三份材料各管哪一层 + 编译器
        ↓
S1  llama2.c：跑通，画前向，画内存
        ↓
S2  engine/：RAII + Tensor，无 cache 前向与 llama2.c 对拍
        ↓
S3  朴素 GEMV 基线 → AVX2（误差 + 延迟）
        ↓
S4  KV cache 对拍，再做同长静态 batch
        ↓
M1  tiny-cnn：算子抽象，并加一个 RMSNorm
        ↓
M2  llama.cpp 核心：图与内存池 → KV → 量化块布局
```

CUDA、FlashAttention、手写 Q4 GEMV、变长 batch、vLLM、TensorRT：全部是加课，不挡主干。

## 目录

```
third_party/llama2.c     S1 对照（L02 克隆）
notes/                   S1、M1、M2 的笔记（学员写）
engine/                  S2–S4 学员的 C++ 推理器
third_party/tiny-dnn     M1
third_party/llama.cpp    M2 只读锚点
learn/PLAN.md
learn/PROGRESS.md
learn/GOALS.md
```

## 默认推进

学员没指定跳课：严格按 `learn/PLAN.md` 的当前课。

指定了错误顺序：纠偏后仍从最近的主干缺口开始。S4 之后若想先看量化，仍然先上 M1，因为 M1 练的是「算子放进框架里」；M2 默认你会读接口而不是把逻辑堆进一个文件。
