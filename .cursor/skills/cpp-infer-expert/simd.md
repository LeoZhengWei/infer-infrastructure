# S3 朴素 GEMV 与 AVX2

decode 一步的线性层是 **GEMV**：一个向量乘权重矩阵。权重从内存流过一次，瓶颈在带宽和对齐，不在峰值 FLOPS。

prefill（一次吃进很多 token）才更接近 GEMM。本阶段不要求分块 GEMM。

## 接口（学员自己实现两个函数）

```cpp
// W 行主序 (rows, cols)，x 长度 cols，y 长度 rows
void gemv_naive(const float* W, const float* x, float* y, int rows, int cols);
void gemv_avx2(const float* W, const float* x, float* y, int rows, int cols);
```

这与 llama2.c 的 `matmul(xout, x, w, n, d)` 一致：`rows = d`，`cols = n`。

导师**不给出** AVX2 循环正文。可以提醒三件事：数据 32 字节对齐、`_mm256_fmadd_ps`、沿 K 做内积。编译选项 `-mavx2 -mfma`。

## L09 基线

至少两行，写进 PROGRESS：

| 形状 | 为什么测 |
|------|----------|
| 本模型一层 `wq` 的 `(dim, dim)`，`M=1` | decode 真实尺寸 |
| 另一组更大的 `(rows, cols)`，仍是 GEMV | 避免只在小尺寸上碰巧变快 |

记下 CPU 型号（`lscpu` 的 model name）和编译选项。

## L10 验收（两条都要）

1. 同一输入，`max abs diff(naive, avx2)` 记下来。差一个数量级以上就先修正确性。
2. AVX2 的 GEMV 毫秒数低于朴素版。
3. 把 AVX2 接回整网后，greedy 16 token 仍与 L08 一致。速度不能靠改数学换来。

没有 AVX2 的机器：用实际指令集做一版，表里写明 ISA，不把那一版标成 AVX2。
