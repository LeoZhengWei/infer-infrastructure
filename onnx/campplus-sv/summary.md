# CAM++ 声纹：baseline 与优化

模型是 3D-Speaker 的 CAM++（`iic/speech_campplus_sv_zh-cn_16k-common`），输入 80 维 fbank，输出 192 维 embedding。同一套权重贯穿所有后端。声纹阈值沿用模型卡的余弦 0.31。

计时协议：Apple M1，PyTorch 与 ONNX Runtime intra-op 均为 4 线程。先跑 10 轮丢掉启动（会话创建、MPS 着色器编译），再记 10 轮，表中取中位数。只计 embedding 前向，不含 fbank。MPS 在 `synchronize` 之后停表。

示例语音三条。同一人 `speaker1_a` vs `speaker1_b`，不同人 `speaker1_a` vs `speaker2_a`。这不是测试集上的等错误率。

## Baseline：PyTorch CPU FP32

| 输入 | 中位数 |
|---|---:|
| T=300（约 3 秒） | 32.77 ms |
| T=370（`speaker1_a`，约 3.7 秒） | 44.25 ms |
| T=800（约 8 秒） | 84.85 ms |

声纹余弦：同一人 0.6936，不同人 −0.0842。后面所有优化都和这组数比。

## 优化 1：导出 ONNX，Python Runtime

手工 `torch.onnx.export`（opset 17，batch 与时间维动态）。`avg_pool1d` 加上 `count_include_pad=False` 后，零填充的数值与 PyTorch 一致，ONNX Runtime 的 `AveragePool` 不再把残窗按整窗长度做平均。

与 PyTorch 的最大绝对误差约 1e-5。三条语音的余弦与 baseline 相同。

| 输入 | PyTorch CPU | ONNX Python | 相对 baseline |
|---|---:|---:|---:|
| T=300 | 32.77 ms | 19.02 ms | 1.72× |
| T=370 | 44.25 ms | 22.13 ms | 2.00× |
| T=800 | 84.85 ms | 48.27 ms | 1.76× |

保留。误差可对拍，速度有实测。

## 优化 2：C++ ONNX Runtime

同一张 FP32 图，C++ API，intra-op 4 线程。

| 输入 | PyTorch CPU | ONNX C++ | 相对 baseline |
|---|---:|---:|---:|
| T=300 | 32.77 ms | 18.75 ms | 1.75× |
| T=370 | 44.25 ms | 21.43 ms | 2.06× |
| T=800 | 84.85 ms | 41.88 ms | 2.03× |

余弦仍与 baseline 相同。保留。C++ 相对 Python Runtime 只在更长的 T=800 上明显更快（41.88 ms 对 48.27 ms）。

## 优化 3：Apple MPS

同一权重 `model.to("mps")`。余弦相对 CPU 为 1，最大绝对误差约 1e-5。

| 输入 | PyTorch CPU | MPS |
|---|---:|---:|
| T=300 | 32.77 ms | 51.44 ms |
| T=370 | 44.25 ms | 54.98 ms |
| T=800 | 84.85 ms | 60.57 ms |

短序列上 Metal 启动仍占主导，MPS 慢于 CPU，也慢于 ONNX。约 8 秒时 MPS 快于 CPU，仍慢于 ONNX。不作为上线后端。

## 优化 4：W4A8 / W4A4

权重按输出通道对称量化到 int4。W4A8 的激活是 int8，W4A4 的激活是 int4。卷积为 im2col 加 `torch._int_mm`，BN 与 ReLU 仍是 FP32。

| 输入 | FP32 | W4A8 | W4A4 |
|---|---:|---:|---:|
| T=300 | 30.32 ms | 272.00 ms | 253.45 ms |
| T=370 | 39.68 ms | 339.66 ms | 385.74 ms |
| T=800 | 72.77 ms | 713.11 ms | 830.53 ms |

与 FP32 embedding 的余弦：T=370 上 W4A8 为 0.4576，W4A4 为 0.1850。声纹配对：

| 配对 | FP32 | W4A8 | W4A4 |
|---|---:|---:|---:|
| 同一人 | 0.6936 | 0.8292 | 0.7245 |
| 不同人 | −0.0842 | 0.5449 | 0.1776 |

W4A8 把不同人打到 0.5449，超过阈值 0.31。两条低比特路径都比 FP32 慢约 8–10 倍。M1 上的 int8 矩阵乘没有打过 FP32 的 AMX。放弃。

## 优化 5：ONNX INT8 静态量化（Q8）

在 FP32 ONNX 上做静态 PTQ。权重和激活都是对称 QInt8，权重 per-channel，校准数据是若干段随机 fbank 加上三条示例语音。图仍是输入 `x`、输出 `embedding`。

Python Runtime（与当次 FP32 同进程比较）：

| 输入 | FP32 | Q8 | 量化前后余弦 |
|---|---:|---:|---:|
| T=300 | 19.21 ms | 15.20 ms | 0.8425 |
| T=370 | 21.52 ms | 17.44 ms | 0.9378 |
| T=800 | 42.69 ms | 31.97 ms | 0.8641 |

C++ Runtime：

| 输入 | FP32 | Q8 | 量化前后余弦 |
|---|---:|---:|---:|
| T=300 | 25.40 ms | 21.04 ms | 0.8425 |
| T=370 | 29.43 ms | 23.59 ms | 0.9378 |
| T=800 | 52.53 ms | 37.13 ms | 0.8641 |

声纹配对（Python 与 C++ 相同）：同一人 0.6936 → 0.6343，不同人 −0.0842 → −0.0731。阈值仍成立。约 8 秒输入上 Q8 相对 FP32 ONNX 约 1.3 倍。保留，作为在 FP32 ONNX 之上的进一步优化。单点 max abs 约 2，不能只用绝对误差判断声纹是否还能用。

## 取舍

上线路径是 PyTorch baseline 对拍通过之后的 ONNX，C++ Runtime 跑 FP32，需要再挤延迟时用 Q8。MPS 和 W4 都测过，不进入这条路径。

相对 baseline（PyTorch CPU，T=370）：C++ FP32 从 44.25 ms 到 21.43 ms。Q8 的绝对毫秒要看它自己那一轮的 FP32，不能和 baseline 表混减。
