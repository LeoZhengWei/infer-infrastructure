# CAM++ 声纹：PyTorch CPU / ONNX Runtime / MPS

独立项目。声纹模型是 3D-Speaker 的 CAM++（`iic/speech_campplus_sv_zh-cn_16k-common`）。

三条后端用同一套权重：

| 后端 | 是什么 |
|---|---|
| 原始 | PyTorch `CAMPPlus`，CPU |
| ONNX | 手写 `torch.onnx.export`，ONNX Runtime 1.32。Python 和 C++ 各跑一次 |
| MPS | 同一模型 `model.to("mps")`，Apple Metal |

## 跑

```bash
/Users/mac/data/campplus-sv/run.sh
```

需要本机已有的 `/Users/mac/miniconda3/bin/python`（torch、onnx、onnxruntime）。结果写到 `results/BENCH.md`。

声纹例子在 `audio/`。余弦阈值沿用模型卡的 0.31。计时只含 embedding 前向，不含 fbank。

W4A8 / W4A4 在 `python/bench_w4.py`，结果写到 `results/QUANT.md`。权重是每通道 int4，激活分别是 int8 和 int4，卷积走 `torch._int_mm`。

Q8 是 FP32 ONNX 上的静态 int8（`python/quantize_q8.py`）。`python/bench_q8.py` 用 Python ORT 和 C++ ORT 各记 10 轮，并写出量化前后的余弦差，结果在 `results/Q8.md`。

## 导出时改过的一处

`speakerlab/models/campplus/layers.py` 里 `avg_pool1d` 加了 `count_include_pad=False`。零填充时 PyTorch 数值不变，导出的 `AveragePool` 在各版本 ONNX Runtime 上和 PyTorch 对齐。

权重与 `speakerlab/models/campplus/` 来自 ModelScope / 3D-Speaker，Apache-2.0，见 `third_party/campplus/LICENSE`。ONNX Runtime 头文件是 MIT。
