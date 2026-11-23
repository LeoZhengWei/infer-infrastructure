# CAM++ 声纹

一周学习用的骨架。计划、过关标准和截止时间在 `summary.md`。推理、导出、量化、计时的代码自己写。

留下的只有这些：

| 路径 | 作用 |
|---|---|
| `weights/config.yaml` | 网络和前端配置 |
| `weights/configuration.json` | ModelScope 的包说明 |
| `weights/campplus_cn_common.bin` | 权重，不进 git。没有就按 `summary.md` 周一的命令下载 |
| `audio/*.wav` | 三条 16 kHz 示例语音 |
| `third_party/campplus/speakerlab/` | CAM++ 模型定义，只读 |
| `requirements.txt` | torch、onnx、onnxruntime、numpy、onnxsim |

环境是 Python 3.11 这一条：

```bash
conda activate /home/hjadmin/zw/infer-infrastructure/cfu-playground/env/conda/envs/onnx
```

每天的笔记写到 `results/dayN.md`，周日收成 `results/week.md`。
