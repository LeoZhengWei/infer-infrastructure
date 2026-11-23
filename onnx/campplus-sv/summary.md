# CAM++ 一周：从权重到 Q8 和图优化

做完这七天，你要能不看稿讲清下面七件事。讲不清的那件，当天不算过。

1. `weights/` 里每个文件是什么，权重从哪来，加载时构造函数和 `state_dict` 怎么对齐。
2. 16 kHz 波形怎样变成 80 维 fbank，模型为什么吐出 192 维 embedding，余弦 0.31 在比较什么。
3. 前向数据流：`FCM` → `TDNN` → 三段 `CAMDenseTDNNBlock` + `TransitLayer` → `StatsPool` → `Dense`。说出每一段的输入输出形状怎么变。
4. `torch.onnx.export` 交出一张什么图。Netron 里哪些节点对应 PyTorch 的哪一个模块。
5. 静态 int8（Q8）里，校准集、对称量化、per-channel、QDQ 各管哪一段。
6. ONNX Runtime 图优化四级（关闭 / Basic / Extended / All）各自改了图的哪一部分。节点数你自己数。
7. 一张协议完全相同的时延表。变慢的每一行都有原因，原因来自你打开的图，不来自猜测。

收获是这条能复述的链路。毫秒数换一台机器就作废。讲不出原因的加速，按没做处理。

## 纪律

今天是 2026-11-23，周日 2026-11-29 23:59 截止。七天，一天一步，不并行。

- 推理、导出、量化、计时的代码全部自己写，自己跑。
- `third_party/campplus/speakerlab/` 是要读的模型，保留。CAM++ 本身不重写。
- 旧的导出脚本、量化脚本、计时脚本、C++ 工程和旧的毫秒数已经从工作区清掉。git 历史里的那些文件不打开。
- 每天 23:59 前把产物写到 `results/dayN.md`。没有文件，当天作废。第二天先补完，再开始新的一天。
- 计时协议在周三写死。之后改线程数、改是否含 fbank、改取平均还是中位数，整张表作废，全部重测。
- 声纹闸门：同一人 `speaker1_a` 对 `speaker1_b` 的余弦高于 0.31，不同人 `speaker1_a` 对 `speaker2_a` 的余弦低于 0.31。时延下降但闸门破了，这一档失败，退回上一档。
- 环境只用这一条：`/home/hjadmin/zw/infer-infrastructure/cfu-playground/env/conda/envs/onnx`（Python 3.11，torch 2.15）。`conda activate onnx` 已经对不上任何环境。

本周不做这三件，它们会把七天撑爆，也会把结论搅浑：

- Apple MPS。这台机器没有。
- 手写 W4A8 / W4A4。低比特核在这颗 CPU 上经常比 FP32 更慢，原理要等 Q8 讲清楚再碰。
- C++ ONNX Runtime。语言绑定不是这周的对象。

## 计时协议（周三写进 day3，之后不许改）

先把下面几条定成数字，再跑第一轮。

- 只计 embedding 前向。fbank 单独记一行，不混进模型时延。
- 正式计时前先 warmup。次数你定，写下来。第一轮含会话创建和图优化，不算成绩。
- 正式轮数至少 10，取中位数。同时记下最小和最大，防止被一次抖动骗了。
- CPU。`intra_op_num_threads` 固定成一个数，PyTorch 的线程数用同一个数。
- 两组输入都要有：三条真实语音；固定长度的 fbank，至少 T=300 和 T=800，形状 `(1, T, 80)`，种子固定。真实语音用来看声纹，固定 T 用来让各后端对齐。
- 每张表写明：机器、日期、线程、warmup、轮数、统计量。

## 周一 11/23　下载，认清三个文件

模型是 ModelScope 上的 `iic/speech_campplus_sv_zh-cn_16k-common`。

```bash
conda activate /home/hjadmin/zw/infer-infrastructure/cfu-playground/env/conda/envs/onnx
cd ~/zw/infer-infrastructure/onnx/campplus-sv
python -c 'import sys,torch; print(sys.executable); print(torch.__version__)'
mkdir -p weights
curl -L -o weights/campplus_cn_common.bin \
  "https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/resolve/master/campplus_cn_common.bin"
file weights/campplus_cn_common.bin
```

`file` 报 Zip archive 是正常的，这是 PyTorch `torch.save` 的包。bin 不要提交。

读完 `weights/config.yaml` 和 `weights/configuration.json`，在 `results/day1.md` 写三段：

- bin 里是什么，谁会 `torch.load` 它。
- `config.yaml` 描述哪张网、特征维、embedding 维、前端采样率。
- `configuration.json` 是 ModelScope 的包说明，指出权重文件名和余弦阈值 0.31 写在哪。

过关：三段都能指到具体字段。解释器路径是 3.11 那个，不是 Anaconda 里已经删掉的 3.10。

## 周二 11/24　结构和特征

只读，不写推理。

按这个顺序读：

- `weights/config.yaml` 的 `model_conf`、`frontend_conf`
- `third_party/campplus/speakerlab/models/campplus/DTDNN.py` 的 `FCM`、`CAMPPlus`、`forward`
- `layers.py` 的 `get_nonlinear`、`StatsPool`、`TDNNLayer`、`CAMDenseTDNNBlock`、`TransitLayer`
- `classifier.py` 开头。确认推理图里没有这个分类头，写出它是训练时用的

`results/day2.md` 里要有一张你自己画的前向，从 `(B, T, 80)` 到 192 维。每一框写输入形状、输出形状、模块名。`permute`、`FCM` 把频率维卷进通道、`StatsPool` 把时间维收成均值和标准差，这三处必须出现。

特征用自己的话写完，不贴代码：

- 16 kHz，帧长 25 ms，帧移 10 ms，80 个 mel，预加重 0.97
- 逐句减去特征均值（CMN）
- 模型吃的是 fbank，不吃波形。余弦是两条 embedding 的相似度，阈值 0.31 来自模型卡

过关：合上文件，能把形状变化讲一遍。讲的时候卡住，就还没过。

## 周三 11/25　PyTorch 推理基线

自己写加载、fbank、前向、余弦、计时。建议落成 `python/fbank.py` 和 `python/infer_pt.py`，名字可以改，笔记里写死路径。

加载时以 bin 里的 key 为准。`load_state_dict` 报 missing 或 unexpected，把 key 打出来再改构造参数。`config.yaml` 里的 `memory_efficient` 和权重不一定是同一个选择。

`results/day3.md` 写上：

- 冻结的计时协议
- 同一人余弦、不同人余弦
- T=300、T=800、三条语音的中位数时延
- fbank 单独的时延

过关：闸门通过，表上每个数都能再说一次是怎么来的。这张表是后面所有行的分母。

## 周四 11/26　导出 ONNX，用 Runtime 推理

自己写 `python/export_onnx.py`。导出前先在笔记里写下你选择的 opset、`dynamic_axes`、`do_constant_folding` 是开还是关。折叠先关掉，图优化留到周六，避免两件事叠在一次导出里。

Netron 装在 3.11 环境里：`pip install netron`，然后 `netron onnx/campplus.onnx`。在图上标出和周二那张图对应的段，尤其是 `StatsPool` 变成了哪些算子。

同一条计时协议跑 ONNX Runtime，`graph_optimization_level` 今天固定为关闭，这样周四的数字是“未优化的 ORT”。和 PyTorch 比最大绝对误差，同时再看余弦闸门。误差阈值你自己定，并写一句：这个阈值为什么没有把声纹比坏。

尾帧和 PyTorch 对不齐时，去 `layers.py` 里看池化对 padding 的处理，再决定导出时改哪一个参数。

过关：至少两个 T 都能跑；`results/day4.md` 有误差、余弦、时延，以及一张你标过的结构对照。

## 周五 11/27　Q8

读 `onnxruntime.quantization.quantize_static` 的参数，自己写 `python/quantize_q8.py`。笔记里先写原理，再写代码里哪一行对应哪一句：

- 这是训练后的静态量化，权重已经固定，不再反传
- 校准集用来估计激活的取值范围。写出你用了哪些样本，为什么三条语音不够或够
- 对称量化：零点在 0
- 权重 per-channel：每个输出通道一把 scale
- `QuantFormat.QDQ`：图上插入 `QuantizeLinear` / `DequantizeLinear`，算子本体仍可以是浮点节点，等运行时再融合

Netron 打开量化后的图，数出这两类节点。用同一协议测时延，并再测余弦闸门。

变慢就照实写。周五不修，周六日专门查。禁止改成“看起来更快”的另一套计时。

过关：`results/day5.md` 能用一段话讲完量化过程，图上的节点数和时延、余弦都在。

## 周六 11/28　图优化

只用周四导出的那张 FP32 图。四个 `SessionOptions.graph_optimization_level`：

- `ORT_DISABLE_ALL`
- `ORT_ENABLE_BASIC`
- `ORT_ENABLE_EXTENDED`
- `ORT_ENABLE_ALL`

每一级用 `optimized_model_filepath` 把优化后的图存下来。Netron 里数节点，写出被消掉或融合的算子（常量折叠、Conv 与 BatchNormalization 这一类）。四级时延用同一协议。

`onnxsim` 如果要用，单独一行，不和 Runtime 的四级混在一起。

某一级变慢，当天写下假设：融合没发生、线程在小矩阵上过多、动态轴让某种融合失效。周日用图来证实或推翻。

过关：`results/day6.md` 有四级的节点数和时延。

## 周日 11/29　Q8 加上图优化，收总表

矩阵至少这四格，协议与周三相同：

| | 关闭图优化 | ENABLE_ALL |
|---|---|---|
| FP32 ORT | 周四的数，可复测 | 周六的数 |
| Q8 | 周五的数，可复测 | 今天测 |

打开 Q8 在 `ENABLE_ALL` 之后存下来的图。看 `QuantizeLinear` / `DequantizeLinear` 还在不在，整数量化卷积有没有出现。Q8 不降反升时，先回答这个，再谈别的。

`results/week.md` 是唯一的总表。行至少包括：PyTorch、ORT FP32 四级、Q8 关闭优化、Q8 全开优化。每行有中位数、相对 PyTorch 的比值、余弦闸门是否通过。变慢的行各写一段：现象、图上看到的节点、结论。

过关：合上笔记讲 7 分钟，七件收获每件都有你自己的数字或一张图当证据。

## 变慢时按这个顺序查

查完一条再查下一条，每一条都在笔记里留一句“看过了，结果是”。

1. 这一轮有没有把会话创建、图优化、第一次运行算进去。
2. 线程数是不是中途变了。小 GEMM 上线程变多，时延会涨。
3. 比的是不是同一个 T。真实语音的帧数不同，毫秒不能横比。
4. FP32 图优化：Netron 里 Conv 和 BatchNormalization 有没有合成一个节点。没有，Extended / All 就不会更快。
5. Q8：全开优化之后，图上是整数量化卷积，还是浮点 Conv 两边仍套着 Quantize / Dequantize。套着的时候，量化只增加了节点。
6. 动态轴是否让某一级融合直接跳过。用一个固定 T 再导一张静态图，只做对照，不替换总表里的动态图结果，除非你把协议改掉并整表重测。
7. 闸门。余弦过了 0.31 的那一档，时延再低也从总表的“可用”列划掉。
