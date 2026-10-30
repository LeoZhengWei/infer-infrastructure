# 课表 T00–T09

原则：一次一课。矩阵用 Eigen，反向按层手写，优化器用 SGD。

## 怎么上每一课

1. 在本目录工作（可单独打开本目录为工作区）。
2. 把该课开场白发给 Agent（可先 `/cpp-transformer-train` 或开成 Custom Mode）。
3. 只追求该课过关标准。
4. 让 Agent 更新 `PROGRESS.md`。学员在 `PLAN.md` 课号前打勾。

卡超过 40 分钟：缩小到「先打印矩阵形状」或「先过拟合 1 个样本」。

---

### T00 · 工具链与 Eigen · 45 分钟

- 做：CMake 工程链接 Eigen。一个小程序：`A(2,2) * B(2,2)`，打印结果。
- 懂：本课用 `MatrixXf`，在 PROGRESS 写明采用 **行主还是列主**（推荐保持 Eigen 默认列主，乘法写成 `Y = X * W` 时约定 `X` 为 `(batch, in)` 还是反过来——必须选定一种，后续所有层遵守）。
- 过关：可执行文件输出与手算一致。
- 不做：网络、反向、优化器。
- 开场白：`按计划上 T00。只做 CMake+Eigen 一次 matmul。不写网络。`

### T01 · 十句话语料 · 45 分钟

- 做：文本文件写约十句短句。字符级（或空格分词，选简单的那种）建 vocab，`encode`/`decode` 往返。
- 过关：十句全部 encode 再 decode 与原文一致（含句末符若有）。vocab 大小写入 PROGRESS。
- 不做：embedding 训练。
- 开场白：`按计划上 T01。只做十句语料和字符级编解码。不写模型。`

### T02 · 线性层 + SGD · 60 分钟

- 做：`Y = X * W`（按 T00 约定）。MSE。手写 `dW`、`dX`。循环里 `W -= lr * dW`。
- 过关：固定一个小样本，训练数十步后 MSE 降到初始的 1/10 以下（数量级即可，记下两个 loss）。
- 不做：Adam、bias 抽象基类。bias 若加，就写 `Y = XW + 1*b`，梯度显式。
- 开场白：`按计划上 T02。只做线性层手写反向和 SGD。不上注意力。`

### T03 · Softmax 与交叉熵 · 60 分钟

- 做：对一行 logits 做 softmax + NLL。手写 `d_logits`。接到 T02 的线性层，单样本分类过拟合。
- 过关：训练后 argmax 等于目标类别；记下最终 loss。
- 不做：label smoothing、复杂 tokenizer。
- 开场白：`按计划上 T03。只做 softmax+交叉熵并过拟合一个分类样本。`

### T04 · 单头因果 Attention 前向 · 90 分钟

- 做：`Q=XW_q` 等；`scores = Q K^T / sqrt(d)`；下三角 mask；softmax；`Y = attn V`。用 2～4 的序列、很小的 `d` 与手算或 NumPy 对拍。
- 过关：输出 `max abs diff` 写入 PROGRESS（建议 `< 1e-5` 量级，同一公式同一布局）。
- 不做：反向、多头拼接（多头若做放到 T07）。
- 开场白：`按计划上 T04。只做单头因果注意力前向并对拍。不写 backward。`

### T05 · Attention 反向 · 90 分钟

- 做：对 T04 的权重手写 `backward`。用有限差分抽查 `Wq` 或 `W` 的若干元素。
- 过关：抽查元素满足 [teaching.md](teaching.md) 的阈值。
- 不做：通用 tape、融合 kernel。
- 开场白：`按计划上 T05。只做注意力反向并用有限差分抽查。不接整网。`

### T06 · FFN 与 LayerNorm · 75 分钟

- 做：`FFN = Linear-激活-Linear`。激活用 ReLU（教学优先）或 GELU。LayerNorm 手写前向+反向。各抽查数值梯度。
- 过关：两处抽查过阈值。
- 不做：RMSNorm 除非学员坚持并自己推梯度；不要两者都做。
- 开场白：`按计划上 T06。只做 FFN 和 LayerNorm 的前向反向抽查。`

### T07 · 一块 Decoder · 90 分钟

- 做：token embedding +（可选）位置 embedding 或正弦位置编码（选一种写进 PROGRESS）+ 一层：LN → 因果注意力 → 残差 → LN → FFN → 残差。多头若做：`d_model` 整除 `n_head`，拼回。
- 过关：固定输入、固定权重（可随机种子），两次前向输出一致。形状：`(seq, d_model)` 或 `(batch, seq, d_model)` 选定一种（教学可用 `batch=1`，矩阵当 `(seq, d)`）。
- 不做：训练循环。
- 开场白：`按计划上 T07。只拼一块 decoder 并保证前向可复现。不开 train loop。`

### T08 · 极小 LM + SGD 训练环 · 90 分钟

- 做：输出投影到 vocab。teacher forcing：输入 `x[:-1]`，目标 `x[1:]`。交叉熵对有效位置平均。SGD 更新所有权重。超参建议：`n_layer=1`，`d_model=32`，`n_head=2`，`seq_len≤32`。
- 过关：同一十句数据上，loss 相对第 1 步明显下降（记下 step0 与 stepN）。
- 不做：Adam、学习率 warmup、梯度裁剪框架（需要时一行 `if (g>clip) g=clip` 即可）。
- 开场白：`按计划上 T08。接训练环和 SGD，证明 loss 下降。不要求会复述。`

### T09 · 过拟合十句话 · 90 分钟

- 做：多训一些 step。greedy 或 argmax 解码：给句首若干字符（或 BOS），看能否复述训练句。
- 过关：约十句中 **至少 7 句** 能完整或接近复述；PROGRESS 记录最终 loss、句数、两句原文与生成对照。
- 不做：泛化测试集、beam search、加速。
- 开场白：`按计划上 T09。过拟合十句话并记录 greedy 复述。不要加通用优化器。`

---

## 明确不做（直到课表写到）

- autograd 引擎、算子注册、计算图
- Adam / 可插拔 Optimizer 层次
- 自制 Tensor 替代 Eigen
- 与 llama2.c、CUDA、AVX2 混线
- 「先写完框架再填训练」的大爆炸实现
