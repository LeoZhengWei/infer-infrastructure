# transformer_cpp 学习计划（按课推进）

原则：一次只上一课。矩阵用 Eigen，反向按层手写，优化器用 SGD。目标见 `readme.md`：能学到大约十句话即可。

课表详解：`.cursor/skills/cpp-transformer-train/references/curriculum.md`。进度勾选：`PROGRESS.md`。配置在本目录 `.cursor/`。聊天：`/cpp-transformer-train` 或 `/cpp-transformer-tutor`。

## 怎么上每一课

1. 打开本目录（可单独作为工作区）。
2. 把该课 **开场白** 原样发给 Agent。
3. 只追求该课过关标准。代码自己手写。
4. 让 Agent 更新 `PROGRESS.md`。你自己在本文件课号前打勾。

---

## 当前课

**T00**（工具链与 Eigen）

---

## 主干（必须按序，T00–T09）

### T00 · 工具链与 Eigen

- 开场白：`按 transformer_cpp 计划上 T00。只做 CMake+Eigen 一次 matmul。不写网络。`

### T01 · 十句话语料

- 开场白：`按计划上 T01。只做十句语料和字符级编解码。不写模型。`

### T02 · 线性层 + SGD

- 开场白：`按计划上 T02。只做线性层手写反向和 SGD。不上注意力。`

### T03 · Softmax 与交叉熵

- 开场白：`按计划上 T03。只做 softmax+交叉熵并过拟合一个分类样本。`

### T04 · 单头因果 Attention 前向

- 开场白：`按计划上 T04。只做单头因果注意力前向并对拍。不写 backward。`

### T05 · Attention 反向

- 开场白：`按计划上 T05。只做注意力反向并用有限差分抽查。不接整网。`

### T06 · FFN 与 LayerNorm

- 开场白：`按计划上 T06。只做 FFN 和 LayerNorm 的前向反向抽查。`

### T07 · 一块 Decoder

- 开场白：`按计划上 T07。只拼一块 decoder 并保证前向可复现。不开 train loop。`

### T08 · 极小 LM + SGD 训练环

- 开场白：`按计划上 T08。接训练环和 SGD，证明 loss 下降。不要求会复述。`

### T09 · 过拟合十句话

- 开场白：`按计划上 T09。过拟合十句话并记录 greedy 复述。不要加通用优化器。`
