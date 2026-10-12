# S1 llama2.c：计算与内存

对照仓库：https://github.com/karpathy/llama2.c  
放到 `third_party/llama2.c`。权重用上游 README 里的 stories15M.bin。

## 先跑

按上游 `Makefile` 编译 `run`，用 stories15M 生成一段文本。把命令、权重路径、一段摘录写入 PROGRESS。

这一课不要改 `run.c`。

## 前向要画清的链

以 `run.c` 里的 forward 为准，笔记按真实调用顺序写，形状用 Config 的名字，不要写死数字：

1. token → embedding 行
2. 每一层：RMSNorm → Q、K、V 的 matmul → RoPE → 注意力（含 mask 与 `1/sqrt(head_size)`）→ 输出投影 → 残差
3. FFN：RMSNorm → w1 / w3 → 激活（上游是 SiLU 门控）→ w2 → 残差
4. 最终 RMSNorm → 分类器 logits

`n_heads` 与 `n_kv_heads` 可以不同。笔记里要写出 `head_size`，以及 K/V 的宽度是 `n_kv_heads * head_size`。

## 内存要画清的三块

让学员从 `Config`、`TransformerWeights`、`RunState`、`memory_map_weights` 自己抄字段顺序。

- **权重**：一个连续 `float` 数组。每个张量是一段偏移，不是各自 `malloc`。
- **激活**：`x`、`xb`、`hb` 这类缓冲在层与层之间复用。推理不必像训练那样留下每一层的激活图。
- **KV**：上游 `RunState` 里已有 cache。L04 笔记要算出它的元素个数（层 × 序列 × 每 token 的 K 或 V 宽度）。L08 的 C++ 先不要移植它。

过关时学员能写出「第 L 层 `wq` 起始元素下标」的公式，并把本模型 Config 代进去得到一个整数。公式必须和 `memory_map_weights` 的顺序一致。导师发现顺序抄错时，指出函数名，让学员重算，不要把下标算好递给他。

## 四个必须能闭卷的式子

- RMSNorm：`x / sqrt(mean(x^2) + eps) * weight`
- matmul：`xout[i] = dot(W[i], x)`，`W` 行主序 `(d, n)`
- RoPE：作用在 Q、K 的成对分量上，位置是 token 的绝对下标
- 注意力：分数除以 `sqrt(head_size)`，因果 mask 把未来位置丢掉后再 softmax

## 验收

- L02：`./run` 打出故事文本
- L03：`notes/L03-forward.md` 从 token id 走到 logits
- L04：`notes/L04-memory.md` 含三块内存和 `wq` 下标公式
- L05：闭卷讲完上面四个式子
