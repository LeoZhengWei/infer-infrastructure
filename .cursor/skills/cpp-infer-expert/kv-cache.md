# S4 KV cache 与静态 batch

L08 的参照物是「每步重算全部历史」。L11 在同一份 `engine/` 上加 cache，使第 t 步只为本步计算新的 K、V，历史从 cache 读。

## 形状

以本模型 Config 为准：

- `key_cache` 元素数：`n_layers * seq_len * (n_kv_heads * head_size)`
- `value_cache` 同理
- 第 t 步写入的槽位是绝对位置 `t`（从 0 计），不是每步都写到 0
- RoPE 的位置下标同样是 `t`

`n_kv_heads < n_heads` 时，注意力里把每组 K/V 重复用到对应的 query 头上。重复系数让学员从两个头数算出来。

## L11

先求能生成 16 token。这一课不对拍。cache 缓冲在生成开始时分配一次，步与步之间保留。

## L12

同一 prompt、greedy：

- 无 cache 路径（L08）
- 有 cache 路径

16 个 token id 必须完全一致。然后把两条路径生成 16 token 的时间写入基线表。prompt 很短时差距可能不大；把生成长度加到 64 再测一次逐步时间，cache 应更短。token 仍须一致。

## L13 静态 batch

只做 **batch = 2 且两条序列一样长**。先不要 padding，不要 mask 的变长分支，不要连续批处理。

每条 prompt 在 batch 里的 16 token，与它自己 batch=1 的结果一致。

cache 的 batch 维由学员自己加。验收看 token，不看吞吐框架。
