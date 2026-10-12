# engine

L06 起在这里建你自己的 C++17 推理器。L06 之前这个目录只有本说明。

目标文件（名字可以不同，职责不要并成一个巨函数之后无法单测 GEMV）：

| 职责 | 出现于 |
|------|--------|
| 读 bin、打印 Config、RAII 持有权重 | L06 |
| Tensor：data / shape / stride / 是否拥有内存 | L07 |
| 无 cache 的整网前向与 greedy | L08 |
| `gemv_naive` 与 `gemv_avx2` | L09–L10 |
| KV cache 与同长 batch | L11–L13 |

验收标准在 `learn/PLAN.md` 每一课，接口细节在 `.cursor/skills/cpp-infer-expert/`。导师只讲改哪里，不替你提交实现。
