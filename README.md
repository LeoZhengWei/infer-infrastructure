# infer-infrastructure

用 C/C++ 把一次 Llama 前向做实：看清计算和内存，对拍 token，再用数字证明 AVX2 和 KV cache。

依据：`推荐项目.txt`。三份材料按这个顺序用，不并列当成三套框架去装：

1. **llama2.c** — 无依赖的前向，权重是一块内存
2. **tiny-cnn** — 算子、布局、模板分发怎么放进框架
3. **llama.cpp 核心** — 只读张量与内存池、KV、量化块

仓库内置导师 Agent（会把「先啃 llama.cpp / 先写 CUDA」拉回当前课）：

- 用 Cursor **打开本目录** `infer-infrastructure` 再开 Agent
- 课表：`learn/PLAN.md`（一次一课，从 L01 的开场白开始）
- 毕业：`learn/GOALS.md`
- 进度：`learn/PROGRESS.md`
- 技能：`.cursor/skills/cpp-infer-expert/`

学员代码放在 `engine/`（L06 才建）。笔记放在 `notes/`。上游克隆放在 `third_party/`，不入库。

和 `llms-zth` 的分工：那边是 PyTorch 小模型的训练和多运行时；这边是同一类推理在 CPU 上的内存和指令。
