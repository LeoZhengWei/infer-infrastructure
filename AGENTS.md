# infer-infrastructure Agent

两条课不要混线。更具体的指令以当前目录的 `AGENTS.md` 为准。

## 默认：推理与 C++ 高性能

必读：`.cursor/skills/cpp-infer-expert/SKILL.md`，然后 `learn/PLAN.md` 与 `learn/PROGRESS.md`。

导师不修改作业代码。可更新 `learn/` 下 PLAN/PROGRESS。不要代写 `engine/`、`third_party/`、`notes/`。

主干：S0 → S1 llama2.c → S2 C++ 对拍 → S3 AVX2 → S4 KV cache。之后才是 tiny-cnn 与 llama.cpp 核心。贯穿 `stories15M`。简体中文。

## `cpp/transformer_cpp/`：独立训练项目

该目录自带 `AGENTS.md` 与 `.cursor/`（skill / subagent / rule）。在那里上课，不要用本仓库推理主干抢课。
