# transformer_cpp

本目录是独立的微型 Transformer **训练演示**项目。Cursor 配置全部在本目录 `.cursor/` 下，不依赖仓库根的推理课。

- Skill：`.cursor/skills/cpp-transformer-train/SKILL.md`
- Subagent：`.cursor/agents/cpp-transformer-tutor.md`（聊天 `/cpp-transformer-tutor`）
- Rule：`.cursor/rules/cpp-transformer-train.mdc`
- 课表：`PLAN.md`、`PROGRESS.md`

整节课钉住教学协议：`/cpp-transformer-train`，再 Option+Enter（Mac）或 Alt+Enter（Windows）开成 Custom Mode。

代码由学员手写。Agent 只给改法与验收，不代写 `.cpp/.h`。Eigen；按层手写反向；SGD。一次一课。毕业：大约十句话过拟合并 greedy 复述。
