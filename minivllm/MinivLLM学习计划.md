# MinivLLM 学习计划（本机版）

> 仓库：`github.com/Wenyueh/MinivLLM`（基于 nano-vllm）
> 本地副本：`/home/hjadmin/zw/MinivLLM`（`origin/main` @ `00db3b4`，3258 行 Python）
> 本机：Ubuntu 22.04 · RTX 4090 24G + RTX 5090 32G（无 NVLink） · 32 核 · 62G 内存 · 根盘剩 70G
> 计划周期：**9 个半天 / 约 21 小时**，跨 7 周分两段（S1–S3 先做；S4–S9 等主干 L11/L12 完成后接着做），不占主干课日
> 本机版制定：2026-10-21

---

## 0. 先看四个事实（决定你怎么学）

### 事实 1：本机能跑，原计划里"Windows 跑不起来"那一整套作废

原计划按 Windows 主机写的：WSL2、远程 Linux、传文件、`apt install nsight-systems`——**全部不需要**。

本机实际条件（2026-10-21 15:45 实测）：

| 项 | 值 | 对学习的影响 |
|---|---|---|
| 系统 | Ubuntu 22.04，Linux 原生 | Triton / NCCL 直接可用，无兼容层 |
| GPU 0 | RTX 4090 24G，sm_89，**空闲 6.2G**，被 `affective_benchmarks` 占 17.8G（已跑 5h43m） | 主力卡；功能实验都放这里 |
| GPU 1 | RTX 5090 32G，sm_120，**空闲 4.5G**，被 `VLLM::EngineCore` 占 27.1G | 只做"能不能跑"验证，不在它上面报性能 |
| 互联 | 两卡都在同一 PCIe host bridge（PHB），**无 NVLink** | TP=2 可以跑，但通信走 PCIe，别指望加速 |
| 驱动 / 工具链 | 驱动 595.91.07（CUDA 13.2 API）；本机 toolkit 是 CUDA 12.8，`nsys`/`nvcc` 在 `/usr/local/cuda-12.8/bin`（**不在 PATH**） | 不用装 CUDA，只要把 PATH 补上 |
| Python 环境 | `.venv` 是空的（84K），uv 0.13.4，Python 3.11.12 | 从 `uv sync` 开始，约 15 分钟 |
| 锁定的版本 | `uv.lock`：torch **2.9.1**（cu128）、triton **3.5.1**、transformers 4.57.3 | 2.9.1 的 cu128 轮子覆盖 sm_120，5090 按版本应可用，冒烟验证即可 |
| 下载 | `huggingface.co` 直连超时；`HF_ENDPOINT=https://hf-mirror.com` **已在环境变量里**，三个候选仓库均 200 | 模型和 tokenizer 都能下 |
| 磁盘 | 根盘 913G，已用 92%，剩 70G；HF 缓存已占 35G | 装完这套约 +10G，够，但别乱存 checkpoint |

两条本机特有的账，先记住：

1. **仓库默认 fp32**（`main.py` 没有 `set_default_dtype`），Qwen3-0.6B 权重就要 **~2.4G 显存**。4090 剩 6.2G 够；5090 剩 4.5G 很紧。
2. `allocate_kv_cache` 里的 `gpu_memory_utilization` 乘的是 **`torch.cuda.mem_get_info()` 的当前空闲**（`model_runner.py:197-204`），不是总显存。所以共享卡上它不会抢邻居，但 KV 池大小会随邻居波动——**两次运行的 `max_cached_blocks` 可能不同，性能对比必须记下这个数**。

选卡约定：

```bash
export CUDA_VISIBLE_DEVICES=0     # 4090，默认
# 只有验证"5090 能不能跑"时才用 1；TP=2 用 0,1
```

> 如果 5090 上的 vLLM 是你自己起的服务，把它的 `gpu_memory_utilization` 调到 0.4 就能空出 ~10G；不确定是谁的进程就别动。

### 事实 2：这是一份"能跑但没打磨完"的教学仓库

它同时具备两个身份：

- **好教材**：从零实现 PagedAttention（Triton kernel）+ FlashAttention（Triton varlen kernel）+ 连续批处理 + prefix cache + CUDA graph + 张量并行，配 3.9 万字中文文档、进度标记、课后练习。
- **有 bug 的原型**：CUDA graph 路径当前跑不通，而且**卡点有两层**——所有入口默认 `enforce_eager: True` 绕过了它：
  1. `model_runner.py:407` 读 `config['max_num_seqs']`，但**所有入口配置的键都叫 `max_num_sequences`** → 一开图就 `KeyError`（5.1a，原计划漏了这一个）；
  2. 修掉键名后，才会撞到原计划写的那句形状不匹配（5.1）：`outputs` 按 `vocab_size` 建，模型却返回 hidden。

  还有一处文档级的坑：`SamplingParams.__post_init__` 明确 `assert temperature > 1e-10`（`sampling_parameters.py:11`），**"temperature=0 保证确定性"在本仓库不成立**（5.13）。验证图化正确性得换方法（见 Day 11）。

这不影响学习价值，反而提高了——你会遇到真实工程里最常见的情况：文档说做了、代码写了、但没验证。第 5 节把已知问题列成了练习题，修完它们你才算真读懂。

### 事实 3：它不含量化、不含 chunked prefill、prefix cache 是死代码

- 无 GPTQ/AWQ/FP8
- 无 chunked prefill（长 prompt 整段进或排队）
- prefix cache 的 hash 表建了但**永不命中**（`block_manager.py:71` 释放时清空 `token_ids`，作者在注释里诚实承认跨序列复用未启用）

所以别指望"学完就能上线"。它的定位是**理解工业引擎的骨架**。

### 事实 4：这是"引擎线"，不是主干，别把 L01 挤掉

`learn/PLAN.md` 的主干是 S0–S4（L01–L13，约 5 周），`learn/PROGRESS.md` 目前还是"尚未开课 L01"；而 PLAN 自己也写明"连续批处理、分页 KV 的生产实现在 `llms-zth` 那条线"——**本计划就是那条线**。

排期规则（照做，别并课）：

1. **主干课日（周一/三/五）不碰本计划**。L01–L13 的节奏照旧，一周 3 课。
2. 本计划只用**非课日**：每周二晚 + 周六上午各一次，一次 2–3 小时，分两段合计约 7 周完成 9 次。
3. **S4 起（含 S4）必须等主干 L11/L12（KV cache 与对拍）完成**。没亲手写过 KV cache 就看 paged KV，只能背名词；写过之后再看，每一行都能对上自己的代码。
4. 如果某周主干卡住 >40 分钟，本周本计划直接跳过，不补。

> 一句话：**主干是"自己造"，这条线是"看工业界怎么造"**。顺序反了，两边都学不深。

---

## 1. 环境准备（30–40 分钟，一次做完）

### 1.1 每次开课前 30 秒预检

```bash
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
df -h /home | tail -1
cd /home/hjadmin/zw/MinivLLM && git status --short && git log --oneline -1
```

判定标准：目标卡空闲 ≥ 4G、利用率 < 10% 才做**计时**实验；否则只做功能实验，或改天。

### 1.2 装环境（本机验证过的路径）

```bash
cd /home/hjadmin/zw/MinivLLM
# pyproject 里的 vllm>=0.15.0 只有 benchmark_tps.py 用到，装它会多拉几个 G，还和 torch 2.9 依赖打架
sed -i '/"vllm>=0.15.0",/d' pyproject.toml
uv sync --python 3.11
```

如果下载慢，再加镜像（pip 已经配了 tuna，uv 没有）：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

验证（期望 torch 2.9.1 / cu128 / triton 3.5.1）：

```bash
uv run python -c "
import torch, triton
print(torch.__version__, torch.version.cuda, triton.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
"
```

- 4090 应打印 `(8, 9)`；5090 应打印 `(12, 0)`。
- 磁盘账：torch+triton 约 5G，模型 1.5G，uv 缓存 3–4G。装完 `uv cache prune --ci` 可回收一部分。

**不要**为了跑 `benchmark_tps.py` 把 vllm 装回来：vllm 0.15 与 torch 2.9 的依赖很容易打架，而这个脚本的价值（和 vLLM 比吞吐）在你还没修完图化路径之前没有意义。

### 1.3 第一次胜利：pytest（不碰 GPU、不碰权重，5 分钟）

```bash
uv run pytest tests/test_scheduler.py -v
```

这 8 个测试只 import `scheduler/sequence/block_manager`（依赖只有 xxhash + numpy），**不碰 GPU、不碰权重**，是仓库里唯一"装完就能过"的东西。先让它绿，你后面改 `can_append` 才有回归网。

### 1.4 第二次胜利：两个 benchmark（只用 torch+triton，不下模型，15 分钟）

```bash
# 4090 上有邻居，先用小尺寸，别一上来 32×2048
uv run python benchmark_decoding.py --batch-sizes 1 8 --seq-lens 128 512 --iters 50
uv run python benchmark_prefilling.py
```

跑之前先读脚本头注释：它把 naive / fast / Triton（旧、新）四版放在一起，**每次计时前都先对 fp32 参考做正确性检查**——这个习惯比数字本身值钱。

> `benchmark_prefilling.py` 最后一项是 1×4096，PyTorch 的 O(N²) 实现要占 1G+ 显存。4090 邻居还在时如 OOM，先注释掉 `benchmark(num_seqs=1, seq_len=4096, ...)` 那一行。

### 1.5 下模型（HF 直连不通，走镜像）

```bash
echo $HF_ENDPOINT          # 期望 https://hf-mirror.com，没有就 export
uv run python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-0.6B'))
"
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B   # 约 1.5G
```

注意两件事：

- `main.py:46` 的 `cache_dir=~/huggingface/Qwen3-0.6B/` 只影响 tokenizer（十几 M），权重走 hub 缓存。别去手动搬文件。
- `main.py` 用 `Path(model_name).name` 匹配模型（`model_runner.py:30-32`），所以配置里**必须保持 `Qwen/Qwen3-0.6B` 这种以模型名结尾的写法**，传本地路径时目录名也得叫 `Qwen3-0.6B`。

### 1.6 冒烟 + 记第一版数字

```bash
CUDA_VISIBLE_DEVICES=0 uv run python main.py
```

能打印出三段 completion 就算成功。**同时记下**：prefill tokens/s、decode tokens/s、`[Rank 0] Global max_cached_blocks`、以及当时 `nvidia-smi` 的邻居占用。这是你后面所有优化的基线，缺一个数就不可比。

### 1.7 nsys 已经在机器上，不用 apt

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
nsys --version
CUDA_VISIBLE_DEVICES=0 nsys profile -t cuda -o /tmp/minivllm_decode --force-overwrite true \
  .venv/bin/python benchmark_decoding.py --batch-sizes 1 --seq-lens 512 --iters 20
```

> 用 `.venv/bin/python` 而不是 `uv run python`：nsys 直接 trace 解释器，少一层进程。

- 在 **4090** 上抓，别在 5090 上抓——那边有常驻 vLLM，trace 会被别人的 kernel 污染。
- 邻居在跑的时候，`--stats=true` 的绝对耗时会虚高，只看**kernel 占比**和**间隙**，不看绝对数。

### 1.8 测量纪律（共享 GPU 专用）

1. 同一组 A/B **必须在同一次进程运行内交替做**，不要"昨天跑一次、今天跑一次"。
2. 用 `torch.cuda.Event` 计时，不用 `time.time()`；每项至少 3 次取中位数。
3. 每次测量把邻居占用写进笔记（`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`）。
4. 4090 报数字，5090 只报"能跑 / 不能跑 / 报什么错"。
5. 显存紧张时优先降 `max_num_sequences` / `max_model_length`，不要动 `gpu_memory_utilization` 去赌。
6. `@torch.compile`（`sampler.py`/`layernorm.py`）首次运行要编译 30–60 秒，别当成卡死；计时前先 warmup 3 步。

### 1.9 顺手做的两件事

1. **不要 `git init`**——这已经是 clone 下来的 git 仓库。改用分支：
   ```bash
   git checkout -b lab
   ```
   每修一个 bug 单独 commit，随时能 `git diff main` 看自己改了什么。
2. 复制一份 `main.py` 为 `main_lab.py`，所有实验都在 lab 里做，不动原文件。

---

## 2. 学完的标准（验收）

分三层，**任何一层没达到就是没学完**：

| 层级 | 标准 | 检验方式 |
|---|---|---|
| **L1 能讲** | 白纸上画出 prefill 和 decode 两条完整调用链，标出每一步的张量形状 | 不看代码讲 15 分钟，讲不下去的地方就是没懂 |
| **L2 能改** | 修好 5.1a + 5.1，`enforce_eager=False` 跑通；同输入下 eager 与图化的输出一致；加速比可测 | 4090 空闲窗口内，`benchmark` 或 `main_lab` 的 A/B 记录 |
| **L3 能迁移** | 给一个新模型（Llama3.2 或自己的 ASR decoder）接入这个引擎 | 完成文档课后练习，或用自己的模型跑通 |

**不要以"读完文档"为标准**——文档写得好，读完会产生"我懂了"的错觉。以"改坏一次再修好"为标准。

---

## 3. 路线总览（9 个半天）

原计划 14 天 × 2.5h ≈ 35h，和主干并行不现实。压缩成 9 次，内容不减，只是把可以连读的部分合并：

| 次 | 原 Day | 内容 | 时长 | 前置 |
|---|---|---|---|---|
| S1 | Day 1–2 | 端到端追踪（先看森林） | 2.5h | 环境就绪 |
| S2 | Day 3–4 | linear / TP / Norm / sampler | 2.5h | S1 |
| S3 | Day 5–6 | Attention 四遍读 + 两个 benchmark + nsys | 3h | S2 |
| S4 | Day 7–8 | Sequence / BlockManager / Scheduler + pytest | 2.5h | **主干 L11/L12 之后** |
| S5 | Day 9 | ModelRunner（信息密度最高） | 2h | S4 |
| S6 | Day 10 | 读懂 CUDA graph 捕获 | 1.5h | S5 |
| S7 | Day 11 | 修 5.1a + 5.1，图化跑通（核心动作） | 2.5h | S6 |
| S8 | Day 12–13 | 3 个性能问题 + 新模型接入 | 3h | S7 |
| S9 | Day 14 | 迁移到音频 / ASR 接口设计 | 1.5h | S8 |

排期：第 1–2 周先做 S1–S3（周二晚 + 周六上午）；等主干 L11/L12 对拍完成，再从下一个周二晚起做 S4–S9（每周二晚 + 周六上午，约 3 周）。**4090 邻居一释放就优先把 S3/S7/S8 提前做**，这三课需要相对干净的卡。

**为什么阶段 A 在前？** 文档的顺序是"造引擎"的顺序（自底向上），但从零读 544 行的 Triton attention kernel 会直接劝退。先用 2 天建立"一个请求怎么变成一个 token"的全景，后面每个文件你都知道它在地图上的位置。

---

## 4. 逐阶段详解

### 阶段 A（S1，原 Day 1–2）：端到端追踪

**目标**：能在不读任何 kernel 的情况下，画出"prompt → token"的完整路径。

**读**（只读主干，跳过所有 `@triton.jit`）：

| 文件 | 行 | 看什么 |
|---|---|---|
| `main.py` | 1-50 | config 字典的每个键对应什么（尤其 `max_num_batched_tokens` / `max_model_length` / `block_size`） |
| `engine/llm_engine.py` | 25-112 | `generate()` → `step()` 的主循环，这是引擎的心跳 |
| `engine/model_runner.py` | 386-397 | `run()`：准备数据 → 跑模型 → 采样，三步 |
| `engine/scheduler.py` | 35-93 | `schedule()` 返回 `(seqs, is_prefill)`，**这是理解整个引擎的钥匙** |

**必答问题**：

1. `LLMEngine.generate()` 的 while 循环在等什么？什么时候退出？
2. `Scheduler.schedule()` 为什么**先**扫 `waiting` 队列、**后**扫 `running` 队列？如果反过来会怎样？
3. 一次 `step()` 里，`is_prefill=True` 和 `False` 分别处理多少 token？（提示：`llm_engine.py:82`）
4. `Sequence.num_cached_tokens` 是谁写的？它和 `block_table` 什么关系？
5. 为什么 `prepare_prefill` 要传 `cu_seqlens_q` 和 `cu_seqlens_k` 两个不同的累积长度？（这一问现在答不上没关系，记下来，S4 回来答）

**动手实验**：

在 `step()` 里（`llm_engine.py:69-87`）`schedule()` 之后、`postprocess()` 之前各加一行 `print`，打印 `is_prefill`、batch 大小、token 数。跑 3 条 prompt，数一下**一共几次 prefill step、几次 decode step**。

然后把 `max_num_batched_tokens` 从 1024 改成 4096、`max_num_sequences` 改成 16，看 prefill step 数怎么变。注意 `max_num_batch_tokens`（默认 4096）是另一个键，warmup 用它算峰值显存，调参时两个都要看（5.12）。

> 这个实验会让你第一次直观感受"连续批处理"是什么。记数字前先确认 4090 利用率 < 10%，否则 step 数对但时间不对。

**交付物**：一张手绘图，两条链路（prefill / decode），标出经过的文件名。

---

### 阶段 B（S2–S3，原 Day 3–6）：自底向上精读 layers → model

每天读文档对应章节 + 代码，**必须对照 HF 实现验证数值**。

#### Day 3：线性层与张量并行（`layers/linear.py` 484 行）

| 类 | 行 | 要点 |
|---|---|---|
| `LinearBase` | 6-62 | 权重加载的 `weight_loader` 钩子怎么工作 |
| `ReplicatedLinear` | 64-82 | 不切分 |
| `ColumnParallelLinear` | 84-111 | 按**输出维度**切 |
| `MergedColumnParallelLinear` | 113-151 | `gate_up` 融合（一次 GEMM 算两个） |
| `QKVColumnParallelLinear` | 152-198 | Q/K/V 打包成一个 GEMM |
| `RowParallelLinear` | 199-224 | 按**输入维度**切，出口 `dist.all_reduce`（`:225`） |

**必答**：

1. 为什么 Column Parallel 之后接 Row Parallel 只需要**一次** all_reduce？（提示：中间那层的每个 rank 算的是部分和）
2. `qkv_projection` 为什么把 Q、K、V 三个权重拼成一个？（答案：kernel 派发次数从 3 降到 1）
3. `packed_module_mapping`（`models/qwen3.py:286-292`）在权重加载时起什么作用？

**实验（本机版换成真 TP=2）**：原计划是"world_size=1 时注释掉 `dist.*` 看能不能跑"，本机有双卡，直接做真的更有价值：

```bash
# main_lab.py 里 'world_size': 2
CUDA_VISIBLE_DEVICES=0,1 uv run python main_lab.py
```

观察三件事，写进笔记：

1. 两卡是**异构**的（4090 + 5090，无 NVLink，走 PCIe PHB）。`allocate_kv_cache` 里的 `all_reduce(MIN)`（`model_runner.py:224-237`）会取两 rank 的较小值——看日志里 `Local max_cached_blocks` 两个数是否不同，`Global max_cached_blocks` 是不是等于小的那个。**这段注释就是为你这台机器写的**。
2. 0.6B 模型 TP=2 大概率**比单卡慢**：模型太小，all_reduce 的通信开销盖过分片收益。把 `world_size=1` 与 `2` 的 decode tokens/s 各跑 3 次记下来，结论写"小模型 TP 不划算，工业界只在装不下或大模型时才用"。
3. 邻居在跑时，rank1（5090，剩 4.5G）最容易 OOM。OOM 也是结论：**TP 的显存下限由最穷的 rank 决定**。

> 两个坑：`dist.init_process_group` 把端口写死成 `tcp://localhost:12345`（`model_runner.py:26`），上次残留的 worker 不退出会直接占端口，先 `ps -ef | grep main_lab` 查一下；world_size=1 也会初始化 NCCL，会多占几百 M 显存，算账时别忽略。

> 这一课的目标不是跑出加速，是拿到"TP 的代价"这手体感。主干线不教这个，只有这里能补。

#### Day 4：RMSNorm + 残差融合 + 采样器

| 文件 | 行 | 要点 |
|---|---|---|
| `layers/layernorm.py` | 16-34 | `residual_rms_forward`：**把 `x+residual` 和 RMSNorm 合成一次**，并返回新的 residual |
| `models/qwen3.py` | 197-225 | DecoderLayer 怎么把 residual 一路"传下去"而不是层层相加 |
| `layers/sampler.py` | 1-18 | `probs / Exp(1)` 再 argmax 的**技巧采样**（等价于多项式采样，但不用 cumsum） |

**必答**：

1. 为什么 residual 要**在层与层之间传递**而不是每层 `x = x + attn(norm(x))`？（提示：省一次全量读写；`norm(x+residual)` 的输入可以直接是上一步的输出）
2. `sampler.py` 里 `probs.div_(exponential_(1)).argmax()` 为什么等价于按概率采样？（这一问很漂亮，值得推 10 分钟）
3. `layernorm.py:16` 的 `@torch.compile` 和后面 CUDA graph 会怎么相互作用？（记下来，Day 10 回来答）
4. 附加：`SamplingParams` 为什么禁止 `temperature=0`（`sampling_parameters.py:11`）？想做 greedy 怎么办？（提示：除以一个极小正数，或直接对 logits 取 argmax——但后者要另写代码，不在本仓库里）

#### Day 5–6：Attention —— 全仓库最难也最值钱的部分（`layers/attention.py` 544 行）

**分四遍读，每遍只看一件事**：

| 遍 | 范围 | 只看 |
|---|---|---|
| 1 | `store_kvcache_kernel` (`:7-66`) | KV 怎么按 `slot_mapping` 写进巨大的 cache 池 |
| 2 | `flash_attention_varlen_kernel` (`:111-211`) | prefill 怎么处理"多序列拼在一起"（`cu_seqlens`） |
| 3 | `paged_attention_decode_kernel` (`:283-394`) | 在线 softmax（`m_i`/`l_i`/`alpha` 递推）+ block_table 逐 token 寻址 |
| 4 | `Attention.forward` (`:472-518`) | 一个类里怎么分流 prefill / decode 两种模式 |

**必答**（这 6 个答不上就回去重读）：

1. KV cache 的物理布局是 `(2, layers, blocks, block_size, kv_heads, head_dim)`（`model_runner.py:247`），为什么把 `blocks` 和 `block_size` 分成两维，而不是直接 `(blocks*block_size)`？
2. `slot_mapping` 和 `block_table` 分别解决"写"和"读"，为什么必须有**两个**映射？
3. `paged_attention_decode_kernel` 为什么是 `grid = (batch_size, num_heads)`？为什么不用 split-K？（答案：教学简化，生产 kernel 会 split-K 提高并行度）
4. 在线 softmax 里 `alpha = exp(m_i - m_i_new)` 是在修什么？（答案：新块的最大值变大时，要把旧累加器整体缩小）
5. 为什么 kernel 里 `physical_block = tl.where(valid, physical_block, 0)`——把无效 lane 指向 block 0？（`:353-356`，注释写得很清楚：地址运算不能越界）
6. GQA 在 kernel 里怎么实现的？（`:313` `kv_head_idx = head_idx // (num_heads // num_kv_heads)`）

**实验（这一天的重点，注意本机的两个约束）**：

```bash
# 1) 先小后大：4090 上还有邻居，直接 32×2048 容易 OOM
uv run python benchmark_decoding.py --batch-sizes 1 8 --seq-lens 128 512 --iters 50
uv run python benchmark_prefilling.py
```

跑完看四组实现的对比（naive torch / optimized torch / Triton old / Triton new）。**关键不是看谁快，而是回答：为什么 Triton 版在 batch 小的时候优势不明显？在序列长的时候优势最大？**

```bash
# 2) 只在 4090 上抓，且等邻居空闲；否则只看占比不看绝对数
export PATH=/usr/local/cuda-12.8/bin:$PATH
CUDA_VISIBLE_DEVICES=0 nsys profile -t cuda -o /tmp/decode --force-overwrite true \
  .venv/bin/python benchmark_decoding.py --batch-sizes 8 --seq-lens 512 --iters 20
```

看 paged attention kernel 的耗时占比。如果它 < 20%，说明瓶颈不在 attention（多半在 lm_head 或 LayerNorm），这个判断会直接决定你后面优化什么。

**本机换算练习**：按本机配置算一遍 KV 池。
`block_size=256`、28 层、8 个 KV head、`head_dim=128`、fp32 →
`256 × 2 × 28 × 8 × 128 × 4 B ≈ 56 MB/块`。
4090 装完 2.4G 的 fp32 权重后大约剩 3.5G，乘 0.9 后约能开 **55–60 块**，也就是 **~1.4 万 token** 的总容量。
这解释了为什么 `max_model_length=128` 时毫无压力，也解释了 5.10 那条"`block_size=256` 太粗"——每块 56M，碎片代价是真实存在的。

**交付物**：用文字写出"一次 decode step 的 KV 读写账本"：读哪些 block、写哪个 slot、各多少字节（按本机 56MB/块的数字算）。

---

### 阶段 C（S4–S5，原 Day 7–9）：引擎核心

#### Day 7：Sequence 与 Block（`sequence.py` 114 行 + `block_manager.py` 160 行）

`Sequence` 是一个**被精心设计过的数据结构**，每个属性都有用：

| 属性 | 行 | 含义 | 谁在用 |
|---|---|---|---|
| `num_tokens` | 28 | 总长度（prompt + 生成） | 到处 |
| `num_prompt_tokens` | 29 | 分界线 | `completion_token_ids` |
| `num_cached_tokens` | 31 | 已有 KV 的长度 | `prepare_prefill:283` |
| `block_table` | 33 | 逻辑块 → 物理块 | attention 读 KV |
| `num_cached_blocks` | 63-64 | `ceil(num_cached/blocksize)` | `prepare_prefill:289` |
| `last_block_num_tokens` | 71-72 | 最后一块用了几个 | `slot_mapping` 计算 |

**必答**：

1. `num_blocks` 用 `ceil`，`last_block_num_tokens` 又用 `num_tokens - (num_blocks-1)*block_size`，两者为什么能自洽？举个 `num_tokens=10, block_size=4` 的例子算一遍。
2. `Sequence.__getstate__/__setstate__`（`:88-114`）为什么 decode 阶段只传 `last_token` 而不是整个 `token_ids`？（提示：多进程通信，省带宽。本机 TP=2 时这条路径真的会走）
3. `BlockManager.allocate()` 里 `h = -1` 初值和"只有满块才算 hash"（`:87`）是什么用意？

#### Day 8：调度器（`scheduler.py` 118 行，最短但最烧脑）

**必答**：

1. `schedule()` 为什么"有 prefill 就返回，不掺 decode"？（`:53-54`）这是 **prefill-first** 策略，代价是什么？
2. 为什么 `waiting` 队列满时用 `break` 而不是 `continue`？（`:51-52`，提示：FIFO 公平性）
3. `can_append()`（`block_manager.py:128-136`）里为什么判断 `num_tokens % block_size == 1` 而不是 `== 0`？（注释解释了，务必读懂——这是最容易写错的一行）
4. `preempt()` 只做 deallocate + 重排队，KV 全丢。这意味着被抢占的序列要**重新 prefill**，代价多大？（按本机算：一条 128 token 的 prompt 重新 prefill，和把 `max_model_length` 调到 1024 后的一条长 prompt，差多少倍？）
5. `:80-91` 的 "no progress" 兜底为什么必要？（如果不抛异常，`generate()` 会死循环）

**实验**：

```bash
uv run pytest tests/test_scheduler.py -v      # 8 个测试，都是真实 bug 的回归测试
```

然后**故意改坏** `can_append` 的 `== 1` 为 `== 0`，看哪个测试挂——这会让你记住这一行。改完记得改回来并 commit 前后各一次，练习用 `git diff` 看自己的改动。

#### Day 9：ModelRunner（`model_runner.py` 457 行，全仓库信息密度最高）

按顺序读 5 段：

| 段 | 行 | 核心 |
|---|---|---|
| 初始化顺序 | 91-96 | `warmup_model()` → `allocate_kv_cache()` → `capture_cudagraph()`，**顺序不能换** |
| warmup | 186-194 | 跑一个最大 batch 的 prefill，目的是**测峰值显存**（不是为了热身） |
| KV 池分配 | 197-253 | `block_bytes` 怎么算；`:224-237` 的 `all_reduce(MIN)` 为什么必要 |
| 数据准备 | 265-343 | prefill 和 decode 各自拼哪些张量 |
| 执行 | 353-378 | prefill 直跑；decode 选图 + `copy_` + replay |

**必答**：

1. 为什么必须"先 warmup 再分配 KV"？顺序反了会怎样？
2. `allocate_kv_cache` 里 `total_free_mem - (peak - current)` 这个式子（`:200-204`）在算什么？（答案：给模型执行期间的临时张量留余量）
3. `prepare_prefill` 的 `slot_mapping` 为什么要区分"最后一个块"和"前面的块"（`:290-293`）？
4. `prepare_decode` 的 `slot_mapping` 为什么只有一个元素（`:326`）？
5. `block_tables` 为什么要 padding 到 `max_num_blocks` 且填 `-1`（`:299`、`:330`）？
6. **本机附加题**：`warmup_model` 的 `batch_size = max_num_batch_tokens // max_model_length = 4096 // 128 = 32`，已经超过 `max_num_sequences=16`。为什么这不报错？（提示：warmup 直接调 `self.run`，不经过调度器）由此说明 `max_num_batch_tokens` 和 `max_num_batched_tokens` 各自管什么。

**交付物**：把 `prepare_prefill` 用 `num_cached_tokens > 0` 的例子手工算一遍 `input_ids` / `slot_mapping` / `cu_seqlens`，和代码输出对照。

---

### 阶段 D（S6–S7，原 Day 10–11）：CUDA graph 专题（本仓库最有价值的两天）

#### Day 10：读懂捕获逻辑

**读**：`model_runner.py:399-458` + `utils/context.py`（27 行）

**必答**：

1. `batch_sizes = [1,2,4,8] + range(16, max_bs+1, 16)`（`:423`）为什么这么设计？（答案：小 batch 密集、大 batch 稀疏，覆盖常见情况又控制图数量）
2. 为什么 `reversed(batch_sizes)` 先捕**最大**的？（`:427`，答案：`graph_pool` 按第一个捕获的图规划内存池）
3. `graph_pool` 共享（`:441-444`）省了什么？（答案：小图复用大图的 workspace，否则 5~36 张图显存会炸）
4. `run_model` 里 `next(bs_ for bs_ in graphs if bs_ >= bs)`（`:365`）为什么向上取整？`bs=3` 时走哪张图？多余的 1 行怎么保证不污染？
5. 为什么 prefill **不**图化？（`:355`，答案：varlen，shape 不固定）
6. `set_context()` 这个**全局单例**（`context.py`）为什么这么设计？（答案：Triton kernel 的参数必须能从 Python 侧拿到；用全局变量而不是函数参数，是为了不改模型 forward 的签名。代价是**不可重入**）
7. **本机附加题**：`:407` 的 `self.config['max_num_seqs']` 从哪里来？对照 `main.py:16` 的键名，你现在应该能预测"一开图就崩在哪一行"。

#### Day 11：修 bug，让它真正跑起来（本计划的核心动作）

**这是全仓库最重要的一天。** 按顺序做：

**Step 1：确认卡点，先 A 后 B**

把 `main_lab.py` 里 `enforce_eager` 改成 `False`，跑：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python main_lab.py
```

你会先撞到 **5.1a**：`model_runner.py:407` 的 `KeyError: 'max_num_seqs'`。原因是所有入口配置都写 `max_num_sequences`（`main.py:16`），而捕获函数读的是另一个拼写。

改法二选一（推荐第一种，改动小）：

```python
# 方案 A：改读取方，和全仓库其他键保持一致
- max_bs = self.config['max_num_seqs']
+ max_bs = self.config['max_num_sequences']
# 方案 B：改所有入口的配置键，把 max_num_sequences 全改成 max_num_seqs
```

改完再跑，才会撞到原计划写的那一层 **5.1**：

```python
outputs = torch.zeros(max_bs, self.config['vocab_size'], ...)   # :420  按 vocab 分配
...
outputs[:batch_size] = self.model(input_ids[:batch_size])       # :439  但模型返回 hidden
```

`Qwen3ForCausalLM.forward` 返回的是 **hidden states**（`models/qwen3.py:336-338`），logits 是 `model_runner.py:376` 用 `compute_logits` 另算的。所以 `outputs` 应该是 `[max_bs, hidden_size]`。

> **这解释了一个谜**：为什么三个入口（`main.py:22`、`main_llama32.py:26`、`benchmark_tps.py:26`）**全部**设成 `enforce_eager: True`——因为图化路径没人跑通过。文档第 4.7 节讲得很完整，但代码没验证。**这是"文档 ≠ 现实"最典型的一课**，而且比原计划多一层：连 bug 都不止一个。

**Step 2：修**

```python
- outputs = torch.zeros(max_bs, self.config['vocab_size'], device=...)
+ outputs = torch.zeros(max_bs, self.config['hidden_size'], device=...)
```

顺手算一下账：这个 buffer 从 `16 × 151936 × 4B ≈ 9.7MB` 降到 `16 × 1024 × 4B = 64KB`。图化路径本来想省时间，结果先白占了 10M 显存——共享卡上这种账要会算。

**Step 3：验证正确性（比性能更重要）**

原计划说"用 `temperature=0` 保证确定性"——**本仓库做不到**，`SamplingParams` 直接 assert 拒绝（5.13）。用下面任一种替代：

1. **比 logits（最稳）**：同一条 prompt、同一个 `input_ids`，分别在 `enforce_eager=True/False` 下只跑一次 decode，打印 `logits[0, :8]`，比 `max abs diff`。期望 < 1e-5（fp32）。
2. **比 token（近似 greedy）**：把 `temperature` 设成 `1e-6`（assert 允许多于 1e-10），跑两次比 token ids。注意这仍然是采样，RNG 状态不同会有极小概率分叉，只适合做二次确认。
3. 无论哪种，都要确认 `slot_mapping` 的 padding 行清了（`:369` 的 `fill_(-1)`）——这是捕获到脏状态最常见的原因。

**Step 4：测收益**

```bash
# 同一进程内交替跑 eager / graph，各 3 次；不要分两天跑
# 记下：decode tokens/s、max_cached_blocks、邻居占用
```

预期：batch=1 时 1.2～2x，batch 大时收益更小（大 batch 本来就 GPU-bound）。**但在本机要打折看**：4090 当前被邻居占着 62% 利用率，派发瓶颈被别人的 kernel 掩盖，加速比可能接近 0。这时正确的结论是"**本机当前负载下测不出**"，换到邻居空闲的窗口重测，而不是硬凑一个数字。

**Step 5：加一个实验**

把 `batch_sizes` 改成 `[1,2,3,4,5,6,7,8,12,16]`，测"图数量 vs 显存 vs 平均加速"。画出曲线，回答：**分桶的边际收益在第几张图之后消失？**（每张图共享 `graph_pool`，所以显存不是线性涨——这也是要观察的点。）

**交付物**：一段 200 字的结论——"在我的 4090 上（邻居占 X G、利用率 Y%），图化带来 X 倍加速，因为 ……；分桶超过 N 张后收益消失，因为 ……"。

---

### 阶段 E（S8，原 Day 12–13）：性能优化实战

从第 5 节的已知问题里挑 3 个改，每个都做"A/B 对比 + 一句话解释"。

**推荐这三个（按性价比排序）**：

**① prefill 的 positions 每层重算（5.5）**

`qwen3.py:206-213` 在 28 个 DecoderLayer 里各做一次 `cu_seqlens_q.cpu().tolist()` → **28 次 GPU→CPU 同步**。

改法：在 `prepare_prefill`（`model_runner.py`）里算一次，存进 context（或直接用 `cu_seqlens` 做张量运算避免 `.cpu()`）。

预期：prefill 阶段明显变快。这是"减少主机同步"的教科书案例。测的时候用 `max_model_length=512`、batch=8，把同步的成本放大。

**② prefill 对全部 token 算 logits（5.6）**

`model_runner.py:359` `compute_logits(hidden_states)` 对 1024 个 token 全算。实际只需要每个序列的最后一个 token。

改法：按 `cu_seqlens_q` 取每个序列的最后一个 hidden 再算 logits。

预期：长 prompt 的 prefill 时间大幅下降（logits 是 `hidden × vocab` 的大 GEMM，fp32 下 1024×151936×1024 每次约 0.3 TFLOP，很贵）。

**③ torch.compile 动态 shape 抖动（5.7）**

`sampler.py:14` 和 `layernorm.py:16` 用了 `@torch.compile`，但 sampler 的输入 batch 每步都变 → 反复 recompile。

改法：加 `dynamic=True`，或者干脆去掉 `@torch.compile` 做 A/B。

顺带回答 Day 4 的第 3 问：`@torch.compile` 与 CUDA graph 叠在一起时，recompile 发生在图外还是图内？如果发生在图内，会不会让捕获失败？

**最后一个练习（文档课后题）：接入 Llama-3.2-1B**

`meta-llama/Llama-3.2-1B-Instruct` 是 **gated 仓库**，hf-mirror 也需要 HF token。本机改用未 gated 的镜像仓库，模型名后缀不变，`model_runner.py` 的 `Path(...).name` 匹配仍然成立：

```python
# main_llama32.py
- 'model_name_or_path': 'meta-llama/Llama-3.2-1B-Instruct',
+ 'model_name_or_path': 'unsloth/Llama-3.2-1B-Instruct',
```

（已确认 `hf-mirror.com/api/models/unsloth/Llama-3.2-1B-Instruct` 返回 200。）

**显存提醒**：1B 参数 fp32 约 4G，加 KV 和临时张量接近 5G。**5090 的 4.5G 空闲跑不动，只在 4090 空闲 ≥ 6G 时做**。想省显存可以在 lab 里 `torch.set_default_dtype(torch.bfloat16)`，但那会改变数值——先记 fp32 基线，再对比 bf16，别混在一起改。

做之前先 `mv src/myvllm/models/llama.py /tmp/`，逼自己从零写；写完再 `diff` 原文，看差在哪。

---

### 阶段 F（S9，原 Day 14）：迁移到音频/ASR

**不要跳过这一步**——这是你学这个仓库的真正目的。本机有两个现成条件：`/home/hjadmin/zw/faster-qwen3-tts` 就在工作区里；`~/.cache/huggingface/hub` 里有 Qwen3-ASR / Qwen3-TTS 的仓库引用（权重还没拉，用到时按需下）。

用一页纸回答三个问题：

1. **接口怎么接？** 你的 ASR/TTS 模型（Whisper / Qwen3-ASR / Qwen3-TTS）要接入这个引擎，`models/xxx.py` 需要实现哪几个接口？（对照 `qwen3.py`：`forward(input_ids)` 返回 hidden，`compute_logits(hidden)` 返回 logits）
2. **KV 语义对得上吗？** 音频模型的 KV cache 是否也是"每层 K/V、按位置索引"？如果它有 cross-attention（Whisper），cross-attn 的 K/V 不属于自回归部分，怎么放进 paged cache？（答案：不能放，得单独预计算成静态 buffer——这就是 faster-qwen3-tts `talker_graph.py:154 prefill_kv` 的思路：把 HF `DynamicCache` 一次性拷进 `StaticCache`，之后只动 `cache_position`。）
3. **哪些优化能直接搬？**（对照第 6 节表格打勾）

**本机对照阅读清单**（各 10 分钟，带着 MinivLLM 的问题去看）：

- `faster_qwen3_tts/predictor_graph.py:170-201`：为什么它只捕 **1 张图**而不分桶？（提示：TTS 的 batch 固定）
- `faster_qwen3_tts/talker_graph.py:154-170`：`prefill_kv` 怎么把变长 prefill 挡在图外
- `faster_qwen3_tts/model.py:248-255`：capture 的调用时机

**交付物**：一张"A 列 = MinivLLM 里的机制 / B 列 = 我的 ASR 用不用得上 / C 列 = 怎么改"的三列表。

---

## 5. 已知问题清单（14 项，当练习题用）

改之前先 `git checkout -b lab`，每修一个单独提交，方便回滚。

| # | 问题 | 位置 | 严重度 | 修法提示 |
|---|---|---|---|---|
| 5.1a | **图化路径读错配置键**：`config['max_num_seqs']`，而所有入口写的是 `max_num_sequences`，开图即 `KeyError` | `model_runner.py:407` vs `main.py:16` | 阻断（**先于 5.1 触发**） | 改读取方为 `max_num_sequences`，或统一全部入口键名 |
| 5.1 | **CUDA graph 的 outputs buffer 形状错**，图化路径跑不通 | `model_runner.py:420` vs `:376`/`:439` | 阻断 | `vocab_size` → `hidden_size`（详见 Day 11） |
| 5.2 | 三个入口都 `enforce_eager: True`，默认没走图 | `main.py:22`、`main_llama32.py:26`、`benchmark_tps.py:26` | 阻断 | 修完 5.1a/5.1 后改 False |
| 5.3 | `vllm>=0.15.0` 死依赖，只有 `benchmark_tps.py` 用 | `pyproject.toml` | 环境 | 删掉；别为了一个对比脚本把 vllm 装回来 |
| 5.4 | prefix cache 永不命中（释放时清 token_ids） | `block_manager.py:71`，原因见 `:62-70` 注释 | 功能缺失 | 需要 paged prefill kernel（cu_seqlens_q ≠ cu_seqlens_k）+ 位置偏移才能启用，别硬改 |
| 5.5 | prefill 的 positions 在每层重算，28 次主机同步 | `qwen3.py:206-213` | 性能 | 提到 `prepare_prefill` 算一次 |
| 5.6 | prefill 对全部 token 算 logits | `model_runner.py:359` | 性能 | 按 `cu_seqlens_q` 取每个 seq 的最后一个 token |
| 5.7 | `@torch.compile` 遇上每步变化的 batch size → 反复 recompile | `sampler.py:14`、`layernorm.py:16` | 性能 | 加 `dynamic=True` 或去掉 |
| 5.8 | 无 chunked prefill，长 prompt 会被整段判断卡住 | `scheduler.py:44` | 功能缺失 | 需要把 seq 切片多次调度 |
| 5.9 | preempt = 丢 KV 重算 | `scheduler.py:96-99` | 策略 | 可加 recompute 计数或 swap 到 CPU |
| 5.10 | `block_size=256` 过粗（vLLM 默认 16），fp32 下每块 56MB | `main.py:20` | 配置 | 改 16 对比 kernel 耗时与碎片率（每块降到 ~3.5MB） |
| 5.11 | padding 行白算完整 forward + 全词表 logits | `model_runner.py:369-376` | 小 | 修完 5.1 后才可见；关键是 `slot_mapping.fill_(-1)` 别丢 |
| 5.12 | **同一个参数有两种拼写**：调度器读 `max_num_batched_tokens`，warmup 读 `max_num_batch_tokens` | `main.py:17` / `main.py:37`，`model_runner.py:189` | 配置陷阱 | 调参时两个都改；否则 warmup 的峰值显存算的是旧值 |
| 5.13 | **`temperature=0` 被 assert 拒绝**，原计划的"确定性对拍"方法不成立 | `sampling_parameters.py:11` | 方法 | 用 1e-6 近似 greedy，或直接比 logits（推荐） |

**优先级**：5.1a → 5.1 → 5.2 → 5.5 → 5.6 → 5.7。前三个是"能不能跑"，后三个是"快不快"。

---

## 6. 与 faster-qwen3-tts 的衔接

学完这个仓库，回去看 `faster-qwen3-tts` 会突然变简单——因为你会发现**两者是同一个原理的两个极端**：

| 维度 | MinivLLM | faster-qwen3-tts（本机 `/home/hjadmin/zw/faster-qwen3-tts`） |
|---|---|---|
| KV | 分页池，多流共享 | 每路一份固定 `max_len` 的 `StaticCache` |
| 图数量 | **N 张**（按 batch 分桶） | **1 张**（batch 固定） |
| 变化量注入 | 5 个静态 buffer（含 `block_tables`） | 1 个 `cache_position` |
| 图内算什么 | forward + 全词表 logits | 只有 forward |
| 变长部分 | prefill 留图外 | prefill 留图外（`talker_graph.prefill_kv`） |
| 主战场 | 吞吐（token/s） | 延迟（TTFA/RTF） |
| 硬件 | 4090/5090 共享卡、fp32 | 单卡 bf16，延迟敏感 |

**两条共用的铁律**：

1. **变长留图外，定长进图。**
2. **图内不能新建张量、不能 `.item()`、不能有 Python 分支。**

这两句话是 21 小时里真正该带走的东西。

---

## 7. 术语速查

| 术语 | 一句话解释 | 代码位置 |
|---|---|---|
| `slot_mapping` | 新 token 的 KV 该写进哪个物理槽 | `prepare_prefill:269` |
| `block_table` | 逻辑块号 → 物理块号，读 KV 用 | `Sequence.block_table` |
| `cu_seqlens` | 多序列拼成一维后的累积边界 | `prepare_prefill:275` |
| `num_cached_tokens` | 已有 KV 的长度（prefix cache 命中部分） | `Sequence:31` |
| prefill-first | 有 prefill 就优先做，不混 decode | `scheduler.py:53` |
| graph bucket | 按 batch size 分桶各捕一张图 | `model_runner.py:423` |
| graph pool | 多张图共享的私有内存池 | `model_runner.py:441` |
| `max_cached_blocks` | 按**当前空闲显存**算出的 KV 块数，共享卡上会变 | `model_runner.py:197-253` |

---

## 8. 两条完整调用链（背下来）

### Prefill

```
LLMEngine.generate
  → add_prompt → Sequence(token_ids)
  → step()
      → Scheduler.schedule()          # waiting 队列: can_allocate → BlockManager.allocate → block_table
      → ModelRunner.run(seqs, True)
          → prepare_prefill()         # input_ids / slot_mapping / cu_seqlens_q,k / block_tables → set_context
          → run_model(True)
              → model.forward(1D input_ids)      # embed → 28×DecoderLayer
                  → LayerNorm(residual fused)
                  → Attention: qkv_proj(Column) → q/k norm → RoPE → store_kvcache(Triton) + flash_varlen(Triton) → o_proj(Row, all_reduce)
              → compute_logits()      # ← 当前对全部 token 算，5.6 要改
          → sampler()
      → Scheduler.postprocess()       # append_token → 检查停止 → deallocate
```

### Decode

```
step()
  → Scheduler.schedule()              # running 队列: can_append → BlockManager.append
  → ModelRunner.run(seqs, False)
      → prepare_decode()              # input_ids=last_token / slot_mapping(1个) / context_lens / block_tables(pad -1)
      → run_model(False)
          → 选图: graphs[最小的 >= bs]
          → graph_vars 各 buffer 切片 copy_（slot_mapping 先 fill_(-1)）
          → graph.replay()            # 图内: forward + store_kvcache + paged_attention(Triton)
          → compute_logits(vars['outputs'][:bs])   # ← outputs 必须是 hidden，5.1
      → sampler()
  → postprocess()
```

---

## 9. 每次课的 checklist

```
□ S1   画出两条链路图；数清 prefill/decode step 次数；记下第一版 token/s
□ S2   答出"为什么 CP+RP 只需一次 all_reduce"；TP=2 跑一次，记下 all_reduce(MIN) 与两个 rank 的块数
□ S3   跑两个 benchmark（小尺寸起步）；nsys 看 decode kernel 占比；算出 56MB/块
□ S4   pytest 8 项通过；故意改坏 can_append 并看测试挂；手工算 num_blocks
□ S5   手工算 prepare_prefill 的四个输出张量；答出 warmup batch=32 为什么不报错
□ S6   答出 reversed(batch_sizes) 的原因；答出 5.1a 会在哪一行崩
□ S7   ★ 修 5.1a + 5.1，图化跑通；logits 与 eager 对上；记下加速比与邻居占用
□ S8   改 3 个性能问题（A/B）；Llama-3.2-1B 跑通（4090 空闲 ≥ 6G）
□ S9   写出"我的 ASR 用不用得上"三列表；对照完 faster-qwen3-tts 三个文件
```

---

## 10. 本版相对上一版的改动（便于 diff）

1. 删除 Windows / WSL2 / 远程机器三选一：本机就是 Linux + 双卡，改为本机实测表与选卡约定。
2. 环境准备全部重写：`uv sync`（删 vllm 死依赖）、torch 2.9.1/triton 3.5.1、sm_120 验证、HF 镜像、磁盘账、nsys 路径、共享 GPU 测量纪律。
3. 14 天 → 9 个半天（约 21h），分两段：S1–S3 先做，S4–S9 等主干 L11/L12 之后再做。
4. 新增本机专属实验：真 TP=2（异构卡 + PCIe，验证 `all_reduce(MIN)`）、fp32 显存账、56MB/块的 KV 池估算。
5. 修 bug 顺序纠正：先 5.1a（KeyError）再 5.1（形状）；新增 5.12（配置键双拼写）、5.13（temperature=0 不可用），并给出可执行的正确性验证方法。
6. 迁移课改为对照本机 `faster-qwen3-tts` 的三个具体文件，而不是泛泛而谈。
