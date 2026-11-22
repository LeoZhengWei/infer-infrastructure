"""Ten steady-state rounds: FP32 vs W4A8 vs W4A4.

Warmup runs are discarded. Only the embedding forward is timed.
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "third_party" / "campplus"))

from export_onnx import load_model  # noqa: E402
from fbank import fbank, load_wav  # noqa: E402
from quant_conv import quantize_model  # noqa: E402


def sample_ms(fn, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def median(samples: list[float]) -> float:
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a * b).sum() / (a.norm() * b.norm()))


def main() -> None:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    warmup, repeat = 10, 10
    fp32 = load_model()
    w4a8 = quantize_model(load_model(), 8)
    w4a4 = quantize_model(load_model(), 4)

    torch.manual_seed(0)
    cases = [("randn_T300", torch.randn(1, 300, 80)), ("randn_T800", torch.randn(1, 800, 80))]
    feat = fbank(load_wav(str(ROOT / "audio" / "speaker1_a_cn_16k.wav")))
    cases.append((f"speaker1_a_T{feat.shape[0]}", feat.unsqueeze(0)))

    lines = [
        "# W4A8 / W4A4 与 FP32",
        "",
        "- 权重：每个输出通道对称 int4（-8..7）。",
        "- W4A8：激活按张量对称量化到 int8。W4A4：量化到 int4。",
        "- 卷积用 im2col + `torch._int_mm`（int8×int8→int32），再乘回两个 scale。BN / ReLU 仍是 FP32。",
        f"- 先跑 {warmup} 轮丢掉启动，再记 {repeat} 轮。中位数是这 {repeat} 轮的。",
        "",
    ]
    print("single forward check")
    x0 = cases[0][1]
    with torch.inference_mode():
        ref0 = fp32(x0)
        t0 = time.perf_counter()
        y8 = w4a8(x0)
        t8 = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        y4 = w4a4(x0)
        t4 = (time.perf_counter() - t0) * 1000
    print(f"cold T300 w4a8 {t8:.1f} ms cos {cosine(y8, ref0):.4f}  w4a4 {t4:.1f} ms cos {cosine(y4, ref0):.4f}")

    for name, x in cases:
        x = x.contiguous()
        with torch.inference_mode():
            ref = fp32(x)
            e8 = w4a8(x)
            e4 = w4a4(x)

        def run_fp(x=x):
            with torch.inference_mode():
                fp32(x)

        def run_8(x=x):
            with torch.inference_mode():
                w4a8(x)

        def run_4(x=x):
            with torch.inference_mode():
                w4a4(x)

        fp_s = sample_ms(run_fp, warmup, repeat)
        a8_s = sample_ms(run_8, warmup, repeat)
        a4_s = sample_ms(run_4, warmup, repeat)
        print(
            f"{name}: fp32 {median(fp_s):.2f}  w4a8 {median(a8_s):.2f}  w4a4 {median(a4_s):.2f}  "
            f"cos8 {cosine(e8, ref):.4f} cos4 {cosine(e4, ref):.4f}"
        )
        lines += [
            f"## {name}（T={x.shape[1]}）",
            "",
            f"与 FP32 的余弦：W4A8 {cosine(e8, ref):.4f}，W4A4 {cosine(e4, ref):.4f}。",
            "",
            "| 轮次 | FP32 ms | W4A8 ms | W4A4 ms |",
            "|---:|---:|---:|---:|",
        ]
        for i in range(repeat):
            lines.append(f"| {i + 1} | {fp_s[i]:.2f} | {a8_s[i]:.2f} | {a4_s[i]:.2f} |")
        lines.append(f"| 中位数 | {median(fp_s):.2f} | {median(a8_s):.2f} | {median(a4_s):.2f} |")
        lines.append("")

    embs = {}
    for wav_name in ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"):
        feat = fbank(load_wav(str(ROOT / "audio" / wav_name))).unsqueeze(0)
        with torch.inference_mode():
            embs[wav_name] = {
                "fp32": fp32(feat),
                "w4a8": w4a8(feat),
                "w4a4": w4a4(feat),
            }
    lines += ["## 声纹余弦（阈值 0.31）", "", "| 配对 | FP32 | W4A8 | W4A4 |", "|---|---:|---:|---:|"]
    pairs = [
        ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav"),
        ("speaker1_a_cn_16k.wav", "speaker2_a_cn_16k.wav"),
    ]
    for a, b in pairs:
        lines.append(
            f"| {a} vs {b} | {cosine(embs[a]['fp32'], embs[b]['fp32']):.4f} | "
            f"{cosine(embs[a]['w4a8'], embs[b]['w4a8']):.4f} | "
            f"{cosine(embs[a]['w4a4'], embs[b]['w4a4']):.4f} |"
        )
    lines.append("")
    lines.append(f"写入时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    text = "\n".join(lines) + "\n"
    out = ROOT / "results" / "QUANT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
