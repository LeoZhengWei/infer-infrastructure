"""Compare the fp32 ONNX graph with the Q8 graph.

Ten warmup runs are dropped. The table is the next ten rounds.
"""

import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fbank import fbank, load_wav  # noqa: E402


def session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def write_case(path: Path, feat: np.ndarray) -> None:
    feat = np.ascontiguousarray(feat, dtype=np.float32)
    if feat.ndim == 2:
        feat = feat[None]
    n, t, f = feat.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(b"FB01")
        out.write(struct.pack("<III", n, t, f))
        out.write(feat.tobytes())


def read_vec(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = struct.unpack_from("<I", raw, 0)[0]
    return np.frombuffer(raw, dtype=np.float32, count=n, offset=4).copy()


def cpp_q8(
    bin_path: Path,
    feat: np.ndarray,
    case: Path,
    dump_fp: Path | None,
    dump_q: Path | None,
    warmup: int = 10,
    repeat: int = 10,
):
    write_case(case, feat)
    cmd = [
        str(bin_path),
        "q8",
        "--onnx",
        str(ROOT / "onnx" / "campplus.onnx"),
        "--q8",
        str(ROOT / "onnx" / "campplus_q8.onnx"),
        "--input",
        str(case),
        "--threads",
        "4",
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
    ]
    if dump_fp is not None:
        cmd += ["--dump-fp32", str(dump_fp), "--dump-q8", str(dump_q)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    kv = {}
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            kv[parts[0]] = parts[1]
    return kv


def main() -> None:
    fp32 = session(ROOT / "onnx" / "campplus.onnx")
    q8 = session(ROOT / "onnx" / "campplus_q8.onnx")
    warmup, repeat = 10, 10
    torch.manual_seed(0)
    cases = [
        ("randn_T300", torch.randn(1, 300, 80).numpy().astype(np.float32)),
        ("randn_T800", torch.randn(1, 800, 80).numpy().astype(np.float32)),
    ]
    feat = fbank(load_wav(str(ROOT / "audio" / "speaker1_a_cn_16k.wav")))
    cases.append((f"speaker1_a_T{feat.shape[0]}", feat.unsqueeze(0).numpy().astype(np.float32)))

    lines = [
        "# Python ORT",
        "",
        "- Q8：导出 FP32 ONNX 之后做静态量化，权重和激活都是对称 QInt8，per-channel 权重。",
        "- 校准数据：几段随机 fbank，加上仓库里的三条示例语音。",
        f"- 先跑 {warmup} 轮丢掉启动，再记 {repeat} 轮。ORT intra-op 线程数为 4。",
        "",
    ]
    for name, x in cases:
        ref = fp32.run(None, {"x": x})[0]
        got = q8.run(None, {"x": x})[0]
        fp_s = sample_ms(lambda x=x: fp32.run(None, {"x": x}), warmup, repeat)
        q_s = sample_ms(lambda x=x: q8.run(None, {"x": x}), warmup, repeat)
        diff = float(np.max(np.abs(ref - got)))
        cos = cosine(ref, got)
        print(f"{name}: fp32 {median(fp_s):.2f}  q8 {median(q_s):.2f}  max {diff:.3e}  cos {cos:.4f}")
        lines += [
            f"## {name}（T={x.shape[1]}）",
            "",
            f"量化前后 embedding 余弦 {cos:.4f}，差值 1−余弦 = {1 - cos:.4f}，max abs {diff:.3e}。",
            "",
            "| 轮次 | ONNX FP32 ms | ONNX Q8 ms |",
            "|---:|---:|---:|",
        ]
        for i in range(repeat):
            lines.append(f"| {i + 1} | {fp_s[i]:.2f} | {q_s[i]:.2f} |")
        lines.append(f"| 中位数 | {median(fp_s):.2f} | {median(q_s):.2f} |")
        lines.append("")

    embs = {"fp32": {}, "q8": {}}
    for wav_name in ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"):
        feat = fbank(load_wav(str(ROOT / "audio" / wav_name))).unsqueeze(0).numpy().astype(np.float32)
        embs["fp32"][wav_name] = fp32.run(None, {"x": feat})[0]
        embs["q8"][wav_name] = q8.run(None, {"x": feat})[0]
    lines += [
        "## Python 声纹余弦（阈值 0.31）",
        "",
        "| 配对 | 量化前 | 量化后 | 差值（后−前） |",
        "|---|---:|---:|---:|",
    ]
    for a, b in (
        ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav"),
        ("speaker1_a_cn_16k.wav", "speaker2_a_cn_16k.wav"),
    ):
        before = cosine(embs["fp32"][a], embs["fp32"][b])
        after = cosine(embs["q8"][a], embs["q8"][b])
        lines.append(f"| {a} vs {b} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")

    bin_path = ROOT / "cpp" / "build" / "campplus_bench"
    if not bin_path.exists():
        raise SystemExit(f"missing {bin_path}")
    case_dir = ROOT / "testdata"
    lines += ["", "# C++ ORT", ""]
    for name, x in cases:
        kv = cpp_q8(bin_path, x, case_dir / f"q8_{name}.bin", None, None)
        fp_s = [float(v) for v in kv["fp32_samples"].split(",")]
        q_s = [float(v) for v in kv["q8_samples"].split(",")]
        cos = float(kv["cosine"])
        print(f"cpp {name}: fp32 {median(fp_s):.2f}  q8 {median(q_s):.2f}  cos {cos:.4f}  delta {1 - cos:.4f}")
        lines += [
            f"## {name}（T={x.shape[1]}）",
            "",
            f"量化前后 embedding 余弦 {cos:.4f}，差值 1−余弦 = {1 - cos:.4f}，max abs {float(kv['max_abs']):.3e}。",
            "",
            "| 轮次 | C++ FP32 ms | C++ Q8 ms |",
            "|---:|---:|---:|",
        ]
        for i in range(repeat):
            lines.append(f"| {i + 1} | {fp_s[i]:.2f} | {q_s[i]:.2f} |")
        lines.append(f"| 中位数 | {median(fp_s):.2f} | {median(q_s):.2f} |")
        lines.append("")

    cpp_embs = {"fp32": {}, "q8": {}}
    for wav_name in ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"):
        feat = fbank(load_wav(str(ROOT / "audio" / wav_name))).unsqueeze(0).numpy().astype(np.float32)
        dump_fp = case_dir / f"{wav_name}.fp32"
        dump_q = case_dir / f"{wav_name}.q8"
        cpp_q8(bin_path, feat, case_dir / f"q8_{wav_name}.bin", dump_fp, dump_q, warmup=1, repeat=1)
        cpp_embs["fp32"][wav_name] = read_vec(dump_fp)
        cpp_embs["q8"][wav_name] = read_vec(dump_q)
        same = cosine(cpp_embs["fp32"][wav_name], cpp_embs["q8"][wav_name])
        lines.append(f"{wav_name} 量化前后余弦 {same:.4f}，差值 {1 - same:.4f}。")
        lines.append("")
    lines += [
        "## C++ 声纹余弦（阈值 0.31）",
        "",
        "| 配对 | 量化前 | 量化后 | 差值（后−前） |",
        "|---|---:|---:|---:|",
    ]
    for a, b in (
        ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav"),
        ("speaker1_a_cn_16k.wav", "speaker2_a_cn_16k.wav"),
    ):
        before = cosine(cpp_embs["fp32"][a], cpp_embs["fp32"][b])
        after = cosine(cpp_embs["q8"][a], cpp_embs["q8"][b])
        lines.append(f"| {a} vs {b} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    lines.append("")
    lines.append(f"写入时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    text = "\n".join(lines) + "\n"
    out = ROOT / "results" / "Q8.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
