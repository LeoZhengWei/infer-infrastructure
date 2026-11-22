"""Compare PyTorch CPU, ONNX Runtime, and PyTorch MPS.

Timed region is the embedding network only. Fbank is not included.
MPS timing calls torch.mps.synchronize() so the GPU work is finished.
Correctness is max-abs and cosine against the CPU embedding.
"""

import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "third_party" / "campplus"))

from export_onnx import load_model  # noqa: E402
from fbank import fbank, load_wav  # noqa: E402


def threads() -> int:
    env = os.environ.get("BENCH_THREADS")
    if env:
        return int(env)
    try:
        n = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True).strip())
        if n > 0:
            return n
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return max(1, os.cpu_count() or 1)


def write_case(path: Path, feat: np.ndarray, ref: np.ndarray | None) -> None:
    feat = np.ascontiguousarray(feat, dtype=np.float32)
    if feat.ndim == 2:
        feat = feat[None, ...]
    n, t, f = feat.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(b"FB01")
        out.write(struct.pack("<III", n, t, f))
        out.write(feat.tobytes())
        if ref is not None:
            ref = np.ascontiguousarray(ref, dtype=np.float32)
            out.write(struct.pack("<I", ref.shape[-1]))
            out.write(ref.tobytes())


def read_vec(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = struct.unpack_from("<I", raw, 0)[0]
    return np.frombuffer(raw, dtype=np.float32, count=n, offset=4).copy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sample_ms(fn, warmup: int, repeat: int) -> list[float]:
    """Discard warmup runs, then return `repeat` steady-state timings in order."""
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


def parse_kv(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def main() -> None:
    import onnxruntime as ort

    nthread = threads()
    torch.set_num_threads(nthread)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthread)

    onnx_path = ROOT / "onnx" / "campplus.onnx"
    pack_path = ROOT / "onnx" / "campplus.pack"
    if not onnx_path.exists() or not pack_path.exists():
        raise SystemExit("run python/export_onnx.py first")

    if not torch.backends.mps.is_available():
        raise SystemExit("这台机器没有可用的 MPS")
    model = load_model()
    mps_model = load_model().to("mps")
    so = ort.SessionOptions()
    so.intra_op_num_threads = nthread
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    bin_path = ROOT / "cpp" / "build" / "campplus_bench"
    if not bin_path.exists():
        raise SystemExit(f"missing {bin_path}; build cpp first")

    warmup, repeat = 10, 10
    rows = []
    case_dir = ROOT / "testdata"
    case_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    cases = []
    for T in (300, 800):
        x = torch.randn(1, T, 80)
        cases.append((f"randn_T{T}", x))
    wav = load_wav(str(ROOT / "audio" / "speaker1_a_cn_16k.wav"))
    feat = fbank(wav)
    cases.append((f"speaker1_a_T{feat.shape[0]}", feat.unsqueeze(0)))

    for name, x in cases:
        x = x.contiguous()
        with torch.inference_mode():
            ref = model(x).numpy()

        def run_pt(x=x):
            with torch.inference_mode():
                model(x)

        def run_ort(x=x):
            sess.run(None, {"x": x.numpy()})

        x_mps = x.to("mps")

        def run_mps(x_mps=x_mps):
            with torch.inference_mode():
                mps_model(x_mps)
            torch.mps.synchronize()

        pt_samples = sample_ms(run_pt, warmup, repeat)
        ort_samples = sample_ms(run_ort, warmup, repeat)
        mps_samples = sample_ms(run_mps, warmup, repeat)
        got = sess.run(None, {"x": x.numpy()})[0]
        with torch.inference_mode():
            mps_emb = mps_model(x_mps)
        torch.mps.synchronize()
        mps_np = mps_emb.float().cpu().numpy()
        case = case_dir / f"{name}.bin"
        write_case(case, x.numpy(), ref)
        proc = subprocess.run(
            [
                str(bin_path),
                "time",
                "--onnx",
                str(onnx_path),
                "--pack",
                str(pack_path),
                "--input",
                str(case),
                "--threads",
                str(nthread),
                "--warmup",
                str(warmup),
                "--repeat",
                str(repeat),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"cpp bench failed on {name}")
        kv = parse_kv(proc.stdout)
        cpp_samples = [float(v) for v in kv["ort_cpp_samples"].split(",")]
        rows.append(
            {
                "name": name,
                "T": int(x.shape[1]),
                "pytorch": pt_samples,
                "onnx_py": ort_samples,
                "onnx_cpp": cpp_samples,
                "mps": mps_samples,
                "onnx_max_abs": float(np.max(np.abs(got - ref))),
                "ort_cpp_max_abs": float(kv["ort_max_abs"]),
                "mps_max_abs": float(np.max(np.abs(mps_np - ref))),
                "mps_cosine": cosine(mps_np, ref),
            }
        )
        print(
            f"{name}: pt {median(pt_samples):.2f} ms | onnx-py {median(ort_samples):.2f} | "
            f"onnx-cpp {median(cpp_samples):.2f} | mps {median(mps_samples):.2f} | "
            f"mps cos {rows[-1]['mps_cosine']:.6f} max {rows[-1]['mps_max_abs']:.3e}"
        )

    # Speaker scores. Threshold in the model card is cosine 0.31.
    embs = {}
    for wav_name in ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"):
        feat = fbank(load_wav(str(ROOT / "audio" / wav_name))).unsqueeze(0).contiguous()
        feat_mps = feat.to("mps")
        with torch.inference_mode():
            ref = model(feat).numpy()
            mps_emb = mps_model(feat_mps)
        torch.mps.synchronize()
        embs[wav_name] = {
            "pytorch": ref.reshape(-1),
            "onnx": sess.run(None, {"x": feat.numpy()})[0].reshape(-1),
            "mps": mps_emb.float().cpu().numpy().reshape(-1),
        }

    pairs = [
        ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "same"),
        ("speaker1_a_cn_16k.wav", "speaker2_a_cn_16k.wav", "different"),
    ]

    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        cpu = "unknown"

    lines = [
        "# CAM++ 速度与对拍",
        "",
        f"- 机器：{cpu}",
        f"- 线程：PyTorch CPU 与 ORT intra-op 为 {nthread}。MPS 走 Metal。",
        f"- 计时：先跑 {warmup} 轮丢掉启动（会话创建、MPS 着色器编译），再记 {repeat} 轮。表内是这 {repeat} 轮，最后一行是中位数。不含 fbank。MPS 每轮在 synchronize 之后停表。",
        "- 权重：`iic/speech_campplus_sv_zh-cn_16k-common`（`campplus_cn_common.bin`）",
        "- MPS：同一套权重，`model.to(\"mps\")`。",
        "",
    ]
    for r in rows:
        lines.append(f"## {r['name']}（T={r['T']}）")
        lines.append("")
        lines.append(f"MPS cosine {r['mps_cosine']:.6f}，max abs {r['mps_max_abs']:.3e}。ONNX max abs {r['onnx_max_abs']:.3e}。")
        lines.append("")
        lines.append("| 轮次 | PyTorch CPU ms | ONNX Python ms | ONNX C++ ms | MPS ms |")
        lines.append("|---:|---:|---:|---:|---:|")
        for i in range(repeat):
            lines.append(
                f"| {i + 1} | {r['pytorch'][i]:.2f} | {r['onnx_py'][i]:.2f} | "
                f"{r['onnx_cpp'][i]:.2f} | {r['mps'][i]:.2f} |"
            )
        lines.append(
            f"| 中位数 | {median(r['pytorch']):.2f} | {median(r['onnx_py']):.2f} | "
            f"{median(r['onnx_cpp']):.2f} | {median(r['mps']):.2f} |"
        )
        lines.append("")
    lines += ["", "## 声纹余弦（阈值 0.31）", "", "| 配对 | 标签 | PyTorch CPU | ONNX | MPS |", "|---|---|---:|---:|---:|"]
    for a, b, label in pairs:
        lines.append(
            f"| {a} vs {b} | {label} | {cosine(embs[a]['pytorch'], embs[b]['pytorch']):.4f} | "
            f"{cosine(embs[a]['onnx'], embs[b]['onnx']):.4f} | {cosine(embs[a]['mps'], embs[b]['mps']):.4f} |"
        )
    lines.append("")
    lines.append(f"写入时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    text = "\n".join(lines) + "\n"
    out = ROOT / "results" / "BENCH.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
