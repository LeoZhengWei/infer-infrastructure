"""Post-training Q8 (int8) of the exported CAM++ ONNX graph.

Weights and activations are both QInt8. Conv stays in the graph as
QuantizeLinear / DequantizeLinear around the original fp32 op so ORT can
fold it into an int8 kernel when the runtime has one.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fbank import fbank, load_wav  # noqa: E402


class FbankReader:
    def __init__(self, batches: list[np.ndarray]):
        self.batches = batches
        self.index = 0

    def get_next(self):
        if self.index >= len(self.batches):
            return None
        batch = self.batches[self.index]
        self.index += 1
        return {"x": batch}

    def rewind(self):
        self.index = 0


def calibration_batches() -> list[np.ndarray]:
    torch.manual_seed(0)
    batches = []
    for frames in (100, 200, 300, 370, 800):
        batches.append(torch.randn(1, frames, 80).numpy().astype(np.float32))
    audio = ROOT / "audio"
    for name in ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"):
        feat = fbank(load_wav(str(audio / name))).unsqueeze(0).numpy().astype(np.float32)
        batches.append(feat)
    return batches


def main() -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    src = ROOT / "onnx" / "campplus.onnx"
    dst = ROOT / "onnx" / "campplus_q8.onnx"
    if not src.exists():
        raise SystemExit("run python/export_onnx.py first")
    reader = FbankReader(calibration_batches())
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    import onnx

    model = onnx.load(str(dst))
    ops: dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    q = ops.get("QuantizeLinear", 0)
    dq = ops.get("DequantizeLinear", 0)
    print(f"wrote {dst}  QuantizeLinear {q}  DequantizeLinear {dq}")
    print("ops", dict(sorted(ops.items(), key=lambda kv: -kv[1])[:12]))


if __name__ == "__main__":
    main()
