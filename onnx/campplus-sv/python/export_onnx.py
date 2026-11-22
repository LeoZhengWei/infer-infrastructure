"""Load CAM++ weights and export a dynamic-length ONNX graph by hand.

The only graph change versus the PyTorch module is CAMLayer.seg_pooling:
AveragePool is exported with count_include_pad=0. With zero padding that
matches PyTorch, and it stops older ONNX Runtime builds from dividing a
partial last window by the full kernel size.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "campplus"))

from speakerlab.models.campplus.DTDNN import CAMPPlus  # noqa: E402


def load_model() -> CAMPPlus:
    model = CAMPPlus(
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=False,
    )
    state = torch.load(ROOT / "weights" / "campplus_cn_common.bin", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def write_pack(model: CAMPPlus, path: Path) -> None:
    items = []
    for name, tensor in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        array = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
        items.append((name, array))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"CAMPACK1")
        f.write(struct.pack("<I", len(items)))
        for name, array in items:
            raw = name.encode("utf-8")
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
            f.write(struct.pack("<I", array.ndim))
            for dim in array.shape:
                f.write(struct.pack("<I", int(dim)))
            f.write(array.tobytes())


def export(model: CAMPPlus, path: Path) -> None:
    dummy = torch.randn(1, 200, 80)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        dynamo=False,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["embedding"],
        dynamic_axes={"x": {0: "N", 1: "T"}, "embedding": {0: "N"}},
    )


@torch.inference_mode()
def check(model: CAMPPlus, onnx_path: Path) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    for T in (100, 200, 373):
        x = torch.randn(1, T, 80)
        ref = model(x)
        got = sess.run(None, {"x": x.numpy()})[0]
        diff = np.max(np.abs(ref.numpy() - got))
        print(f"ort vs pytorch T={T} max_abs={diff:.3e} emb={tuple(ref.shape)}")
        if diff > 1e-4:
            raise SystemExit(f"ONNX mismatch at T={T}: {diff}")


def main() -> None:
    torch.manual_seed(0)
    model = load_model()
    onnx_path = ROOT / "onnx" / "campplus.onnx"
    pack_path = ROOT / "onnx" / "campplus.pack"
    print("export", onnx_path)
    export(model, onnx_path)
    print("pack", pack_path)
    write_pack(model, pack_path)
    check(model, onnx_path)
    print("export ok")


if __name__ == "__main__":
    main()
