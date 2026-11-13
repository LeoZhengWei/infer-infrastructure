"""第一步的主角：把一张 ONNX 图讲成人话。

跑法：
    python onnx_step1_inspect.py tiny_asr_like.onnx
    python onnx_step1_inspect.py tiny_asr_like.onnx --trace features
    python onnx_step1_inspect.py sensevoice.onnx --trace features_lens --shapes

只用 onnx 一个包，不依赖 Netron。造小图那个脚本才需要 numpy。
"""

import argparse
import collections
import os

import onnx
from onnx import TensorProto, shape_inference

# 这些算子最容易在 RKNN 上被踢回 CPU，或者逼着模型带动态形状。
# 第四步要拆的就是它们，第一步先学会认出来。
TROUBLE_OPS = {
    "Range", "Where", "Gather", "ScatterND", "Shape", "Unsqueeze",
    "NonZero", "If", "Loop", "Compress", "Tile", "Expand",
    "ConstantOfShape", "Slice", "Resize",
}


def dtype_name(value):
    try:
        return TensorProto.DataType.Name(value)
    except Exception:
        return str(value)


def dim_text(dim):
    if dim.HasField("dim_param"):
        return "{} (会变)".format(dim.dim_param)
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    return "?"


def shape_text(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return "[" + ", ".join(dim_text(d) for d in dims) + "]"


def describe_tensor(value_info):
    return "{:<20} {:<8} {}".format(
        value_info.name,
        dtype_name(value_info.type.tensor_type.elem_type),
        shape_text(value_info),
    )


def print_inputs_outputs(graph):
    # 老一点的导出器会把权重也列在 graph.input 里，这里排除掉，只留真正要喂的输入。
    init_names = {i.name for i in graph.initializer}
    real_inputs = [v for v in graph.input if v.name not in init_names]

    print("\n输入 {} 个：".format(len(real_inputs)))
    for v in real_inputs:
        print("  " + describe_tensor(v))
        if "会变" in shape_text(v):
            print("    ^ 这一维会变，第四步要把它钉成固定值")

    print("\n输出 {} 个：".format(len(graph.output)))
    for v in graph.output:
        print("  " + describe_tensor(v))


def print_nodes(graph):
    counts = collections.Counter(n.op_type for n in graph.node)
    print("\n节点共 {} 个，按算子排：".format(len(graph.node)))
    for op, cnt in counts.most_common():
        flag = "   <-- 以后要留意" if op in TROUBLE_OPS else ""
        print("  {:<20} x {}{}".format(op, cnt, flag))

    bad = {op: counts[op] for op in TROUBLE_OPS if counts[op]}
    if bad:
        print("\n  这批算子容易让 NPU 头疼：")
        print("  " + ", ".join("{} x{}".format(k, v) for k, v in sorted(bad.items())))
    else:
        print("\n  没看到常见的动态掩码算子，图挺干净。")


def print_weights(graph):
    # 不把权重解成 numpy，只按「元素个数 × 每个元素几字节」估个大概。
    bytes_per = {
        TensorProto.FLOAT: 4, TensorProto.INT32: 4, TensorProto.UINT32: 4,
        TensorProto.FLOAT16: 2, TensorProto.BFLOAT16: 2,
        TensorProto.INT16: 2, TensorProto.UINT16: 2,
        TensorProto.INT64: 8, TensorProto.UINT64: 8, TensorProto.DOUBLE: 8,
        TensorProto.INT8: 1, TensorProto.UINT8: 1, TensorProto.BOOL: 1,
    }
    total = 0
    for i in graph.initializer:
        count = 1
        for d in i.dims:
            count *= int(d)
        total += count * bytes_per.get(i.data_type, 4)
    print("\n权重 {} 个，粗略合计 {:.1f} KB".format(len(graph.initializer), total / 1024))


def trace_downstream(graph, start, max_depth=8):
    """从某个名字出发，顺着图往下找谁用了它，一层层打印出来。

    在 SenseVoice 上跑 `--trace features_lens`，就能亲眼看到那条
    Range / Where / Gather 掩码链，也就是文档里 A1 要拆掉的东西。
    """
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node)

    print("\n从 [{}] 往下追：".format(start))
    queue = collections.deque([(start, 0)])
    seen = set()
    found = False
    while queue:
        name, depth = queue.popleft()
        if name in seen or depth > max_depth:
            continue
        seen.add(name)
        for node in consumers.get(name, []):
            found = True
            outs = ", ".join(o for o in node.output if o) or "(无输出)"
            print("  " * (depth + 1) + "-> {:<14} {:<22} => {}".format(
                node.op_type, node.name or "(没起名)", outs
            ))
            for o in node.output:
                if o:
                    queue.append((o, depth + 1))
    if not found:
        print("  （没有节点用它，可能是没接上的悬空输入）")


def print_shapes(model, limit=12):
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception as exc:
        print("\n形状推不出来：{}".format(exc))
        return
    infos = list(inferred.graph.value_info)
    print("\n中间结果 {} 个，前 {} 个的形状：".format(len(infos), min(limit, len(infos))))
    for v in infos[:limit]:
        print("  " + describe_tensor(v))


def main():
    ap = argparse.ArgumentParser(description="把一张 ONNX 图讲成人话")
    ap.add_argument("model", help="模型文件路径，例如 tiny_asr_like.onnx")
    ap.add_argument("--trace", metavar="NAME", help="顺着某个输入/张量往下追，例如 features_lens")
    ap.add_argument("--shapes", action="store_true", help="顺便推断中间结果的形状（大模型会慢）")
    ap.add_argument("--nodes", action="store_true", help="把每个节点的输入输出都列出来")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit("找不到文件：{}".format(args.model))

    size = os.path.getsize(args.model) / (1024 * 1024)
    print("文件：{}  ({:.2f} MB)".format(args.model, size))
    if size > 500:
        print("  文件不小，onnx.load 会吃几百 MB 内存，正常现象。")

    # 大文件把路径交给 checker，别把整个 proto 再塞进去。
    try:
        onnx.checker.check_model(args.model)
        checker_msg = "通过"
    except Exception as exc:
        checker_msg = "没过：{}".format(exc)

    model = onnx.load(args.model)
    graph = model.graph

    print("IR 版本：{}   生成者：{}".format(model.ir_version, model.producer_name or "(没写)"))
    opsets = ", ".join("{} {}".format(o.domain or "ai.onnx", o.version) for o in model.opset_import)
    print("算子集：{}".format(opsets))
    print("文件体检：{}".format(checker_msg))

    print_inputs_outputs(graph)
    print_nodes(graph)
    print_weights(graph)

    if args.nodes:
        print("\n逐个节点：")
        for idx, n in enumerate(graph.node):
            print("  {:>3} {:<14} {:<22} {} -> {}".format(
                idx, n.op_type, n.name or "(没起名)",
                ",".join(x for x in n.input if x),
                ",".join(x for x in n.output if x),
            ))

    if args.trace:
        trace_downstream(graph, args.trace)

    if args.shapes:
        print_shapes(model)

    print("\n看完问自己三句：数据从哪进？哪一维度会变？出字之前都经过哪些节点？")


if __name__ == "__main__":
    main()