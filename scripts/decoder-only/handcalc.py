# -*- coding: utf-8 -*-
"""
讲义风格手算演示：V=5, d_model=3, 1 层 1 头, FFN 3->4->3
所有权重为手工设定的整数/一位小数，便于纸笔验算。
演示目标：输入「我爱」推出「你」。
"""
import math

V, D, DFF = 5, 3, 4
ITOS = ["我", "爱", "你", "他", "<eos>"]

# ---------- 参数 ----------
E = [[1, 0, 0],
     [0, 1, 0],
     [1, 1, 0],
     [0, 0, 1],
     [-1, 0, 0]]

P = [[0.0, 0.0, 0.0],
     [0.1, 0.0, 0.0],
     [0.2, 0.0, 0.0]]

WQ = [[1, 0, 1], [0, 1, 0], [1, 0, -1]]
WK = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
WV = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
WO = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

W1 = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0]]
b1 = [0.1, 0.1, 0.1, 0.1]
W2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]

WU = E   # weight tying: logits = E · h

def mv(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e], e

def r(x, k=4):
    return round(x, k)

def fmt(v, k=4):
    return "[" + ", ".join(f"{x:+.{k}f}" for x in v) + "]"

log = []
def w(s=""):
    log.append(s)

def attn_block(ids):
    """返回每个位置的全部中间量"""
    out = []
    for pos, t in enumerate(ids):
        x = [E[t][j] + P[pos][j] for j in range(D)]
        out.append({"pos": pos, "tok": ITOS[t], "x": x,
                    "q": mv(WQ, x), "k": mv(WK, x), "v": mv(WV, x)})
    for i, rec in enumerate(out):
        sc = [dot(rec["q"], out[j]["k"]) / math.sqrt(D) for j in range(i + 1)]
        al, ex = softmax(sc)
        a = [sum(al[j] * out[j]["v"][d] for j in range(i + 1)) for d in range(D)]
        ao = mv(WO, a)
        x1 = [rec["x"][d] + ao[d] for d in range(D)]
        pre = [sum(x1[d] * W1[d][c] for d in range(D)) + b1[c] for c in range(DFF)]
        rel = [max(0.0, v) for v in pre]
        ffn = [sum(rel[c] * W2[c][d] for c in range(DFF)) for d in range(D)]
        x2 = [x1[d] + ffn[d] for d in range(D)]
        rec.update({"score_raw": [dot(rec["q"], out[j]["k"]) for j in range(i + 1)],
                    "score": sc, "alpha": al, "exp": ex, "a": a, "ao": ao,
                    "x1": x1, "pre": pre, "rel": rel, "ffn": ffn, "h": x2,
                    "logits": mv(WU, x2)})
        rec["probs"], _ = softmax(rec["logits"])
    return out

w("================ 输入「我爱」 ================")
rec = attn_block([0, 1])
for p in rec:
    w(f"\n--- pos{p['pos']}  {p['tok']} ---")
    w("x  = " + fmt(p["x"]) + "   (E+P)")
    w("q  = " + fmt(p["q"]))
    w("k  = " + fmt(p["k"]))
    w("v  = " + fmt(p["v"]))
    w("q·k     = " + fmt(p["score_raw"]))
    w("q·k/√3  = " + fmt(p["score"]))
    w("alpha   = " + fmt(p["alpha"]) + "   exp=" + fmt(p["exp"]))
    w("a=α·V   = " + fmt(p["a"]))
    w("a·WO    = " + fmt(p["ao"]))
    w("x1=x+a  = " + fmt(p["x1"]))
    w("pre     = " + fmt(p["pre"]))
    w("relu    = " + fmt(p["rel"]))
    w("ffn     = " + fmt(p["ffn"]))
    w("h=x1+ffn= " + fmt(p["h"]))
    w("logits  = " + fmt(p["logits"]))
    w("softmax = " + fmt(p["probs"]))

w("\n汇总（最后位置）")
h = rec[-1]["h"]
lg = rec[-1]["logits"]
pr = rec[-1]["probs"]
for i, t in enumerate(ITOS):
    w(f"  {t:<5}  h·E[{t}] = {lg[i]:+.4f}   p = {pr[i]:.4f}")
w("  argmax -> " + ITOS[max(range(V), key=lambda i: lg[i])])

w("\n================ 第 3 步：把「你」接上去 ================")
rec3 = attn_block([0, 1, 2])
p2 = rec3[2]
w("x2  = " + fmt(p2["x"]))
w("q2  = " + fmt(p2["q"]) + "  k2 = " + fmt(p2["k"]) + "  v2 = " + fmt(p2["v"]))
w("scores = " + fmt(p2["score"]))
w("alpha  = " + fmt(p2["alpha"]))
w("h2     = " + fmt(p2["h"]))
w("logits = " + fmt(p2["logits"]))
w("softmax= " + fmt(p2["probs"]) + "  -> " + ITOS[max(range(V), key=lambda i: p2["logits"][i])])

w("\n================ 手算复核（关键点） ================")
w("q1·k0 = 0.1*1 + 1*0 + 0.1*1 = 0.2")
w("q1·k1 = 0.1*1.1 + 1*1 + 0.1*0.1 = 1.12")
w("0.2/1.7320508 = %.5f ; 1.12/1.7320508 = %.5f" % (0.2/math.sqrt(3), 1.12/math.sqrt(3)))
w("α = %.5f / %.5f" % (rec[1]["alpha"][0], rec[1]["alpha"][1]))
w("α·V dim0 = 0.37025*0 + 0.62975*1 = %.5f" % (rec[1]["alpha"][1]))
w("α·V dim1 = 0.37025*1 + 0.62975*0.1 = %.5f" % (rec[1]["alpha"][0] + rec[1]["alpha"][1]*0.1))
w("α·V dim2 = 0.37025*1 + 0.62975*1.1 = %.5f" % (rec[1]["alpha"][0] + rec[1]["alpha"][1]*1.1))
w("logit(你) = h0 + h1 = %.4f + %.4f = %.4f" % (h[0], h[1], h[0]+h[1]))

txt = "\n".join(log)
open("handcalc.log", "w", encoding="utf-8").write(txt)
print(txt)
