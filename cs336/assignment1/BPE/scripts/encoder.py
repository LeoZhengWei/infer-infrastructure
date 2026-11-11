from collections import Counter

# ---------- 训练 -------------
def train(corpus, num_merges):
    # 1. 初始化：拆成字符 + 词尾标记  : 频次
    vocab = {tuple(w) + ("</w>",): f for w, f in corpus.items()}
    merges = []

    # 合并规则执行次数
    for _ in range(num_merges):
        # 2. 统计所有相邻对(乘上词频）
        pairs = Counter()
        for word, f in vocab.items():
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += f

        if not pairs:
            break
        
        # 3 取频率最高的一对
        best = max(pairs, key=lambda p : (pairs[p], p))
        merges.append(best)

        # 4. 在语料里把这一对全部粘起来
        new_vocab = {}
        # 外层处理每个单词
        for word, f in vocab.items():
            out, i = [], 0
            # 内层循环合并处理每个单词的相邻对
            while i < len(word):
                if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                    # 合并相邻对
                    out.append(word[i] + best[1]);
                    i += 2
                else:
                    # 不合并，直接添加当前字符
                    out.append(word[i])
                    i += 1
            # 将当前处理词汇的字符组合加入词表，出现过的组合要累加频次，没出现过的要0 + f
            new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + f
            # key = tuple(out)
            # if key in new_vocab:
            #     new_vocab[key] = new_vocab[key] + f
            # else:
            #     new_vocab[key] = f
        # 更新词表 进入下一轮
        vocab = new_vocab
    return merges

# ---------- 编码 -------------
def encode(word, merges):
    toks = list(word) + ['</w>']
    for (a, b) in merges:
        out, i = [], 0
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
    return toks

corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
m = train(corpus, 10)
print(m)
print(encode('lowest', m))
print(encode('newer', m))
print(encode('slow', m))