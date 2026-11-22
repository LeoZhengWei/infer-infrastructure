#include "mlp.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

struct T1 {
    int n = 0, c = 0, t = 0;
    std::vector<float> d;
    float& at(int b, int ci, int ti) { return d[(static_cast<size_t>(b) * c + ci) * t + ti]; }
    float at(int b, int ci, int ti) const { return d[(static_cast<size_t>(b) * c + ci) * t + ti]; }
};

struct T2 {
    int n = 0, c = 0, h = 0, w = 0;
    std::vector<float> d;
    size_t index(int b, int ci, int hi, int wi) const {
        return (((static_cast<size_t>(b) * c + ci) * h + hi) * w) + wi;
    }
    float& at(int b, int ci, int hi, int wi) { return d[index(b, ci, hi, wi)]; }
    float at(int b, int ci, int hi, int wi) const { return d[index(b, ci, hi, wi)]; }
};

const W& getw(const Pack& pack, const std::string& key) {
    auto it = pack.w.find(key);
    if (it == pack.w.end()) {
        throw std::runtime_error("missing weight " + key);
    }
    return it->second;
}

const W* maybe(const Pack& pack, const std::string& key) {
    auto it = pack.w.find(key);
    return it == pack.w.end() ? nullptr : &it->second;
}

int conv_out(int in, int kernel, int stride, int pad, int dilation) {
    return (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

void gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    // These GEMMs are small (often 32–128 rows). Splitting them across threads
    // was slower than one Accelerate call, so this stays single-threaded.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, K, B, N, 0.f, C, N);
}

void add_bias_rows(float* y, int rows, int cols, const float* bias) {
    for (int r = 0; r < rows; ++r) {
        float b = bias[r];
        float* row = y + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) row[c] += b;
    }
}

T1 conv1d(const T1& x, const W& weight, const W* bias, int stride, int dilation, int pad) {
    if (weight.shape.size() != 3) throw std::runtime_error("conv1d weight rank");
    const int oc = weight.shape[0];
    const int ic = weight.shape[1];
    const int k = weight.shape[2];
    if (ic != x.c) throw std::runtime_error("conv1d channel mismatch");
    const int tout = conv_out(x.t, k, stride, pad, dilation);
    if (tout <= 0) throw std::runtime_error("conv1d empty output");
    T1 y;
    y.n = x.n;
    y.c = oc;
    y.t = tout;
    y.d.resize(static_cast<size_t>(y.n) * oc * tout);

    if (k == 1 && stride == 1 && dilation == 1 && pad == 0) {
        for (int b = 0; b < x.n; ++b) {
            const float* src = x.d.data() + static_cast<size_t>(b) * ic * x.t;
            float* dst = y.d.data() + static_cast<size_t>(b) * oc * tout;
            gemm(oc, tout, ic, weight.data.data(), src, dst);
            if (bias) add_bias_rows(dst, oc, tout, bias->data.data());
        }
        return y;
    }

    std::vector<float> col(static_cast<size_t>(ic) * k * tout);
    for (int b = 0; b < x.n; ++b) {
        for (int ci = 0; ci < ic; ++ci) {
            for (int ki = 0; ki < k; ++ki) {
                float* col_row = col.data() + static_cast<size_t>(ci * k + ki) * tout;
                for (int to = 0; to < tout; ++to) {
                    int ti = to * stride + ki * dilation - pad;
                    col_row[to] = (ti >= 0 && ti < x.t) ? x.at(b, ci, ti) : 0.f;
                }
            }
        }
        float* dst = y.d.data() + static_cast<size_t>(b) * oc * tout;
        gemm(oc, tout, ic * k, weight.data.data(), col.data(), dst);
        if (bias) add_bias_rows(dst, oc, tout, bias->data.data());
    }
    return y;
}

T2 conv2d(const T2& x, const W& weight, int stride_h, int stride_w, int pad_h, int pad_w) {
    if (weight.shape.size() != 4) throw std::runtime_error("conv2d weight rank");
    const int oc = weight.shape[0];
    const int ic = weight.shape[1];
    const int kh = weight.shape[2];
    const int kw = weight.shape[3];
    if (ic != x.c) throw std::runtime_error("conv2d channel mismatch");
    const int oh = conv_out(x.h, kh, stride_h, pad_h, 1);
    const int ow = conv_out(x.w, kw, stride_w, pad_w, 1);
    if (oh <= 0 || ow <= 0) throw std::runtime_error("conv2d empty output");
    T2 y;
    y.n = x.n;
    y.c = oc;
    y.h = oh;
    y.w = ow;
    y.d.resize(static_cast<size_t>(y.n) * oc * oh * ow);
    const int spatial = oh * ow;
    const int kcol = ic * kh * kw;
    std::vector<float> col(static_cast<size_t>(kcol) * spatial);
    for (int b = 0; b < x.n; ++b) {
        int kidx = 0;
        for (int ci = 0; ci < ic; ++ci) {
            for (int iy = 0; iy < kh; ++iy) {
                for (int ix = 0; ix < kw; ++ix, ++kidx) {
                    float* col_row = col.data() + static_cast<size_t>(kidx) * spatial;
                    for (int yh = 0; yh < oh; ++yh) {
                        int ih = yh * stride_h + iy - pad_h;
                        for (int xw = 0; xw < ow; ++xw) {
                            int iw = xw * stride_w + ix - pad_w;
                            float v = 0.f;
                            if (ih >= 0 && ih < x.h && iw >= 0 && iw < x.w) v = x.at(b, ci, ih, iw);
                            col_row[yh * ow + xw] = v;
                        }
                    }
                }
            }
        }
        float* dst = y.d.data() + static_cast<size_t>(b) * oc * spatial;
        gemm(oc, spatial, kcol, weight.data.data(), col.data(), dst);
    }
    return y;
}

void bn_inplace_1(T1& x, const Pack& pack, const std::string& prefix) {
    const W& mean = getw(pack, prefix + ".running_mean");
    const W& var = getw(pack, prefix + ".running_var");
    const W* scale = maybe(pack, prefix + ".weight");
    const W* shift = maybe(pack, prefix + ".bias");
    constexpr float eps = 1e-5f;
    if (static_cast<int>(mean.data.size()) != x.c) throw std::runtime_error("bn1 channels " + prefix);
    for (int ci = 0; ci < x.c; ++ci) {
        float s = scale ? scale->data[ci] : 1.f;
        float b = shift ? shift->data[ci] : 0.f;
        float mul = s / std::sqrt(var.data[ci] + eps);
        float add = b - mean.data[ci] * mul;
        for (int n = 0; n < x.n; ++n) {
            float* row = &x.at(n, ci, 0);
            for (int t = 0; t < x.t; ++t) row[t] = row[t] * mul + add;
        }
    }
}

void bn_inplace_2(T2& x, const Pack& pack, const std::string& prefix) {
    const W& mean = getw(pack, prefix + ".running_mean");
    const W& var = getw(pack, prefix + ".running_var");
    const W* scale = maybe(pack, prefix + ".weight");
    const W* shift = maybe(pack, prefix + ".bias");
    constexpr float eps = 1e-5f;
    const int hw = x.h * x.w;
    for (int ci = 0; ci < x.c; ++ci) {
        float s = scale ? scale->data[ci] : 1.f;
        float b = shift ? shift->data[ci] : 0.f;
        float mul = s / std::sqrt(var.data[ci] + eps);
        float add = b - mean.data[ci] * mul;
        for (int n = 0; n < x.n; ++n) {
            float* plane = x.d.data() + (static_cast<size_t>(n) * x.c + ci) * hw;
            for (int i = 0; i < hw; ++i) plane[i] = plane[i] * mul + add;
        }
    }
}

void relu_inplace(std::vector<float>& d) {
    for (float& v : d) {
        if (v < 0.f) v = 0.f;
    }
}

T1 bn_relu_1(T1 x, const Pack& pack, const std::string& prefix) {
    bn_inplace_1(x, pack, prefix);
    relu_inplace(x.d);
    return x;
}

T1 cat_channel(const T1& a, const T1& b) {
    if (a.n != b.n || a.t != b.t) throw std::runtime_error("cat shape");
    T1 o;
    o.n = a.n;
    o.c = a.c + b.c;
    o.t = a.t;
    o.d.resize(static_cast<size_t>(o.n) * o.c * o.t);
    const size_t a_block = static_cast<size_t>(a.c) * a.t;
    const size_t b_block = static_cast<size_t>(b.c) * b.t;
    const size_t o_block = static_cast<size_t>(o.c) * o.t;
    for (int n = 0; n < a.n; ++n) {
        std::copy(a.d.begin() + n * a_block, a.d.begin() + n * a_block + a_block, o.d.begin() + n * o_block);
        std::copy(b.d.begin() + n * b_block, b.d.begin() + n * b_block + b_block,
                  o.d.begin() + n * o_block + a_block);
    }
    return o;
}

T1 seg_context(const T1& x) {
    constexpr int seg_len = 100;
    const int nseg = (x.t + seg_len - 1) / seg_len;
    std::vector<float> seg(static_cast<size_t>(x.n) * x.c * nseg);
    for (int b = 0; b < x.n; ++b) {
        for (int ci = 0; ci < x.c; ++ci) {
            for (int s = 0; s < nseg; ++s) {
                int begin = s * seg_len;
                int end = std::min(begin + seg_len, x.t);
                float sum = 0.f;
                for (int t = begin; t < end; ++t) sum += x.at(b, ci, t);
                seg[(static_cast<size_t>(b) * x.c + ci) * nseg + s] = sum / static_cast<float>(end - begin);
            }
        }
    }
    T1 ctx;
    ctx.n = x.n;
    ctx.c = x.c;
    ctx.t = x.t;
    ctx.d.resize(x.d.size());
    for (int b = 0; b < x.n; ++b) {
        for (int ci = 0; ci < x.c; ++ci) {
            float mean = 0.f;
            for (int t = 0; t < x.t; ++t) mean += x.at(b, ci, t);
            mean /= static_cast<float>(x.t);
            for (int t = 0; t < x.t; ++t) {
                int s = t / seg_len;
                float pooled = seg[(static_cast<size_t>(b) * x.c + ci) * nseg + s];
                ctx.at(b, ci, t) = mean + pooled;
            }
        }
    }
    return ctx;
}

void sigmoid_inplace(std::vector<float>& d) {
    for (float& v : d) {
        if (v >= 0.f) {
            float z = std::exp(-v);
            v = 1.f / (1.f + z);
        } else {
            float z = std::exp(v);
            v = z / (1.f + z);
        }
    }
}

T1 cam_layer(const T1& x, const Pack& pack, const std::string& prefix, int kernel, int dilation) {
    int pad = (kernel - 1) / 2 * dilation;
    T1 y = conv1d(x, getw(pack, prefix + ".linear_local.weight"), maybe(pack, prefix + ".linear_local.bias"), 1,
                  dilation, pad);
    T1 ctx = seg_context(x);
    ctx = conv1d(ctx, getw(pack, prefix + ".linear1.weight"), maybe(pack, prefix + ".linear1.bias"), 1, 1, 0);
    relu_inplace(ctx.d);
    T1 mask = conv1d(ctx, getw(pack, prefix + ".linear2.weight"), maybe(pack, prefix + ".linear2.bias"), 1, 1, 0);
    sigmoid_inplace(mask.d);
    if (mask.d.size() != y.d.size()) throw std::runtime_error("cam mask shape");
    for (size_t i = 0; i < y.d.size(); ++i) y.d[i] *= mask.d[i];
    return y;
}

T1 cam_dense(const T1& x, const Pack& pack, const std::string& prefix, int kernel, int dilation) {
    T1 h = bn_relu_1(x, pack, prefix + ".nonlinear1.batchnorm");
    h = conv1d(h, getw(pack, prefix + ".linear1.weight"), maybe(pack, prefix + ".linear1.bias"), 1, 1, 0);
    h = bn_relu_1(std::move(h), pack, prefix + ".nonlinear2.batchnorm");
    return cam_layer(h, pack, prefix + ".cam_layer", kernel, dilation);
}

T1 dense_block(T1 x, const Pack& pack, const std::string& name, int layers, int kernel, int dilation) {
    for (int i = 1; i <= layers; ++i) {
        T1 y = cam_dense(x, pack, name + ".tdnnd" + std::to_string(i), kernel, dilation);
        x = cat_channel(x, y);
    }
    return x;
}

T1 transit(T1 x, const Pack& pack, const std::string& prefix) {
    x = bn_relu_1(std::move(x), pack, prefix + ".nonlinear.batchnorm");
    return conv1d(x, getw(pack, prefix + ".linear.weight"), maybe(pack, prefix + ".linear.bias"), 1, 1, 0);
}

T2 bn_relu_2(T2 x, const Pack& pack, const std::string& prefix) {
    bn_inplace_2(x, pack, prefix);
    relu_inplace(x.d);
    return x;
}

T2 add2(T2 a, const T2& b) {
    if (a.n != b.n || a.c != b.c || a.h != b.h || a.w != b.w) {
        throw std::runtime_error("residual shape mismatch");
    }
    for (size_t i = 0; i < a.d.size(); ++i) a.d[i] += b.d[i];
    return a;
}

T2 res_block(const T2& x, const Pack& pack, const std::string& prefix, int stride) {
    T2 y = conv2d(x, getw(pack, prefix + ".conv1.weight"), stride, 1, 1, 1);
    y = bn_relu_2(std::move(y), pack, prefix + ".bn1");
    y = conv2d(y, getw(pack, prefix + ".conv2.weight"), 1, 1, 1, 1);
    bn_inplace_2(y, pack, prefix + ".bn2");
    T2 skip = x;
    if (maybe(pack, prefix + ".shortcut.0.weight")) {
        skip = conv2d(x, getw(pack, prefix + ".shortcut.0.weight"), stride, 1, 0, 0);
        bn_inplace_2(skip, pack, prefix + ".shortcut.1");
    }
    y = add2(std::move(y), skip);
    relu_inplace(y.d);
    return y;
}

T1 fcm(const T2& x, const Pack& pack) {
    T2 h = conv2d(x, getw(pack, "head.conv1.weight"), 1, 1, 1, 1);
    h = bn_relu_2(std::move(h), pack, "head.bn1");
    h = res_block(h, pack, "head.layer1.0", 2);
    h = res_block(h, pack, "head.layer1.1", 1);
    h = res_block(h, pack, "head.layer2.0", 2);
    h = res_block(h, pack, "head.layer2.1", 1);
    h = conv2d(h, getw(pack, "head.conv2.weight"), 2, 1, 1, 1);
    h = bn_relu_2(std::move(h), pack, "head.bn2");
    T1 y;
    y.n = h.n;
    y.c = h.c * h.h;
    y.t = h.w;
    y.d = std::move(h.d);
    return y;
}

void stats_and_dense(const T1& x, const Pack& pack, float* out) {
    if (x.t < 2) throw std::runtime_error("stats pool needs T>=2 for unbiased std");
    const int c2 = x.c * 2;
    std::vector<float> stats(static_cast<size_t>(x.n) * c2);
    const float inv_n = 1.f / static_cast<float>(x.t);
    const float inv_nm1 = 1.f / static_cast<float>(x.t - 1);
    for (int b = 0; b < x.n; ++b) {
        for (int ci = 0; ci < x.c; ++ci) {
            float mean = 0.f;
            for (int t = 0; t < x.t; ++t) mean += x.at(b, ci, t);
            mean *= inv_n;
            float var = 0.f;
            for (int t = 0; t < x.t; ++t) {
                float d = x.at(b, ci, t) - mean;
                var += d * d;
            }
            var *= inv_nm1;
            stats[static_cast<size_t>(b) * c2 + ci] = mean;
            stats[static_cast<size_t>(b) * c2 + x.c + ci] = std::sqrt(var);
        }
    }
    const W& weight = getw(pack, "xvector.dense.linear.weight");
    const int emb = weight.shape[0];
    const int in_c = weight.shape[1];
    if (in_c != c2) throw std::runtime_error("dense in channels");
    std::vector<float> proj(static_cast<size_t>(x.n) * emb);
    for (int b = 0; b < x.n; ++b) {
        gemm(emb, 1, in_c, weight.data.data(), stats.data() + static_cast<size_t>(b) * c2,
             proj.data() + static_cast<size_t>(b) * emb);
    }
    T1 h;
    h.n = x.n;
    h.c = emb;
    h.t = 1;
    h.d = std::move(proj);
    bn_inplace_1(h, pack, "xvector.dense.nonlinear.batchnorm");
    std::copy(h.d.begin(), h.d.end(), out);
}

uint32_t read_u32(std::istream& in) {
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), 4);
    if (!in) throw std::runtime_error("truncated pack");
    return v;
}

}  // namespace

Pack load_pack(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != "CAMPACK1") throw std::runtime_error("bad pack magic");
    uint32_t n = read_u32(in);
    Pack pack;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t nlen = read_u32(in);
        std::string name(nlen, '\0');
        in.read(name.data(), nlen);
        uint32_t ndim = read_u32(in);
        W w;
        w.shape.resize(ndim);
        size_t count = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            w.shape[d] = static_cast<int>(read_u32(in));
            count *= static_cast<size_t>(w.shape[d]);
        }
        w.data.resize(count);
        in.read(reinterpret_cast<char*>(w.data.data()), static_cast<std::streamsize>(count * sizeof(float)));
        if (!in) throw std::runtime_error("truncated tensor " + name);
        pack.w.emplace(std::move(name), std::move(w));
    }
    return pack;
}

int embedding_dim(const Pack& pack) {
    return getw(pack, "xvector.dense.linear.weight").shape[0];
}

void mlp_forward(const Pack& pack, const float* in, int batch, int frames, int feat, float* out, int threads) {
    (void)threads;
    if (feat != 80) throw std::runtime_error("CAM++ expects 80-dim fbank");
    T2 x;
    x.n = batch;
    x.c = 1;
    x.h = feat;
    x.w = frames;
    x.d.resize(static_cast<size_t>(batch) * feat * frames);
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < frames; ++t) {
            for (int f = 0; f < feat; ++f) {
                x.at(b, 0, f, t) = in[(static_cast<size_t>(b) * frames + t) * feat + f];
            }
        }
    }
    T1 h = fcm(x, pack);
    h = conv1d(h, getw(pack, "xvector.tdnn.linear.weight"), maybe(pack, "xvector.tdnn.linear.bias"), 2, 1, 2);
    h = bn_relu_1(std::move(h), pack, "xvector.tdnn.nonlinear.batchnorm");
    const int layers[3] = {12, 24, 16};
    const int kernels[3] = {3, 3, 3};
    const int dil[3] = {1, 2, 2};
    for (int i = 0; i < 3; ++i) {
        h = dense_block(std::move(h), pack, "xvector.block" + std::to_string(i + 1), layers[i], kernels[i], dil[i]);
        h = transit(std::move(h), pack, "xvector.transit" + std::to_string(i + 1));
    }
    h = bn_relu_1(std::move(h), pack, "xvector.out_nonlinear.batchnorm");
    stats_and_dense(h, pack, out);
}
