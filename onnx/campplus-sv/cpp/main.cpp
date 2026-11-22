#include "mlp.h"

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Input {
    int n = 0, t = 0, f = 0;
    std::vector<float> data;
    std::vector<float> ref;
};

uint32_t ru32(std::istream& in) {
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), 4);
    if (!in) throw std::runtime_error("truncated input");
    return v;
}

Input load_input(const std::string& path, bool with_ref) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "FB01") throw std::runtime_error("bad input magic");
    Input x;
    x.n = static_cast<int>(ru32(in));
    x.t = static_cast<int>(ru32(in));
    x.f = static_cast<int>(ru32(in));
    x.data.resize(static_cast<size_t>(x.n) * x.t * x.f);
    in.read(reinterpret_cast<char*>(x.data.data()), static_cast<std::streamsize>(x.data.size() * 4));
    if (with_ref) {
        uint32_t dim = ru32(in);
        x.ref.resize(static_cast<size_t>(x.n) * dim);
        in.read(reinterpret_cast<char*>(x.ref.data()), static_cast<std::streamsize>(x.ref.size() * 4));
    }
    if (!in) throw std::runtime_error("truncated input body");
    return x;
}

void write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream out(path, std::ios::binary);
    uint32_t n = static_cast<uint32_t>(v.size());
    out.write(reinterpret_cast<const char*>(&n), 4);
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

float cosine(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

struct OrtRunner {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "campplus"};
    Ort::SessionOptions opt;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem;

    OrtRunner(const std::string& model, int threads)
        : mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        opt.SetIntraOpNumThreads(threads);
        opt.SetInterOpNumThreads(1);
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, model.c_str(), opt);
    }

    std::vector<float> run(const Input& in) {
        int64_t shape[3] = {in.n, in.t, in.f};
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(in.data.data()), in.data.size(), shape, 3);
        const char* in_names[] = {"x"};
        const char* out_names[] = {"embedding"};
        auto out = session->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        float* p = out[0].GetTensorMutableData<float>();
        auto info = out[0].GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        return std::vector<float>(p, p + n);
    }
};

std::string arg(int argc, char** argv, const std::string& key, const std::string& fallback = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) return argv[i + 1];
    }
    if (fallback.empty()) throw std::runtime_error("missing " + key);
    return fallback;
}

int argi(int argc, char** argv, const std::string& key, int fallback) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) return std::stoi(argv[i + 1]);
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: campplus_bench time|embed --onnx --pack --input [--threads N]\n";
            return 2;
        }
        std::string mode = argv[1];
        std::string onnx = arg(argc, argv, "--onnx");
        std::string input = arg(argc, argv, "--input");
        int threads = argi(argc, argv, "--threads", 4);
        setenv("VECLIB_MAXIMUM_THREADS", "1", 1);

        OrtRunner ort(onnx, threads);

        if (mode == "embed") {
            std::string pack_path = arg(argc, argv, "--pack");
            Pack pack = load_pack(pack_path);
            int emb = embedding_dim(pack);
            Input in = load_input(input, false);
            auto y_ort = ort.run(in);
            std::vector<float> y_mlp(static_cast<size_t>(in.n) * emb);
            mlp_forward(pack, in.data.data(), in.n, in.t, in.f, y_mlp.data(), threads);
            write_f32(arg(argc, argv, "--out-ort"), y_ort);
            write_f32(arg(argc, argv, "--out-mlp"), y_mlp);
            return 0;
        }

        if (mode == "q8") {
            std::string q8_path = arg(argc, argv, "--q8");
            Input in = load_input(input, false);
            OrtRunner q8(q8_path, threads);
            int warmup = argi(argc, argv, "--warmup", 10);
            int repeat = argi(argc, argv, "--repeat", 10);
            for (int i = 0; i < warmup; ++i) {
                ort.run(in);
                q8.run(in);
            }
            std::vector<double> fp_ms, q_ms;
            std::vector<float> y_fp, y_q;
            for (int i = 0; i < repeat; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                y_fp = ort.run(in);
                fp_ms.push_back(ms_since(t0));
                t0 = std::chrono::steady_clock::now();
                y_q = q8.run(in);
                q_ms.push_back(ms_since(t0));
            }
            float cos = cosine(y_fp.data(), y_q.data(), static_cast<int>(y_fp.size()));
            std::cout << "T " << in.t << "\n";
            std::cout << "fp32_ms " << median(fp_ms) << "\n";
            std::cout << "q8_ms " << median(q_ms) << "\n";
            std::cout << "fp32_samples ";
            for (size_t i = 0; i < fp_ms.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << fp_ms[i];
            }
            std::cout << "\n";
            std::cout << "q8_samples ";
            for (size_t i = 0; i < q_ms.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << q_ms[i];
            }
            std::cout << "\n";
            std::cout << "cosine " << cos << "\n";
            std::cout << "cosine_delta " << (1.f - cos) << "\n";
            std::cout << "max_abs " << max_abs(y_fp, y_q) << "\n";
            std::string dump_fp = arg(argc, argv, "--dump-fp32", "-");
            std::string dump_q = arg(argc, argv, "--dump-q8", "-");
            if (dump_fp != "-") write_f32(dump_fp, y_fp);
            if (dump_q != "-") write_f32(dump_q, y_q);
            return 0;
        }

        if (mode != "time") throw std::runtime_error("mode must be time, embed, or q8");
        std::string pack_path = arg(argc, argv, "--pack");
        Pack pack = load_pack(pack_path);
        int emb = embedding_dim(pack);
        Input in = load_input(input, true);
        int warmup = argi(argc, argv, "--warmup", 5);
        int repeat = argi(argc, argv, "--repeat", 20);

        for (int i = 0; i < warmup; ++i) ort.run(in);
        std::vector<double> ort_ms;
        std::vector<float> y_ort;
        for (int i = 0; i < repeat; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            y_ort = ort.run(in);
            ort_ms.push_back(ms_since(t0));
        }

        std::cout << "threads " << threads << "\n";
        std::cout << "T " << in.t << "\n";
        std::cout << "ort_cpp_ms " << median(ort_ms) << "\n";
        std::cout << "ort_cpp_samples ";
        for (size_t i = 0; i < ort_ms.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << ort_ms[i];
        }
        std::cout << "\n";
        std::cout << "ort_max_abs " << max_abs(y_ort, in.ref) << "\n";
        std::cout << "ort_cosine " << cosine(y_ort.data(), in.ref.data(), emb) << "\n";
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ort: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
