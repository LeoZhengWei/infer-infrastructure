#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct W {
    std::vector<int> shape;
    std::vector<float> data;
};

struct Pack {
    std::unordered_map<std::string, W> w;
};

Pack load_pack(const std::string& path);
int embedding_dim(const Pack& pack);
void mlp_forward(const Pack& pack, const float* in, int batch, int frames, int feat, float* out, int threads);
