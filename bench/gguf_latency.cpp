// Same tokens, shapes, f32 caches, warmups and host-logit readback as gguf_latency.rs.
// Build against a recorded llama.cpp checkout, not a Python wheel.
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char ** argv) {
    if (argc != 3) { std::cerr << "gguf_latency model.gguf output-prefix\n"; return 2; }
    auto start = Clock::now();
    ggml_backend_load_all();
    auto device = ggml_backend_dev_by_name("Vulkan0");
    if (!device) { throw std::runtime_error("Vulkan0 unavailable; no CPU fallback"); }
    ggml_backend_dev_t devices[] = {device, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) { throw std::runtime_error("model load failed"); }
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = cp.n_ubatch = 128;
    cp.n_threads = cp.n_threads_batch = 6;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) { throw std::runtime_error("context creation failed"); }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const double prepare = ms(start);
    auto run = [&](int position, int count) {
        std::vector<llama_token> tokens(count);
        for (int i = 0; i < count; ++i) { tokens[i] = 42 + (position + i) % 31; }
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), count))) {
            throw std::runtime_error("decode failed");
        }
        auto logits = llama_get_logits_ith(ctx, -1);
        std::vector<float> out(logits, logits + vocab);
        for (float x : out) { if (!std::isfinite(x)) { throw std::runtime_error("nonfinite logits"); } }
        return out;
    };
    std::vector<double> prefill_ms, decode_ms;
    std::vector<float> outputs;
    for (int sample = 0; sample < 10; ++sample) {
        llama_memory_clear(llama_get_memory(ctx), false);
        start = Clock::now();
        auto logits = run(0, 128);
        double elapsed = ms(start);
        if (sample >= 3) { prefill_ms.push_back(elapsed); }
        if (sample == 3) { outputs.insert(outputs.end(), logits.begin(), logits.end()); }
        for (int pos = 128; pos < 160; ++pos) {
            start = Clock::now();
            logits = run(pos, 1);
            elapsed = ms(start);
            if (sample >= 3) { decode_ms.push_back(elapsed); }
            if (sample == 3) { outputs.insert(outputs.end(), logits.begin(), logits.end()); }
        }
    }
    std::ofstream raw(std::string(argv[2]) + ".logits.f32", std::ios::binary);
    raw.write(reinterpret_cast<const char *>(outputs.data()), outputs.size() * sizeof(float));
    std::ofstream result(std::string(argv[2]) + ".json");
    result << "{\"engine\":\"llama.cpp\",\"prompt\":128,\"decode\":32,\"context\":256,\"cache\":\"f32\",\"vocab\":"
           << vocab << ",\"prepare_ms\":" << prepare;
    auto write = [&](const char * name, const std::vector<double>& values) {
        result << ",\"" << name << "\":[";
        for (size_t i = 0; i < values.size(); ++i) { result << (i ? "," : "") << values[i]; }
        result << "]";
    };
    write("prefill_ms", prefill_ms);
    write("decode_ms", decode_ms);
    result << "}\n";
    llama_free(ctx);
    llama_model_free(model);
}
