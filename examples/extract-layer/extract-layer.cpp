// extract-layer.cpp — Extract residual-stream activations at a specific layer
// from a GGUF model using llama.cpp's eval callback mechanism.
//
// Usage:
//   extract-layer -m model.gguf -p "prompt text" --layer 20
//
// Outputs a summary to stderr and writes the raw fp32 activation vector
// for the last token position to the file specified by --out (or stdout).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Capture state for the callback
struct capture_state {
    int32_t target_layer;
    int32_t l_out_count  = 0;
    int64_t n_embd       = 0;
    int64_t n_tokens     = 0;
    std::vector<float> captured;
    bool found           = false;
};

static bool extract_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;

    if (ask) {
        return true; // always compute
    }

    std::string name(t->name);

    // Log every tensor name for debugging (only first decode)
    static int log_count = 0;
    if (log_count < 5) {
        LOG_INF("cb: name='%s' ne=[%lld,%lld]\n",
                t->name, (long long)t->ne[0], (long long)t->ne[1]);
        log_count++;
    }

    // Per-layer residual stream output is named "l_out-<layer_idx>"
    // e.g. "l_out-0", "l_out-1", ..., "l_out-27" for Qwen2.5-7B (28 layers)
    std::string target_name = "l_out-" + std::to_string(st->target_layer);
    if (name == target_name) {
        int64_t n_el = ggml_nelements(t);
        LOG_INF("CAPTURED layer %d: %lld elements, n_tokens=%lld\n",
                st->target_layer, (long long)n_el, (long long)t->ne[1]);
        st->captured.resize(n_el);
        st->n_tokens = t->ne[1];

        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (is_host) {
            for (int64_t i = 0; i < n_el; i++) {
                st->captured[i] = ggml_get_f32_1d(t, i);
            }
        } else {
            auto n_bytes = ggml_nbytes(t);
            std::vector<uint8_t> buf(n_bytes);
            ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);
            ggml_tensor tmp = *t;
            tmp.data = buf.data();
            tmp.buffer = nullptr;
            for (int64_t i = 0; i < n_el; i++) {
                st->captured[i] = ggml_get_f32_1d(&tmp, i);
            }
        }
        st->found = true;
        st->l_out_count++;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "Hello, how are you?";

    // Custom args — strip our custom flags before common_params_parse sees them
    int32_t target_layer = 20;
    std::string output_file;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--layer" && i + 1 < argc) {
            target_layer = std::atoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    // Parse common params (handles -m, -p, -ngl, etc.)
    if (!common_params_parse((int)filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // CPU only (VRAM occupied)
    params.n_gpu_layers = 0;

    // Set up capture callback
    capture_state state;
    state.target_layer = target_layer;

    params.cb_eval = extract_cb;
    params.cb_eval_user_data = &state;
    params.warmup = false; // skip warmup to avoid triggering callback

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    state.n_embd = llama_model_n_embd(model);

    // Tokenize
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens\n", __func__);
        return 1;
    }

    LOG_INF("%s: %zu tokens, extracting layer %d, d_model=%d\n",
            __func__, tokens.size(), target_layer, (int)state.n_embd);

    // Enable embeddings so the graph includes the embedding output path
    llama_set_embeddings(ctx, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (batch.logits) {
        batch.logits[batch.n_tokens - 1] = true;
    }

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LOG_ERR("%s: llama_decode failed (%d)\n", __func__, ret);
        return 1;
    }

    if (!state.found) {
        LOG_ERR("%s: failed to capture layer %d (saw %d l_out tensors)\n",
                __func__, target_layer, state.l_out_count);
        return 1;
    }

    // Extract last token position
    int64_t pos = state.n_tokens - 1;
    std::vector<float> activation(state.n_embd);
    // ggml layout: [n_embd, n_tokens] — column-major in ggml terms,
    // but ggml_get_f32_1d linearizes, so element (i, pos) = pos*n_embd + i
    for (int64_t i = 0; i < state.n_embd; i++) {
        activation[i] = state.captured[pos * state.n_embd + i];
    }

    // L2 norm
    float l2 = 0.0f;
    for (auto v : activation) l2 += v * v;
    l2 = std::sqrt(l2);

    // Write output
    if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(activation.data()),
                  activation.size() * sizeof(float));
        ofs.close();
        LOG_INF("Wrote %zu floats to %s\n", activation.size(), output_file.c_str());
    }

    // Summary to stderr
    LOG("\n=== Layer %d Activation (last token, pos=%lld) ===\n", target_layer, (long long)pos);
    LOG("Prompt: %s\n", params.prompt.c_str());
    LOG("Tokens: %zu total\n", tokens.size());
    LOG("d_model: %d\n", (int)state.n_embd);
    LOG("Vector: %zu floats (%zu bytes)\n", activation.size(), activation.size() * sizeof(float));
    LOG("L2 norm: %.4f\n", l2);
    LOG("First 8: ");
    for (int i = 0; i < (int)std::min((size_t)8, activation.size()); i++) {
        LOG("%.6f ", activation[i]);
    }
    LOG("\n");
    LOG("Model layers: %d, l_out count: %d\n",
        llama_model_n_layer(model), state.l_out_count);

    llama_backend_free();
    return 0;
}
