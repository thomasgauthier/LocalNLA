#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m actor.gguf --ids IDS --inject-pos P --activation vec.{bin,npy} [options]\n"
        "\n"
        "NLA actor generation for llama.cpp: evaluate prompt token IDs, replace one\n"
        "position with an input embedding vector, then sample.\n"
        "\n"
        "custom options:\n"
        "  --ids IDS              comma/space separated token ids\n"
        "  --ids-file FILE        token ids from file (comma/space/newline separated)\n"
        "  --inject-pos P         0-based token position to replace with activation\n"
        "  --activation FILE      raw fp32 .bin or little-endian float32 .npy vector\n"
        "  --scale S              L2-normalize activation to S before injection (default: 150)\n"
        "  --no-normalize         inject activation exactly as provided\n"
        "  --n-predict N          max generated tokens (default: common -n or 200)\n"
        "  --temp T               temperature for sampler (default: common temp)\n"
        "\n"
        "also accepts common llama.cpp args: -m, -ngl, -c, -t, -s, etc.\n",
        argv0);
}

static std::string read_text_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<llama_token> parse_ids_text(std::string s) {
    for (char & c : s) {
        if (c == ',' || c == '[' || c == ']') c = ' ';
    }
    std::istringstream iss(s);
    std::vector<llama_token> ids;
    long long x;
    while (iss >> x) ids.push_back((llama_token) x);
    return ids;
}

static std::vector<float> read_activation(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open activation file " + path);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() >= 10 && bytes[0] == char(0x93) && std::memcmp(bytes.data() + 1, "NUMPY", 5) == 0) {
        const uint8_t major = (uint8_t) bytes[6];
        size_t header_len_off = 8;
        size_t header_len_size = major <= 1 ? 2 : 4;
        if (bytes.size() < header_len_off + header_len_size) throw std::runtime_error("bad npy header");
        uint32_t hlen = 0;
        for (size_t i = 0; i < header_len_size; ++i) hlen |= ((uint8_t) bytes[header_len_off + i]) << (8*i);
        size_t data_off = header_len_off + header_len_size + hlen;
        if (data_off > bytes.size()) throw std::runtime_error("bad npy data offset");
        std::string hdr(bytes.data() + header_len_off + header_len_size, hlen);
        if (hdr.find("'descr': '<f4'") == std::string::npos && hdr.find("\"descr\": \"<f4\"") == std::string::npos) {
            throw std::runtime_error("only little-endian float32 .npy is supported");
        }
        size_t nbytes = bytes.size() - data_off;
        if (nbytes % sizeof(float) != 0) throw std::runtime_error("npy data is not float aligned");
        std::vector<float> v(nbytes / sizeof(float));
        std::memcpy(v.data(), bytes.data() + data_off, nbytes);
        return v;
    }
    if (bytes.size() % sizeof(float) != 0) throw std::runtime_error("raw activation size is not float aligned");
    std::vector<float> v(bytes.size() / sizeof(float));
    std::memcpy(v.data(), bytes.data(), bytes.size());
    return v;
}

static void normalize_to(std::vector<float> & v, float scale) {
    double ss = 0.0;
    for (float x : v) ss += double(x) * double(x);
    double n = std::sqrt(ss);
    if (n < 1e-12) return;
    float a = (float) (scale / n);
    for (float & x : v) x *= a;
}

static bool eval_tokens(llama_context * ctx, const std::vector<llama_token> & toks, int start_pos, bool logits_last) {
    if (toks.empty()) return true;
    llama_batch b = llama_batch_init((int32_t)toks.size(), 0, 1);
    b.n_tokens = (int32_t)toks.size();
    for (int32_t i = 0; i < b.n_tokens; ++i) {
        b.token[i] = toks[i];
        b.pos[i] = start_pos + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = logits_last && i == b.n_tokens - 1;
    }
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static bool eval_embedding(llama_context * ctx, const std::vector<float> & embd, int pos, bool logits) {
    int n_embd = (int) embd.size();
    llama_batch b = llama_batch_init(1, n_embd, 1);
    b.n_tokens = 1;
    std::memcpy(b.embd, embd.data(), sizeof(float) * embd.size());
    b.pos[0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = logits;
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string ids_arg, ids_file, activation_file;
    int inject_pos = -1;
    float scale = 150.0f;
    bool do_normalize = true;
    int n_predict_override = -1;

    std::vector<char *> filtered;
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--ids" && i + 1 < argc) ids_arg = argv[++i];
        else if (a == "--ids-file" && i + 1 < argc) ids_file = argv[++i];
        else if (a == "--inject-pos" && i + 1 < argc) inject_pos = std::atoi(argv[++i]);
        else if (a == "--activation" && i + 1 < argc) activation_file = argv[++i];
        else if (a == "--scale" && i + 1 < argc) scale = std::atof(argv[++i]);
        else if (a == "--no-normalize") do_normalize = false;
        else if (a == "--n-predict" && i + 1 < argc) n_predict_override = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else filtered.push_back(argv[i]);
    }

    common_params params;
    params.n_predict = 200;
    if (!common_params_parse((int)filtered.size(), filtered.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (n_predict_override >= 0) params.n_predict = n_predict_override;
    params.warmup = false;

    if ((ids_arg.empty() == ids_file.empty()) || inject_pos < 0 || activation_file.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::vector<llama_token> ids = ids_arg.empty() ? parse_ids_text(read_text_file(ids_file)) : parse_ids_text(ids_arg);
    if (ids.empty()) { LOG_ERR("no token ids supplied\n"); return 1; }
    if (inject_pos < 0 || inject_pos >= (int) ids.size()) { LOG_ERR("inject-pos out of range\n"); return 1; }

    std::vector<float> act;
    try { act = read_activation(activation_file); }
    catch (const std::exception & e) { LOG_ERR("%s\n", e.what()); return 1; }

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    if ((int)act.size() != n_embd) { LOG_ERR("activation length %zu != model n_embd %d\n", act.size(), n_embd); return 1; }
    if (do_normalize) normalize_to(act, scale);

    LOG_INF("nla-generate: ids=%zu inject_pos=%d n_embd=%d n_predict=%d normalize=%s scale=%.3f\n",
            ids.size(), inject_pos, n_embd, params.n_predict, do_normalize ? "true" : "false", scale);

    std::vector<llama_token> prefix(ids.begin(), ids.begin() + inject_pos);
    std::vector<llama_token> suffix(ids.begin() + inject_pos + 1, ids.end());

    if (!eval_tokens(ctx, prefix, 0, false)) { LOG_ERR("failed eval prefix\n"); return 1; }
    if (!eval_embedding(ctx, act, inject_pos, suffix.empty())) { LOG_ERR("failed eval injected embedding\n"); return 1; }
    if (!eval_tokens(ctx, suffix, inject_pos + 1, true)) { LOG_ERR("failed eval suffix\n"); return 1; }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.sampling.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.sampling.top_p, params.sampling.min_keep));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.sampling.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    std::string out;
    llama_token tok;
    for (int i = 0; i < params.n_predict; ++i) {
        tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char buf[1024];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) { out.append(buf, n); fwrite(buf, 1, n, stdout); fflush(stdout); }
        std::vector<llama_token> one = { tok };
        if (!eval_tokens(ctx, one, (int)ids.size() + i, true)) { LOG_ERR("failed eval generated token\n"); break; }
    }
    printf("\n");

    llama_sampler_free(smpl);
    return 0;
}
