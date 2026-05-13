# NLA llama.cpp inference

[Natural Language Autoencoder](https://www.anthropic.com/research/natural-language-autoencoders) inference with a patched llama.cpp server
and a built-in [Mikupad](https://github.com/lmg-anon/mikupad) UI.

![NLA demo](demo.gif)

Hover any token in the editor, click **Explain NLA**, edit the explanation,
and apply steering directly back into the model.

## Quickstart

### Docker (recommended)

```bash
docker run -it --gpus all -p 18090:18090 \
  -v /path/to/models/where/you/want/models/saved:/models \
  ghcr.io/thomasgauthier/nla.cpp:cuda
```

The container prompts for a quantization level and downloads
Qwen2.5-7B base, actor, and critic models automatically.

If you already have models:

```bash
docker run --gpus all -p 18090:18090 \
  -v /path/to/models:/models:ro \
  -e BASE_MODEL=/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -e ACTOR_MODEL=/models/nla-qwen2.5-7b-L20-av-Q4_K_M.gguf \
  -e CRITIC_MODEL=/models/nla-qwen2.5-7b-L20-ar-Q4_K_M.gguf \
  ghcr.io/thomasgauthier/nla.cpp:cuda
```

Then open `http://localhost:18090/`.

For CPU-only, use `ghcr.io/thomasgauthier/nla.cpp:cpu` instead.

### Manual build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
```

For CPU-only builds, omit `-DGGML_CUDA=ON`.

Start the three-server stack manually:

```bash
# base server
build/bin/llama-server \
  -m models/base-q8_0.gguf \
  -ngl 99 -c 2048 --port 18080 --host 127.0.0.1

# actor server
build/bin/llama-server \
  -m models/actor-q8_0.gguf \
  -ngl 99 -c 512 --port 18082 --host 127.0.0.1 --cache-ram 0

# critic server (must contain nla.value_head.weight)
build/bin/llama-server \
  -m models/critic-nla-q8_0.gguf \
  -ngl 0 -c 512 -np 1 --port 18084 --host 127.0.0.1 --cache-ram 0

# Caddy proxy + frontend
caddy run --config Caddyfile
```

Then open `http://127.0.0.1:18090/`.

## API

### `POST /extract`

Extract the hidden-state activation of a specific token at a specific layer.

```json
{
  "prompt": [151644, 872, 198],
  "layer": 20,
  "token_pos": -1,
  "add_special": false
}
```

Returns activation metadata and `activation: [3584 floats]`.

### `POST /explain`

Generate a natural-language explanation for an activation vector.

```json
{
  "activation": [3584 floats],
  "input_ids": [actor prompt token ids],
  "inject_pos": 111,
  "scale": 150,
  "n_predict": 200
}
```

Returns generated text and parsed explanation.

### `POST /reconstruct`

Reconstruct an activation vector from a natural-language explanation.

```json
{
  "explanation": "..."
}
```

Returns reconstruction metadata and `activation: [3584 floats]`.

### `POST /score`

Score how well an explanation reconstructs the original activation.

```json
{
  "explanation": "...",
  "activation": [original extracted activation]
}
```

Returns MSE, cosine similarity, and reconstruction metadata.

### `POST /edit-direction`

Compute the steering direction between two explanations.

```json
{
  "original_explanation": "...",
  "edited_explanation": "..."
}
```

Returns `direction = AR(edited) - AR(original)` plus norms and metadata.

### Steered `POST /completion`

Normal `/completion` with optional token-level steering:

```json
{
  "prompt": "...",
  "n_predict": 128,
  "steering": [
    {
      "token_pos": 123,
      "layer": 20,
      "alpha": 1.0,
      "direction": [3584 floats]
    }
  ]
}
```

At that prompt token/layer, the server applies:

```
h ← h + alpha * ||h|| * direction / ||direction||
```

## Model roles

The NLA roundtrip uses three models, each served by its own `llama-server`:

1. **Base model** — the full Qwen2.5-7B GGUF
   - extracts original target activations
   - endpoint: `POST /extract`

2. **Actor model**
   - activation vector → natural-language explanation
   - endpoint: `POST /explain`

3. **Critic model** (must contain `nla.value_head.weight`)
   - explanation → reconstructed activation
   - endpoints: `POST /reconstruct`, `POST /score`, `POST /edit-direction`

## Creating GGUFs

### Actor

```bash
# Download the actor checkpoint
uvx hf download kitft/nla-qwen2.5-7b-L20-av --local-dir actor_hf

# Convert directly to Q4_K_M
uv run python convert_hf_to_gguf.py actor_hf \
  --outfile models/nla-qwen2.5-7b-L20-av-Q4_K_M.gguf \
  --outtype Q4_K_M
```

### Critic

The critic contains `nla.value_head.weight`, which `llama-quantize` drops.
Keep a BF16 copy as the source, then inject the tensor back after quantization.

```bash
# Download the critic checkpoint
uvx hf download kitft/nla-qwen2.5-7b-L20-ar --local-dir critic_hf

# Convert to BF16 (preserves value_head and NLA metadata)
uv run python convert_hf_to_gguf.py critic_hf \
  --outfile models/nla-qwen2.5-7b-L20-ar-BF16.gguf \
  --outtype bf16

# Quantize (value_head is lost here)
./build/bin/llama-quantize \
  models/nla-qwen2.5-7b-L20-ar-BF16.gguf \
  models/nla-qwen2.5-7b-L20-ar-Q4_K_M.gguf \
  Q4_K_M

# Inject value_head back from the BF16 source
uv run python scripts/inject_value_head.py \
  --from-src models/nla-qwen2.5-7b-L20-ar-BF16.gguf \
  --into   models/nla-qwen2.5-7b-L20-ar-Q4_K_M.gguf \
  --output models/nla-qwen2.5-7b-L20-ar-Q4_K_M.gguf
```

### Base model

```bash
uvx hf download bartowski/Qwen2.5-7B-Instruct-GGUF \
  Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --local-dir models
```

Or convert from a HF checkpoint:

```bash
uv run python convert_hf_to_gguf.py Qwen/Qwen2.5-7B-Instruct \
  --outfile models/base-Q4_K_M.gguf \
  --outtype Q4_K_M
```

## Important files

```
convert_hf_to_gguf.py                     patched converter (handles nla.* tensors)
tools/server/server-context.cpp           NLA endpoints (extract, explain, reconstruct, score)
frontend/index.html                       Mikupad-based UI
scripts/inject_value_head.py              copy value_head into quantized critic GGUF
examples/extract-layer/                   activation extraction CLI
examples/nla-generate/                    actor injection/generation CLI
```
