# NLA llama.cpp inference

> [!WARNING]
> This repo is codex slop

> [!TIP]
> The slop might be useful. Please give feedback.

Local prototype for running Natural Language Autoencoder inference with patched `llama.cpp` server.

Features a built-in extended [Mikupad](https://github.com/lmg-anon/mikupad) UI for NLA.

![NLA demo](demo.gif)

## Model roles

Full roundtrip uses three model roles, served from a **single `llama-server`** process using LoRA hot-switching:

1. **Base / target model** — the full Qwen2.5-7B GGUF
   - extracts original target activations
   - endpoint: `POST /extract`

2. **Actor / AV LoRA adapter**
   - activation vector → natural-language explanation
   - endpoint: `POST /explain`

3. **Critic / AR LoRA adapter** (carries `nla.value_head.weight`)
   - explanation → reconstructed activation
   - endpoints: `POST /reconstruct`, `POST /score`, `POST /edit-direction`

The server identifies adapters by their `nla.role` GGUF metadata (`av` = actor, `ar` = critic) and activates the correct adapter per-request.

## Default local ports

```text
18080  single llama-server (all endpoints)
3001   Mikupad UI (static file server)
```

## Added llama-server endpoints

### `/extract`

Base model only (all LoRA disabled).

```json
{
  "prompt": [151644, 872, 198],
  "layer": 20,
  "token_pos": -1,
  "add_special": false
}
```

Returns activation metadata and `activation: [3584 floats]`.

### `/explain`

Actor LoRA activated.

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

### `/reconstruct`

Critic LoRA activated. Reads `nla.value_head.weight` from the critic adapter GGUF.

```json
{
  "explanation": "..."
}
```

Returns reconstruction metadata and `activation: [3584 floats]`.

### `/score`

Critic LoRA activated.

```json
{
  "explanation": "...",
  "activation": [original extracted activation]
}
```

Returns MSE/cosine plus reconstruction metadata.

### `/edit-direction`

Critic LoRA activated.

```json
{
  "original_explanation": "...",
  "edited_explanation": "..."
}
```

Returns `direction = AR(edited_explanation) - AR(original_explanation)` plus norms/metadata.

### Steered `/completion`

Base model only. Normal `/completion` accepts optional token-level NLA steering:

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

```text
h ← h + alpha * ||h|| * direction / ||direction||
```

## Create GGUFs

GGUF files are generated locally from Hugging Face safetensors repos; they are not committed.

### Base model

```bash
uv run python convert_hf_to_gguf.py hf/base \
  --outfile models/base-q8_0.gguf \
  --outtype q8_0
```

### LoRA adapters

Extract LoRA adapters from the full actor/critic HF checkpoints against the base model using `mergekit-extract-lora`, then convert to GGUF, then inject NLA metadata.

```bash
# extract actor LoRA (from mergekit/ directory)
cd mergekit
uv run mergekit-extract-lora \
  --model ../hf/actor \
  --base-model ../hf/base \
  --out-path ../adapters/actor_r128 \
  --max-rank 128 --cuda --skip-undecomposable \
  --include-regex 'model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'

# extract critic LoRA (21 layers only)
uv run mergekit-extract-lora \
  --model ../hf/critic \
  --base-model ../hf/base \
  --out-path ../adapters/critic_r128 \
  --max-rank 128 --cuda --skip-undecomposable \
  --include-regex 'model\.layers\.([0-9]|1[0-9]|20)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
cd ..
```

Convert to GGUF:

```bash
# actor
uv run python convert_lora_to_gguf.py \
  --base hf/base \
  --outfile models/actor_lora.gguf \
  adapters/actor_r128

# critic
uv run python convert_lora_to_gguf.py \
  --base hf/base \
  --outfile models/critic_lora.gguf \
  adapters/critic_r128
```

Inject NLA metadata (and `value_head` tensor for critic):

```bash
# actor: role=av, no value_head
uv run python ../scripts/inject_nla_into_lora_gguf.py \
  --lora-gguf models/actor_lora.gguf \
  --meta-yaml hf/actor/nla_meta.yaml \
  --nla-role av \
  --output models/actor_lora_nla.gguf

# critic: role=ar, with value_head
uv run python ../scripts/inject_nla_into_lora_gguf.py \
  --lora-gguf models/critic_lora.gguf \
  --meta-yaml hf/critic/nla_meta.yaml \
  --nla-role ar \
  --value-head hf/critic/value_head.safetensors \
  --output models/critic_lora_nla.gguf
```

### Full-model GGUFs (legacy three-server mode)

The previous approach used three separate full-model servers. These are still supported but no longer recommended:

```bash
uv run python convert_hf_to_gguf.py hf/actor \
  --outfile models/actor-q8_0.gguf --outtype q8_0

uv run python convert_hf_to_gguf.py hf/critic \
  --outfile models/critic-nla-q8_0.gguf --outtype q8_0
```

## Build llama.cpp

From this directory:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
```

For CPU-only builds, omit `-DGGML_CUDA=ON`.

## Start services

Single server with `--actor` and `--critic` flags. The server auto-detects whether each file is a LoRA adapter or a full model GGUF:

```bash
build/bin/llama-server \
  -m models/base-q8_0.gguf \
  --actor models/actor_lora_nla.gguf \
  --critic models/critic_lora_nla.gguf \
  -ngl 99 \
  -c 2048 \
  --port 18080 \
  --host 127.0.0.1
```

You can also use full model GGUFs directly (no LoRA extraction needed):

```bash
build/bin/llama-server \
  -m models/base-q8_0.gguf \
  --actor models/actor-q8_0.gguf \
  --critic models/critic-nla-q8_0.gguf \
  -ngl 99 \
  -c 2048 \
  --port 18080 \
  --host 127.0.0.1
```

Or mix and match — actor as LoRA, critic as full model:

```bash
build/bin/llama-server \
  -m models/base-q8_0.gguf \
  --actor models/actor_lora_nla.gguf \
  --critic models/critic-nla-q8_0.gguf \
  -ngl 99 -c 2048 --port 18080
```

The server peeks at the GGUF `general.type` metadata to determine adapter vs. full model.

Alternatively, use generic `--lora` (server discovers roles from `nla.role` metadata):

```bash
build/bin/llama-server \
  -m models/base-q8_0.gguf \
  --lora models/actor_lora_nla.gguf,models/critic_lora_nla.gguf \
  -ngl 99 \
  -c 2048 \
  --port 18080 \
  --host 127.0.0.1
```

Serve the frontend:

```bash
python3 -m http.server 3001 --directory frontend
```

Then open:

```text
http://127.0.0.1:3001/
```

## Frontend

UI flow:

```text
hover token → Explain / NLA → /extract → /explain → modal
modal Score reconstruction button → /score
```

## Important files

```text
src/llama-adapter.cpp                     nla.* tensor skip in adapter loader
convert_hf_to_gguf.py                     patched converter (skip modules_to_save)
convert_lora_to_gguf.py                   patched (skip full-rank + bias tensors)
tools/server/server-context.cpp           LoRA hot-switching + NLA endpoints
frontend/index.html                       frontend
scripts/inject_nla_into_lora_gguf.py      inject NLA metadata into LoRA GGUFs
examples/extract-layer/                   activation extraction CLI
examples/nla-generate/                    actor injection/generation CLI
```
