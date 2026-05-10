#!/bin/bash
set -e

# Verify /models is a mounted volume, not part of the overlay filesystem
ROOT_DEV=$(stat -c "%d" /)
MODELS_DEV=$(stat -c "%d" /models)

if [ "$ROOT_DEV" = "$MODELS_DEV" ]; then
    echo "ERROR: /models is not a mounted volume."
    echo "Run with: docker run -v /path/to/models:/models:ro ..."
    exit 1
fi

is_interactive() {
    [ -t 0 ] && [ -t 1 ]
}

# ─── Interactive model setup ───
if [ -z "$BASE_MODEL" ] || [ -z "$ACTOR_MODEL" ] || [ -z "$CRITIC_MODEL" ]; then
    if ! is_interactive; then
        echo "ERROR: BASE_MODEL, ACTOR_MODEL, and CRITIC_MODEL must be set."
        echo "Run interactively with -it, or mount pre-downloaded models and set all three env vars."
        exit 1
    fi

    echo "========================================"
    echo "  NLA Inference — Model Setup"
    echo "========================================"
    echo ""
    echo "Some model paths are not configured."
    read -rp "Download models from Hugging Face? [y/N] " download
    download=${download:-N}

    if [ "$download" != "y" ] && [ "$download" != "Y" ]; then
        echo ""
        echo "Please mount your models to /models and set:"
        echo "  BASE_MODEL=/models/<base>.gguf"
        echo "  ACTOR_MODEL=/models/<actor>.gguf"
        echo "  CRITIC_MODEL=/models/<critic>.gguf"
        exit 1
    fi

    echo ""
    echo "Select quantization level:"
    echo "  1) BF16  — best quality, ~15 GB base"
    echo "  2) Q8_0  — high quality, ~8  GB base"
    echo "  3) Q6_K  — good quality, ~6  GB base"
    echo "  4) Q4_K_M — balanced,    ~5  GB base"
    read -rp "Choice [1-4, default 4]: " quant_choice
    quant_choice=${quant_choice:-4}

    case "$quant_choice" in
        1)
            quant="BF16"
            base_file="Qwen2.5-7B-Instruct-f16.gguf"
            actor_file="nla-qwen2.5-7b-L20-av-BF16.gguf"
            critic_file="nla-qwen2.5-7b-L20-ar-BF16.gguf"
            ;;
        2)
            quant="Q8_0"
            base_file="Qwen2.5-7B-Instruct-Q8_0.gguf"
            actor_file="nla-qwen2.5-7b-L20-av-Q8_0.gguf"
            critic_file="nla-qwen2.5-7b-L20-ar-Q8_0.gguf"
            ;;
        3)
            quant="Q6_K"
            base_file="Qwen2.5-7B-Instruct-Q6_K.gguf"
            actor_file="nla-qwen2.5-7b-L20-av-Q6_K.gguf"
            critic_file="nla-qwen2.5-7b-L20-ar-Q6_K.gguf"
            ;;
        4)
            quant="Q4_K_M"
            base_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            actor_file="nla-qwen2.5-7b-L20-av-Q4_K_M.gguf"
            critic_file="nla-qwen2.5-7b-L20-ar-Q4_K_M.gguf"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac

    echo ""
    echo "Downloading $quant models to /models ..."
    echo ""

    uvx hf download bartowski/Qwen2.5-7B-Instruct-GGUF "$base_file" \
        --local-dir /models
    uvx hf download thomasgauthier/nla-qwen2.5-7b-L20-av-GGUF "$actor_file" \
        --local-dir /models
    uvx hf download thomasgauthier/nla-qwen2.5-7b-L20-ar-GGUF "$critic_file" \
        --local-dir /models

    export BASE_MODEL="/models/$base_file"
    export ACTOR_MODEL="/models/$actor_file"
    export CRITIC_MODEL="/models/$critic_file"

    echo ""
    echo "Download complete."
    echo ""
fi

: "${BASE_MODEL:?Need BASE_MODEL env var}"
: "${ACTOR_MODEL:?Need ACTOR_MODEL env var}"
: "${CRITIC_MODEL:?Need CRITIC_MODEL env var}"

echo "Using models:"
echo "  Base:   $BASE_MODEL"
echo "  Actor:  $ACTOR_MODEL"
echo "  Critic: $CRITIC_MODEL"
echo ""

echo "Starting NLA inference stack..."

llama-server -m "${BASE_MODEL}" -ngl 99 -c 2048 --port 18080 --host 0.0.0.0 &
BASE_PID=$!

llama-server -m "${ACTOR_MODEL}" -ngl 99 -c 512 --port 18082 --host 0.0.0.0 --cache-ram 0 &
ACTOR_PID=$!

llama-server -m "${CRITIC_MODEL}" -ngl 0 -c 512 -np 1 --port 18084 --host 0.0.0.0 --cache-ram 0 &
CRITIC_PID=$!

# Wait for backends
for port in 18080 18082 18084; do
    until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        echo "Waiting for port ${port}..."
        sleep 1
    done
done

caddy run --config /app/Caddyfile &
CADDY_PID=$!

echo "NLA ready at http://localhost:18090/"

wait -n $BASE_PID $ACTOR_PID $CRITIC_PID $CADDY_PID
exit $?
