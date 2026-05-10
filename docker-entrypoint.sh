#!/bin/bash
set -e

: "${BASE_MODEL:?Need BASE_MODEL env var}"
: "${ACTOR_MODEL:?Need ACTOR_MODEL env var}"
: "${CRITIC_MODEL:?Need CRITIC_MODEL env var}"

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
