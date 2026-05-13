# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# ─── Build stage ───
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG CUDA_DOCKER_ARCH=default

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential cmake git libssl-dev libgomp1 ccache && \
    apt-get clean

ENV CC="ccache gcc-14" CXX="ccache g++-14" CUDAHOSTCXX="g++-14"

WORKDIR /app

# Copy the source tree for the native/CUDA build, but leave the app frontend out
# of this expensive stage so editing frontend/index.html does not invalidate it.
COPY --exclude=frontend . .

# Cache ccache across docker builds; build dir is fresh each time
RUN --mount=type=cache,target=/root/.ccache \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
          -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j$(nproc) \
          --target llama-server extract-layer nla-generate && \
    mkdir -p /app/out && \
    cp -P build/bin/llama-server build/bin/extract-layer build/bin/nla-generate /app/out/ && \
    find build -name "*.so*" -exec cp -P {} /app/out/ \;

# ─── Runtime stage ───
FROM ${BASE_CUDA_RUN_CONTAINER} AS nla

RUN apt-get update && \
    apt-get install -y libgomp1 curl caddy && \
    apt autoremove -y && apt clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

# Binaries
COPY --from=build /app/out/llama-server /usr/local/bin/
COPY --from=build /app/out/extract-layer /usr/local/bin/
COPY --from=build /app/out/nla-generate /usr/local/bin/

# Shared libs (symlinks preserved)
COPY --from=build /app/out/ /usr/local/lib/

# Caddy config + startup script
COPY Caddyfile /app/Caddyfile
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Frontend is copied directly from the build context and late in the Dockerfile,
# so changing frontend/index.html only rebuilds this cheap layer and metadata after it.
COPY frontend/ /app/frontend/

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LLAMA_ARG_HOST=0.0.0.0

EXPOSE 18090

WORKDIR /app
ENTRYPOINT ["/app/docker-entrypoint.sh"]
