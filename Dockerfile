FROM nvidia/cuda:13.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Build tools
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (for test vector generation only)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . .

# Generate test vectors
RUN cd rust-reference && cargo build --release

# Build CUDA project
RUN mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make -j$(nproc)

CMD ["bash"]
