.PHONY: all build test bench clean docker-build docker-test vectors

BUILD_DIR := build
CUDA_ARCH ?= 75

all: build

build:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) && make -j$$(nproc)

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

bench: build
	cd $(BUILD_DIR) && cmake .. -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) -DENABLE_PHASE4=ON && make -j$$(nproc) bench_msm && ./bench_msm

vectors:
	cd rust-reference && cargo run --release -- generate

clean:
	rm -rf $(BUILD_DIR)

docker-build:
	docker compose build

docker-test:
	docker compose run --rm cuda-verkle bash -c "cd build && ctest --output-on-failure"

docker-bench:
	docker compose run --rm cuda-verkle bash -c "cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 -DENABLE_PHASE4=ON && make -j$$(nproc) bench_msm && ./bench_msm"
