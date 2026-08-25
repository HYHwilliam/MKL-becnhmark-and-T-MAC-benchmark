CXX ?= g++
CROSS_CXX ?= aarch64-linux-gnu-g++

COMMON_FLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic
ARM_FLAGS := -march=armv8.2-a+fp16
SCALAR_FLAGS := -fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off
THREAD_FLAGS := -pthread

LOCAL_DIR := build/local
ARM64_DIR := build/arm64
NATIVE_DIR := build/native

BITS ?= 2
THREADS ?= 4

.PHONY: all native local cross verify qemu-verify clean

all: native

native: \
	$(NATIVE_DIR)/math_scalar_wxa16 \
	$(NATIVE_DIR)/tmac_scalar_wxa16 \
	$(NATIVE_DIR)/tmac_neon_st_wxa16 \
	$(NATIVE_DIR)/tmac_neon_mt_fixed_wxa16

local: \
	$(LOCAL_DIR)/math_scalar_wxa16 \
	$(LOCAL_DIR)/tmac_scalar_wxa16

cross: \
	$(ARM64_DIR)/math_scalar_wxa16 \
	$(ARM64_DIR)/tmac_scalar_wxa16 \
	$(ARM64_DIR)/tmac_neon_st_wxa16 \
	$(ARM64_DIR)/tmac_neon_mt_fixed_wxa16

$(LOCAL_DIR)/math_scalar_wxa16: math_scalar_wxa16.cpp benchmark_common.h
	mkdir -p $(dir $@)
	$(CXX) $(COMMON_FLAGS) $(SCALAR_FLAGS) $< -o $@

$(LOCAL_DIR)/tmac_scalar_wxa16: tmac_scalar_wxa16.cpp benchmark_common.h tmac_layout.h
	mkdir -p $(dir $@)
	$(CXX) $(COMMON_FLAGS) $(SCALAR_FLAGS) $< -o $@

$(ARM64_DIR)/math_scalar_wxa16: math_scalar_wxa16.cpp benchmark_common.h
	mkdir -p $(dir $@)
	$(CROSS_CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $(SCALAR_FLAGS) $< -o $@

$(ARM64_DIR)/tmac_scalar_wxa16: tmac_scalar_wxa16.cpp benchmark_common.h tmac_layout.h
	mkdir -p $(dir $@)
	$(CROSS_CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $(SCALAR_FLAGS) $< -o $@

$(ARM64_DIR)/tmac_neon_st_wxa16: tmac_neon_st_wxa16.cpp benchmark_common.h tmac_layout.h
	mkdir -p $(dir $@)
	$(CROSS_CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $< -o $@

$(ARM64_DIR)/tmac_neon_mt_fixed_wxa16: tmac_neon_mt_fixed_wxa16.cpp benchmark_common.h tmac_layout.h
	mkdir -p $(dir $@)
	$(CROSS_CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $(THREAD_FLAGS) $< -o $@

$(NATIVE_DIR)/math_scalar_wxa16: math_scalar_wxa16.cpp benchmark_common.h
	mkdir -p $(dir $@)
	$(CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $(SCALAR_FLAGS) $< -o $@

$(NATIVE_DIR)/tmac_scalar_wxa16: tmac_scalar_wxa16.cpp benchmark_common.h tmac_layout.h
	mkdir -p $(dir $@)
	$(CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $(SCALAR_FLAGS) $< -o $@

$(NATIVE_DIR)/tmac_neon_st_wxa16: tmac_neon_st_wxa16.cpp benchmark_common.h tmac_layout.h
	mkdir -p $(dir $@)
	$(CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $< -o $@

$(NATIVE_DIR)/tmac_neon_mt_fixed_wxa16: tmac_neon_mt_fixed_wxa16.cpp benchmark_common.h tmac_layout.h
	mkdir -p $(dir $@)
	$(CXX) $(COMMON_FLAGS) $(ARM_FLAGS) $(THREAD_FLAGS) $< -o $@

verify: native
	./$(NATIVE_DIR)/math_scalar_wxa16 --bits $(BITS) --verify-only
	./$(NATIVE_DIR)/tmac_scalar_wxa16 --bits $(BITS) --verify-only
	./$(NATIVE_DIR)/tmac_neon_st_wxa16 --bits $(BITS) --verify-only
	./$(NATIVE_DIR)/tmac_neon_mt_fixed_wxa16 --bits $(BITS) --threads $(THREADS) --verify-only

qemu-verify: cross
	qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu ./$(ARM64_DIR)/math_scalar_wxa16 --bits $(BITS) --verify-only
	qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu ./$(ARM64_DIR)/tmac_scalar_wxa16 --bits $(BITS) --verify-only
	qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu ./$(ARM64_DIR)/tmac_neon_st_wxa16 --bits $(BITS) --verify-only
	qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu ./$(ARM64_DIR)/tmac_neon_mt_fixed_wxa16 --bits $(BITS) --threads $(THREADS) --verify-only

clean:
	rm -rf build
