# Makefile for get_features — C feature extractor for LOB+trades parquet files.
#
# Dependencies:
#   - Carquet (pure C parquet library, auto-fetched as submodule)
#   - CMake 3.16+
#   - C11 compiler
#
# Carquet auto-fetches zstd/zlib/lz4 if not found on the system.
#
# Usage:
#   make          — build everything
#   make release  — optimized build
#   make clean    — remove build artifacts
#   ./build/get_features data.parquet -o features.csv

.PHONY: all release clean debug

BUILD_DIR ?= build
BUILD_TYPE ?= Release

all: $(BUILD_DIR)/get_features

debug: BUILD_TYPE=Debug
debug: $(BUILD_DIR)/get_features

release: BUILD_TYPE=Release
release: $(BUILD_DIR)/get_features

$(BUILD_DIR)/get_features: carquet/CMakeLists.txt CMakeLists.txt lobcore.h lobcore.c main.c
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@echo ""
	@echo "Build complete: $(BUILD_DIR)/get_features"
	@echo "Usage: $(BUILD_DIR)/get_features <input.parquet> [options]"

clean:
	rm -rf $(BUILD_DIR)

# Build with native arch optimizations for maximum performance
native: BUILD_TYPE=Release
native:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON
	cmake --build $(BUILD_DIR) -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@echo ""
	@echo "Optimized build complete: $(BUILD_DIR)/get_features"
