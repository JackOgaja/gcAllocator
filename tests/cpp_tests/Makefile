#+
CXX = g++
NVCC = nvcc

# Get PyTorch paths
TORCH_PATH = $(shell python -c "import torch; import os; print(os.path.dirname(torch.__file__))")

TORCH_INCLUDE = -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include

TORCH_LIB = $(TORCH_PATH)/lib

# CRITICAL: Proper library order for PyTorch
LIBS = -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart

# CUDA paths
CUDA_PATH = /usr/local/cuda-12.8
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Runtime path
RPATH = -Wl,-rpath,$(TORCH_LIB)

CXXFLAGS = -std=c++17 -O2 -g -DUSE_CUDA -DGC_ALLOCATOR_INSTRUMENT_STATS=1
INCLUDES = $(TORCH_INCLUDE) -I$(CUDA_INCLUDE) -I../../gcAllocator/src
LDFLAGS = -L$(TORCH_LIB) -L$(CUDA_LIB) $(LIBS) $(RPATH)

SOURCES = test_allocator_direct.cpp \
          ../../gcAllocator/src/gc_allocator_core.cpp \
          ../../gcAllocator/src/allocator_stats_instrument.cpp \
          ../../gcAllocator/src/retry_strategy.cpp

TARGET = test_allocator_direct

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "Building with:"
	@echo "  TORCH_INCLUDE: $(TORCH_INCLUDE)"
	@echo "  TORCH_LIB: $(TORCH_LIB)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(TORCH_LIB):$$LD_LIBRARY_PATH ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

# Debug target to verify paths
debug-paths:
	@echo "TORCH_PATH: $(TORCH_PATH)"
	@echo "TORCH_INCLUDE: $(TORCH_INCLUDE)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "TORCH_LIB: $(TORCH_LIB)"


