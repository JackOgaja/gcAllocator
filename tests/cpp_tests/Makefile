# tests/cpp_tests/Makefile
CXX = g++
PYTHON = python

# Simple, single-expression Python queries
PYTHON_INCLUDE = $(shell $(PYTHON) -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/include')")
TORCH_INCLUDE = $(shell $(PYTHON) -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")
TORCH_LIB = $(shell $(PYTHON) -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")

# Validate paths were found (Make conditional)
ifeq ($(TORCH_INCLUDE),)
    $(error PyTorch not found. Install: pip install torch)
endif

CXXFLAGS = -std=c++17 -O2 -g -DGC_ALLOCATOR_INSTRUMENT_STATS=1
INCLUDES = -I$(PYTHON_INCLUDE) -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include -I../../gcAllocator/src
LDFLAGS = -L$(TORCH_LIB) -ltorch -ltorch_cuda -lc10 -lc10_cuda -lcudart

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
	@echo "PYTHON_INCLUDE: $(PYTHON_INCLUDE)"
	@echo "TORCH_INCLUDE: $(TORCH_INCLUDE)"
	@echo "TORCH_LIB: $(TORCH_LIB)"
