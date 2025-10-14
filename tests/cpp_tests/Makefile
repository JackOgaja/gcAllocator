# Build direct C++ tests for GCAllocator

CXX = g++
PYTHON_INCLUDE = $(shell python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/include')")
TORCH_INCLUDE = $(shell python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")
TORCH_LIB = $(shell python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")

CXXFLAGS = -std=c++17 -O2 -g -DGC_ALLOCATOR_INSTRUMENT_STATS=1
INCLUDES = -I$(PYTHON_INCLUDE) -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE)/torch/csrc/api/include
LDFLAGS = -L$(TORCH_LIB) -ltorch -ltorch_cuda -lc10 -lc10_cuda -lcudart

SOURCES = test_allocator_direct.cpp \
          ../gcAllocator/src/gc_allocator_core.cpp \
          ../gcAllocator/src/allocator_stats.cpp \
          ../gcAllocator/src/retry_strategy.cpp

TARGET = test_allocator_direct

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(TORCH_LIB):$$LD_LIBRARY_PATH ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
