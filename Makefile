CUDA_INSTALL_PATH ?= /usr/local/cuda
GCC_VER = -5

CXX := /usr/bin/g++
CC := /usr/bin/gcc
LINK := $(CXX) -fPIC
CCPATH := ./gcc
NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CXX)

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIB_CUDA := -L/usr/lib/nvidia-current -lcuda

# Options
NVCCOPTIONS = -arch=sm_86 -ptx -Wno-deprecated-gpu-targets
CXXOPTIONS = -O2 -std=c++0x
# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(CXXOPTIONS) $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

CUDA_OBJS = bfs.ptx
OBJS = main.cc.o digraph.cc.o
TARGET = main
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES:	.c	.cc	.cu	.o
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cc.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(LINKLINE)

clean:
	rm -rf $(TARGET) *.o *.ptx

#prepare:
#	rm -rf $(CCPATH);\
	mkdir -p $(CCPATH);\
	ln -s $(CXX) $(CCPATH)/g++;\
	ln -s $(CC) $(CCPATH)/gcc

