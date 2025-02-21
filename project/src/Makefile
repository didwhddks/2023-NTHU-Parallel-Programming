NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_75 
LDFLAGS  := -lm
CXXFLAGS := -O3 -lm
EXES     := MCP_serial
ALLS     := MCP_serial generate_testcase calc MCP_GPU_baseline MCP_GPU_coalesced MCP_GPU_optimized MCP_GPU_unroll	

MCP_GPU_unroll: NVFLAGS += -Xcompiler="-fopenmp"

alls: $(EXES)

clean:
	rm -f $(ALLS)

MCP_serial: MCP_serial.cc
	g++ $(CXXFLAGS) -o $@ $?

generate_testcase: generate_testcase.cc
	g++ $(CXXFLAGS) -o $@ $?

calc: calc.cc
	g++ $(CXXFLAGS) -o $@ $?

MCP_GPU_baseline: MCP_GPU_baseline.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

MCP_GPU_coalesced: MCP_GPU_coalesced.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

MCP_GPU_optimized: MCP_GPU_optimized.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

MCP_GPU_unroll: MCP_GPU_unroll.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?