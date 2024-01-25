#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <chrono>

#define thread_per_block 1024
#define maxN 10000

const long long INF = 2E18;
int N;
long long *dp_host, *cut_host;
int *p_host;

__constant__ int p_constant[maxN + 1];

__host__ __device__ int convertIdx(int len, int i, int N) {
    return len * N + i;
}

void input() {
    FILE *input_file = fopen("testcase", "r");
    fread(&N, sizeof(int), 1, input_file);

    // malloc host memory
    cudaMallocHost((void **) &p_host, (N + 1) * sizeof(int));
    cudaMallocHost((void **) &dp_host, (N + 1) * N * sizeof(long long));
    cudaMallocHost((void **) &cut_host, (N + 1) * N * sizeof(long long));

    // read dimension of each matrix to p_host
    fread(p_host, sizeof(int), N + 1, input_file);
    fclose(input_file);

    // initialize dp_host & cut_host
    for (int len = 1; len <= N; ++len) {
        for (int i = 0; i < N; ++i) {
            dp_host[convertIdx(len, i, N)] = len == 1 ? 0 : INF;
            cut_host[convertIdx(len, i, N)] = -1;
        }
    }
}

__global__ void oneThreadPerEntry(long long *dp_device, long long *cut_device, int len, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i + len - 1;

    if (i >= N || j >= N) {
        return;
    }

    __shared__ long long dp_shared[4 * 1024];

    for (int k = 0; k < 4; ++k) {
        int idx = convertIdx(len - k - 1, i + k + 1, N);
        dp_shared[convertIdx(k, threadIdx.x, blockDim.x)] = 0 <= idx && idx < (N + 1) * N ? dp_device[idx] : INF;
    }

    __syncthreads();

    long long minimum = dp_device[convertIdx(len, i, N)];
    long long cut_point = -1;
    long long i_dim = p_constant[i];
    long long j_dim = p_constant[j + 1];

    int i_offset = blockIdx.x * blockDim.x;

    #pragma unroll
    for (int k = i; k < j; ++k) {
        int left_idx = convertIdx(k - i + 1, i, N);
        int right_idx = convertIdx(j - k, k + 1, N);

        int diff1 = len - k + i - 1;
        int diff2 = len - j + k;
        int bound1 = convertIdx(k - i + 1, i_offset + diff1, N);
        int bound2 = convertIdx(j - k, i_offset + diff2, N);

        long long left_cost = diff1 <= 4 && bound1 <= left_idx ? dp_shared[convertIdx(diff1 - 1, i - diff1 - i_offset, blockDim.x)] : dp_device[left_idx];
        long long right_cost = diff2 <= 4 && bound2 <= right_idx ? dp_shared[convertIdx(diff2 - 1, k + 1 - diff2 - i_offset, blockDim.x)] : dp_device[right_idx];

        long long cost = left_cost + right_cost + 1LL * i_dim * j_dim * p_constant[k + 1];
        cut_point = cost < minimum ? k : cut_point;
        minimum = min(minimum, cost);
    }

    dp_device[convertIdx(len, i, N)] = minimum;
    cut_device[convertIdx(len, i, N)] = cut_point;
}

int main() {

    double execution_time = 0.0;
    auto start = std::chrono::steady_clock::now();

    cudaSetDevice(0);
    input();

    long long *dp_device, *cut_device;
    // int *p_device;
    // cudaMalloc((void **) &p_device, (N + 1) * sizeof(int));
    cudaMalloc((void **) &dp_device, (N + 1) * N * sizeof(long long));
    cudaMalloc((void **) &cut_device, (N + 1) * N * sizeof(long long));

    // cudaMemcpy(p_device, p_host, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(p_constant, p_host, (N + 1) * sizeof(int));
    cudaMemcpy(dp_device, dp_host, (N + 1) * N * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(cut_device, cut_host, (N + 1) * N * sizeof(long long), cudaMemcpyHostToDevice);

    // TODO: implement the parallel version of matrix chain multiplication here
    // Can try OneThreadPerEntry, OneBlockPerEntry, MultipleBlocksPerEntry
    // Can try shared memory, coalesced memory access, etc.
    // Observe the memory access pattern of each entry (thread) to explore potential optimization

    // OneThreadPerEntry
    for (int len = 2; len <= N; ++len) {
        int num_blocks = (N - len + thread_per_block) / thread_per_block;
        oneThreadPerEntry<<<num_blocks, thread_per_block>>>(dp_device, cut_device, len, N);
    }

    cudaMemcpy(dp_host, dp_device, (N + 1) * N * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(cut_host, cut_device, (N + 1) * N * sizeof(long long), cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    execution_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << execution_time << " ms" << std::endl;
    std::cout << "Minimum number of multiplications: " << dp_host[convertIdx(N, 0, N)] << std::endl;

    // free the memory spaces that are allocated in host and device
    // cudaFree(p_device);
    cudaFree(dp_device);
    cudaFree(cut_device);

    cudaFreeHost(p_host);
    cudaFreeHost(dp_host);
    cudaFreeHost(cut_host);
}