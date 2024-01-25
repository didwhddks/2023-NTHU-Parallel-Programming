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

__host__ __device__ int convertIdx(int i, int j, int N) {
    return i * N + j;
}

void input() {
    FILE *input_file = fopen("testcase", "r");
    fread(&N, sizeof(int), 1, input_file);

    // malloc host memory
    cudaMallocHost((void **) &p_host, (N + 1) * sizeof(int));
    cudaMallocHost((void **) &dp_host, N * N * sizeof(long long));
    cudaMallocHost((void **) &cut_host, N * N * sizeof(long long));

    // read dimension of each matrix to p_host
    fread(p_host, sizeof(int), N + 1, input_file);
    fclose(input_file);

    // initialize dp_host & cut_host
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            dp_host[convertIdx(i, j, N)] = i == j ? 0 : INF;
            cut_host[convertIdx(i, j, N)] = -1;
        }
    }
}

__global__ void oneThreadPerEntry(long long *dp_device, long long *cut_device, int len, int N) {
    int i = blockIdx.x * thread_per_block + threadIdx.x;
    int j = i + len - 1;

    if (i >= N || j >= N) {
        return;
    }

    for (int k = i; k < j; ++k) {
        long long cost = dp_device[convertIdx(i, k, N)] + dp_device[convertIdx(k + 1, j, N)] +
                         1LL * p_constant[i] * p_constant[k + 1] * p_constant[j + 1];
        if (cost < dp_device[convertIdx(i, j, N)]) {
            dp_device[convertIdx(i, j, N)] = cost;
            cut_device[convertIdx(i, j, N)] = k;
        }
    }
}

int main() {

    double execution_time = 0.0;
    auto start = std::chrono::steady_clock::now();

    cudaSetDevice(0);
    input();

    long long *dp_device, *cut_device;
    // int *p_device;
    // cudaMalloc((void **) &p_device, (N + 1) * sizeof(int));
    cudaMalloc((void **) &dp_device, N * N * sizeof(long long));
    cudaMalloc((void **) &cut_device, N * N * sizeof(long long));

    // cudaMemcpy(p_device, p_host, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(p_constant, p_host, (N + 1) * sizeof(int));
    cudaMemcpy(dp_device, dp_host, N * N * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(cut_device, cut_host, N * N * sizeof(long long), cudaMemcpyHostToDevice);

    // TODO: implement the parallel version of matrix chain multiplication here
    // Can try OneThreadPerEntry, OneBlockPerEntry, MultipleBlocksPerEntry
    // Can try shared memory, coalesced memory access, etc.
    // Observe the memory access pattern of each entry (thread) to explore potential optimization

    // OneThreadPerEntry
    for (int len = 2; len <= N; ++len) {
        int num_blocks = (N - len + thread_per_block) / thread_per_block;
        oneThreadPerEntry<<<num_blocks, thread_per_block>>>(dp_device, cut_device, len, N);
    }

    cudaMemcpy(dp_host, dp_device, N * N * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(cut_host, cut_device, N * N * sizeof(long long), cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    execution_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << execution_time << " ms" << std::endl;
    std::cout << "Minimum number of multiplications: " << dp_host[convertIdx(0, N - 1, N)] << std::endl;

    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << dp_host[convertIdx(i, j, N)] << " \n"[j == N - 1];
    //     }
    // }

    // free the memory spaces that are allocated in host and device
    // cudaFree(p_device);
    cudaFree(dp_device);
    cudaFree(cut_device);

    cudaFreeHost(p_host);
    cudaFreeHost(dp_host);
    cudaFreeHost(cut_host);
}