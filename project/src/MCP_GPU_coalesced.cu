#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <chrono>

#define thread_per_block 256
#define i64 long long

const i64 inf = 2E18;
int N;
i64 *dp_host, *cut_host;
int *p_host;

__host__ __device__ int convertIdx(int i, int j, int N) {
    return i * N + j;
}

__host__ int ceil(int a, int b) {
    return (a + b - 1) / b;
}

__host__ void input() {
    FILE *input_file = fopen("testcase", "r");
    if (fread(&N, sizeof(int), 1, input_file) != 1) {
        std::cerr << "Error reading from file" << std::endl;
        exit(1);
    }

    std::cout << "Number of matrices: " << N << std::endl;

    // allocate host memory for p_host, dp_host, cut_host
    cudaMallocHost((void **) &p_host, (N + 1) * sizeof(int));
    cudaMallocHost((void **) &dp_host, N * N * sizeof(i64));
    cudaMallocHost((void **) &cut_host, N * N * sizeof(i64));

    // read dimensions of matrices
    if (fread(p_host, sizeof(int), N + 1, input_file) != N + 1) {
        std::cerr << "Error reading from file" << std::endl;
        exit(1);
    }
    fclose(input_file);

    // for (int i = 0; i <= N; ++i) {
    //     std::cout << "p[" << i << "] = " << p_host[i] << std::endl;
    // }

    // initialize dp_host and cut_host
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            int idx = convertIdx(j - i, i, N);
            dp_host[idx] = i == j ? 0 : inf;
            cut_host[idx] = -1;
        }
    }
}

__global__ void oneThreadPerEntry(i64 *dp_device, i64 *cut_device, int *p_device, int len, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i + len;

    if (i < N && j < N) {
        for (int k = i; k < j; ++k) {
            i64 cost = dp_device[convertIdx(k - i, i, N)] + dp_device[convertIdx(j - k - 1, k + 1, N)] +
                            1LL * p_device[i] * p_device[k + 1] * p_device[j + 1];
            if (cost < dp_device[convertIdx(len, i, N)]) {
                dp_device[convertIdx(len, i, N)] = cost;
                cut_device[convertIdx(len, i, N)] = k;
            }
        }
    }
}

int main() {
    cudaSetDevice(0);
    input();

    i64 *dp_device, *cut_device;
    int *p_device;
    cudaMalloc((void **) &p_device, (N + 1) * sizeof(int));
    cudaMalloc((void **) &dp_device, N * N * sizeof(i64));
    cudaMalloc((void **) &cut_device, N * N * sizeof(i64));

    cudaMemcpy(p_device, p_host, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dp_device, dp_host, N * N * sizeof(i64), cudaMemcpyHostToDevice);
    cudaMemcpy(cut_device, cut_host, N * N * sizeof(i64), cudaMemcpyHostToDevice);

    // CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start, 0);

    // Launch kernel for each length
    for (int len = 1; len < N; ++len) {
        int num_blocks = ceil(N - len, thread_per_block);
        // std::cout << "Launching kernel for length " << len << " with " << num_blocks << " blocks" << std::endl;
        oneThreadPerEntry<<<num_blocks, thread_per_block>>>(dp_device, cut_device, p_device, len, N);
        // Check if the kernel launch was successful
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Record stop time
    cudaEventRecord(stop, 0);

    // Wait for all GPU tasks to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Copy results back to host
    cudaMemcpy(dp_host, dp_device, N * N * sizeof(i64), cudaMemcpyDeviceToHost);
    cudaMemcpy(cut_host, cut_device, N * N * sizeof(i64), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < N; ++i) {
    //     for (int j = i; j < N; ++j) {
    //         int idx = convertIdx(i, j, N);
    //         std::cout << "dp[" << i << "][" << j << "] = " << dp_host[idx] << ", cut = " << cut_host[idx] << std::endl;
    //     }
    // }

    // Print results
    std::cout << "Execution time: " << elapsed_time << " ms" << std::endl;
    std::cout << "Minimum number of multiplications: " << dp_host[convertIdx(N - 1, 0, N)] << std::endl;

    // Cleanup
    cudaFree(p_device);
    cudaFree(dp_device);
    cudaFree(cut_device);

    cudaFreeHost(p_host);
    cudaFreeHost(dp_host);
    cudaFreeHost(cut_host);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
