#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const int INF = (1 << 30) - 1;
const int B = 64;
int V, E, nV;
int *dist;

__constant__ int nV_d;

int ceil(int a, int b) {
    return (a + b - 1) / b;
}

void input(char *input_filename) {
    FILE* input_file = fopen(input_filename, "rb");
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);

    // printf("V = %d, E = %d\n", V, E);

    nV = ceil(V, B) * B;
    cudaMallocHost((void **)&dist, nV * nV * sizeof(int));

    for (int i = 0; i < nV; ++i) {
        for (int j = 0; j < nV; ++j) {
            dist[i * nV + j] = i == j && i < V ? 0 : INF;
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, input_file);
        dist[pair[0] * nV + pair[1]] = pair[2];
    }
    fclose(input_file);
}

void output(char* output_filename) {
    FILE* output_file = fopen(output_filename, "w");
    for (int i = 0; i < V; ++i) {
        fwrite(dist + i * nV, sizeof(int), V, output_file);
    }
    fclose(output_file);
}

//======================
__global__ void phase1(int *dist, int r) {
    int x = threadIdx.y, y = threadIdx.x;
    int i = x + (r << 6), j = y + (r << 6);

    __shared__ int dist_shared[4096];
    dist_shared[(x << 6) + y] = dist[i * nV_d + j];
    dist_shared[(x << 6) + y + 32] = dist[i * nV_d + j + 32];
    dist_shared[((x + 32) << 6) + y] = dist[(i + 32) * nV_d + j];
    dist_shared[((x + 32) << 6) + y + 32] = dist[(i + 32) * nV_d + j + 32];

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        __syncthreads();
        dist_shared[(x << 6) + y] = min(dist_shared[(x << 6) + y], dist_shared[(x << 6) + k] + dist_shared[(k << 6) + y]);
        dist_shared[(x << 6) + y + 32] = min(dist_shared[(x << 6) + y + 32], dist_shared[(x << 6) + k] + dist_shared[(k << 6) + y + 32]);
        dist_shared[((x + 32) << 6) + y] = min(dist_shared[((x + 32) << 6) + y], dist_shared[((x + 32) << 6) + k] + dist_shared[(k << 6) + y]);
        dist_shared[((x + 32) << 6) + y + 32] = min(dist_shared[((x + 32) << 6) + y + 32], dist_shared[((x + 32) << 6) + k] + dist_shared[(k << 6) + y + 32]);
    }

    dist[i * nV_d + j] = dist_shared[(x << 6) + y];
    dist[i * nV_d + j + 32] = dist_shared[(x << 6) + y + 32];
    dist[(i + 32) * nV_d + j] = dist_shared[((x + 32) << 6) + y];
    dist[(i + 32) * nV_d + j + 32] = dist_shared[((x + 32) << 6) + y + 32];
}

__global__ void phase2(int *dist, int r) {
    int bx = blockIdx.x, by = blockIdx.y >= r ? blockIdx.y + 1 : blockIdx.y;
    int tx = threadIdx.y, ty = threadIdx.x;

    int i = bx ? tx + (by << 6) : tx + (r << 6);
    int j = bx ? ty + (r << 6) : ty + (by << 6);
    int pivot_i = tx + (r << 6);
    int pivot_j = ty + (r << 6);

    __shared__ int dist_shared[8192];
    // First row => current block
    dist_shared[(tx << 6) + ty] = dist[i * nV_d + j];
    dist_shared[(tx << 6) + ty + 32] = dist[i * nV_d + j + 32];
    dist_shared[((tx + 32) << 6) + ty] = dist[(i + 32) * nV_d + j];
    dist_shared[((tx + 32) << 6) + ty + 32] = dist[(i + 32) * nV_d + j + 32];

    int v1 = dist_shared[(tx << 6) + ty];
    int v2 = dist_shared[(tx << 6) + ty + 32];
    int v3 = dist_shared[((tx + 32) << 6) + ty];
    int v4 = dist_shared[((tx + 32) << 6) + ty + 32];
    int offset = 4096;
    
    // Second row => pivot block
    dist_shared[(tx << 6) + ty + offset] = dist[pivot_i * nV_d + pivot_j];
    dist_shared[(tx << 6) + ty + 32 + offset] = dist[pivot_i * nV_d + pivot_j + 32];
    dist_shared[((tx + 32) << 6) + ty + offset] = dist[(pivot_i + 32) * nV_d + pivot_j];
    dist_shared[((tx + 32) << 6) + ty + 32 + offset] = dist[(pivot_i + 32) * nV_d + pivot_j + 32];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        v1 = min(v1, dist_shared[(1 - bx) * offset + (tx << 6) + k] + dist_shared[bx * offset + (k << 6) + ty]);
        v2 = min(v2, dist_shared[(1 - bx) * offset + (tx << 6) + k] + dist_shared[bx * offset + (k << 6) + ty + 32]);
        v3 = min(v3, dist_shared[(1 - bx) * offset + ((tx + 32) << 6) + k] + dist_shared[bx * offset + (k << 6) + ty]);
        v4 = min(v4, dist_shared[(1 - bx) * offset + ((tx + 32) << 6) + k] + dist_shared[bx * offset + (k << 6) + ty + 32]);
    }

    dist[i * nV_d + j] = v1;
    dist[i * nV_d + j + 32] = v2;
    dist[(i + 32) * nV_d + j] = v3;
    dist[(i + 32) * nV_d + j + 32] = v4;
}

__global__ void phase3(int *dist, int r) {
    int bx = blockIdx.x >= r ? blockIdx.x + 1 : blockIdx.x;
    int by = blockIdx.y >= r ? blockIdx.y + 1 : blockIdx.y;

    int tx = threadIdx.y, ty = threadIdx.x;
    int i = tx + (bx << 6), j = ty + (r << 6);
    int a = tx + (r << 6), b = ty + (by << 6);
    int c = tx + (bx << 6), d = ty + (by << 6);

    __shared__ int dist_shared[8192];
    // current block
    int v1 = dist[c * nV_d + d];
    int v2 = dist[c * nV_d + d + 32];
    int v3 = dist[(c + 32) * nV_d + d];
    int v4 = dist[(c + 32) * nV_d + d + 32];
    int offset = 4096;

    // First row => pivot column
    dist_shared[(tx << 6) + ty] = dist[i * nV_d + j];
    dist_shared[(tx << 6) + ty + 32] = dist[i * nV_d + j + 32];
    dist_shared[((tx + 32) << 6) + ty] = dist[(i + 32) * nV_d + j];
    dist_shared[((tx + 32) << 6) + ty + 32] = dist[(i + 32) * nV_d + j + 32];

    // Second row => pivot row
    dist_shared[(tx << 6) + ty + offset] = dist[a * nV_d + b];
    dist_shared[(tx << 6) + ty + 32 + offset] = dist[a * nV_d + b + 32];
    dist_shared[((tx + 32) << 6) + ty + offset] = dist[(a + 32) * nV_d + b];
    dist_shared[((tx + 32) << 6) + ty + 32 + offset] = dist[(a + 32) * nV_d + b + 32];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        v1 = min(v1, dist_shared[(tx << 6) + k] + dist_shared[(k << 6) + ty + offset]);
        v2 = min(v2, dist_shared[(tx << 6) + k] + dist_shared[(k << 6) + ty + 32 + offset]);
        v3 = min(v3, dist_shared[((tx + 32) << 6) + k] + dist_shared[(k << 6) + ty + offset]);
        v4 = min(v4, dist_shared[((tx + 32) << 6) + k] + dist_shared[(k << 6) + ty + 32 + offset]);
    }

    dist[c * nV_d + d] = v1;
    dist[c * nV_d + d + 32] = v2;
    dist[(c + 32) * nV_d + d] = v3;
    dist[(c + 32) * nV_d + d + 32] = v4;
}

int main(int argc, char *argv[]) {

    input(argv[1]);

    int *dist_d;
    cudaMalloc((void **)&dist_d, nV * nV * sizeof(int));
    cudaMemcpy(dist_d, dist, nV * nV * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(nV_d, &nV, sizeof(int));

    // also the number of rounds
    const int N = nV / B;

    dim3 grid1(2, N - 1);
    dim3 grid2(N - 1, N - 1);
    dim3 block(32, 32);

    for (int r = 0; r < N; ++r) {
        phase1<<<1, block>>>(dist_d, r);
        phase2<<<grid1, block>>>(dist_d, r);
        phase3<<<grid2, block>>>(dist_d, r);
    }

    cudaMemcpy(dist, dist_d, nV * nV * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);

    cudaFree(dist_d);
    cudaFreeHost(dist);

    return 0;
}