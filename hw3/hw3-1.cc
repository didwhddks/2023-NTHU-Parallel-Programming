#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <sched.h>
#include <omp.h>

const int maxV = 6000;
const int INF = (1 << 30) - 1;
const int chunk_size = 1;

int dist[maxV][maxV];

int main(int argc, char *argv[]) {
    if (argc != 3) {
        return 0;
    }
    char *input_filename = argv[1];
    char *output_filename = argv[2];

    FILE *input_file = fopen(input_filename, "rb");
    FILE *output_file = fopen(output_filename, "wb");

    int V, E;
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);

    // Get the number of CPU cores available
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    const int thread_num = CPU_COUNT(&cpu_set);

    #pragma omp parallel for schedule(guided, chunk_size) num_threads(thread_num) collapse(2)
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i == j) {
                dist[i][j] = 0;
            } else {
                dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, input_file);
        dist[pair[0]][pair[1]] = pair[2];
    }

    for (int k = 0; k < V; ++k) {
        #pragma omp parallel for schedule(guided, chunk_size) collapse(2) num_threads(thread_num)
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    // Output the distance
    for (int i = 0; i < V; ++i) {
        fwrite(dist[i], sizeof(int), V, output_file);
    }

    fclose(input_file);
    fclose(output_file);

    return 0;
}