#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

int main(int argc, char *argv[]) {

    double execution_time = 0.0;
    auto start = std::chrono::steady_clock::now();

    int N;
    FILE *input_file = fopen("testcase", "r");
    fread(&N, sizeof(int), 1, input_file);

    std::vector<int> p(N + 1);
    fread(&p[0], sizeof(int), N + 1, input_file);
    fclose(input_file);

    const long long INF = 2E18;

    std::vector<std::vector<long long>> dp(N, std::vector<long long>(N, INF));
    std::vector<std::vector<int>> cut(N, std::vector<int>(N, -1));

    for (int i = 0; i < N; ++i) {
        dp[i][i] = 0;
    }

    for (int len = 2; len <= N; ++len) {
        for (int i = 0; i <= N - len; ++i) {
            int j = i + len - 1;
            for (int k = i; k < j; ++k) {
                long long cost = dp[i][k] + dp[k + 1][j] + 1LL * p[i] * p[k + 1] * p[j + 1];
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                    cut[i][j] = k;
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    execution_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << execution_time << " ms\n";
    std::cout << "Minimum number of multiplications: " << dp[0][N - 1] << "\n";
}