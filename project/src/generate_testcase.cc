#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

int main(int argc, char *argv[]) {

    int N = atoi(argv[1]);
    int maxD = atoi(argv[2]);

    FILE *output_file = fopen("testcase", "w");
    fwrite(&N, sizeof(int), 1, output_file);

    for (int i = 0; i <= N; ++i) {
        int p = (rand() % maxD) + 1;
        fwrite(&p, sizeof(int), 1, output_file);
    }

    fclose(output_file);
}