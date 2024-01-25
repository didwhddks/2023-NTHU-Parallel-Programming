#include <iostream>
#include <cmath>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <mpi.h>

float *temp;

void merge(float *arr, float *buf, int n, int m, int chk) {
    if (chk) {
        int ptr1 = n - 1, ptr2 = m - 1, idx = n - 1;
        while (~ptr1 && ~ptr2 && ~idx) {
            if (arr[ptr1] >= buf[ptr2]) {
                temp[idx--] = arr[ptr1--];
            } else {
                temp[idx--] = buf[ptr2--];
            }
        }
        while (~ptr1 && ~idx) {
            temp[idx--] = arr[ptr1--];
        }
        while (~ptr2 && ~idx) {
            temp[idx--] = buf[ptr2--];
        }
        for (int i = 0; i < n; ++i) {
            arr[i] = temp[i];
        }
    } else {
        int ptr1 = 0, ptr2 = 0, idx = 0;
        while (ptr1 < n && ptr2 < m && idx < n) {
            if (arr[ptr1] <= buf[ptr2]) {
                temp[idx++] = arr[ptr1++];
            } else {
                temp[idx++] = buf[ptr2++];
            }
        }
        while (ptr1 < n && idx < n) {
            temp[idx++] = arr[ptr1++];
        }
        while (ptr2 < m && idx < n) {
            temp[idx++] = buf[ptr2++];
        }
        for (int i = 0; i < n; ++i) {
            arr[i] = temp[i];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = atoi(argv[1]);
    int mod = n % size;
    int partition_size = n / size + (rank < mod ? 1 : 0);
    int start = (rank < mod ? rank : mod) * (n / size + 1) +
                (rank < mod ? 0 : rank - mod) * (n / size);

    int right_partition_size = rank + 1 < size ? (rank + 1 == mod ? partition_size - 1 : partition_size) : 0;
    int left_partition_size = rank - 1 >= 0 ? (rank == mod ? partition_size + 1 : partition_size) : 0;

    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_File input_file, output_file;
    float *arr = (float*) malloc(partition_size * sizeof(float));
    float *buf = (float*) malloc(std::max(left_partition_size, 
                    right_partition_size) * sizeof(float));
    temp = (float*) malloc(std::max(partition_size, left_partition_size) * sizeof(float));

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * start, arr, partition_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    if (partition_size) {
        boost::sort::spreadsort::spreadsort(arr, arr + partition_size);
    }

    float x;

    for (int even_phase = 1, time = 0; time < size + 1; ++time, even_phase ^= 1) {
        if (even_phase) {
            if (rank & 1 && partition_size) {
                MPI_Sendrecv(&arr[0], 1, MPI_FLOAT, rank - 1, 0, &x, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (x > arr[0]) {
                    int lb = std::lower_bound(arr, arr + partition_size, x) - arr;
                    int need;
                    MPI_Sendrecv(&lb, 1, MPI_INT, rank - 1, 0, &need, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Sendrecv(arr, lb, MPI_FLOAT, rank - 1, 0, buf, need, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge(arr, buf, partition_size, need, 1);
                }
            } else if (rank & 1 ^ 1 && rank + 1 < size && right_partition_size) {
                MPI_Sendrecv(&arr[partition_size - 1], 1, MPI_FLOAT, rank + 1, 0, &x, 1, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (arr[partition_size - 1] > x) {
                    int ub = std::upper_bound(arr, arr + partition_size, x) - arr;
                    int tot = partition_size - ub;
                    int need;
                    MPI_Sendrecv(&tot, 1, MPI_INT, rank + 1, 0, &need, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Sendrecv(arr + ub, tot, MPI_FLOAT, rank + 1, 0, buf, need, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge(arr, buf, partition_size, need, 0);
                }
            }
        } else {
            if (rank & 1 && rank + 1 < size && right_partition_size) {
                MPI_Sendrecv(&arr[partition_size - 1], 1, MPI_FLOAT, rank + 1, 0, &x, 1, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (arr[partition_size - 1] > x) {
                    int ub = std::upper_bound(arr, arr + partition_size, x) - arr;
                    int tot = partition_size - ub;
                    int need;
                    MPI_Sendrecv(&tot, 1, MPI_INT, rank + 1, 0, &need, 1, MPI_INT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Sendrecv(arr + ub, tot, MPI_FLOAT, rank + 1, 0, buf, need, MPI_FLOAT, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge(arr, buf, partition_size, need, 0);
                }
            } else if (rank & 1 ^ 1 && rank && partition_size) {
                MPI_Sendrecv(&arr[0], 1, MPI_FLOAT, rank - 1, 0, &x, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (x > arr[0]) {
                    int lb = std::lower_bound(arr, arr + partition_size, x) - arr;
                    int need;
                    MPI_Sendrecv(&lb, 1, MPI_INT, rank - 1, 0, &need, 1, MPI_INT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Sendrecv(arr, lb, MPI_FLOAT, rank - 1, 0, buf, need, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge(arr, buf, partition_size, need, 1);
                }
            }
        }
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * start, arr, partition_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    free(arr);
    free(buf);
    free(temp);

    MPI_Finalize();
    
    return 0;
}
