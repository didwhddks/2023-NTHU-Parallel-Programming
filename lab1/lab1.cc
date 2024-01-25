#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		unsigned long long ans = 0;
		for (unsigned long long x = 0; x < r / size; ++x) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			ans = (ans + y) % k;
		}
		for (int i = 1; i < size; ++i) {
			unsigned long long y;
			MPI_Recv(&y, 1, MPI_UNSIGNED_LONG_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			ans = (ans + y) % k;
		}
		pixels = 4 * ans % k;
		printf("%llu\n", pixels);
	} else {
		unsigned long long ans = 0;
		for (unsigned long long x = rank * r / size; x < (rank + 1) * r / size; ++x) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			ans = (ans + y) % k;
		}
		MPI_Send(&ans, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}
