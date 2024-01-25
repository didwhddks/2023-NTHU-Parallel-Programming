#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long mod = r % size;
	unsigned long long p = r / size;
	unsigned long long left = rank < mod ? rank * (p + 1) : mod * (p + 1) + (rank - mod) * p;
	unsigned long long right = rank < mod ? left + p : left + p - 1;
	unsigned long long x;
	unsigned long long r_squared = r * r;

	const int chunk_size = 10000;

	#pragma omp parallel shared(left, right, k, r_squared) private(x)
	{
		#pragma omp for schedule(dynamic, chunk_size) reduction(+: pixels) nowait
		for (x = left; x <= right; ++x) {
			unsigned long long y = ceil(sqrtl(r_squared - x * x));
			pixels += y;
			// pixels -= pixels >= k ? k : 0;
		}
		pixels %= k;
	}

	unsigned long long ans;

	MPI_Reduce(&pixels, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		ans %= k;
		printf("%llu\n", (4 * ans) % k);
	}

	MPI_Finalize();
}