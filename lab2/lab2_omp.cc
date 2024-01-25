#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long x;
	unsigned long long r_squared = r * r;

	const int chunk_size = 10000;

	#pragma omp parallel shared(k, r_squared) private(x)
	{
		#pragma omp for schedule(dynamic, chunk_size) reduction(+: pixels) nowait
		for (x = 0; x < r; ++x) {
			unsigned long long y = ceil(sqrtl(r_squared - x * x));
			pixels += y;
			// pixels -= pixels >= k ? k : 0;
		}
		pixels %= k;
	}

	pixels %= k;

	printf("%llu\n", (4 * pixels) % k);
}