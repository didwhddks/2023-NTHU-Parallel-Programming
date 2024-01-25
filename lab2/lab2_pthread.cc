#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long r, k, p, m;
unsigned long long r_squared;

void* calc(void* id) {
	unsigned long long i = (unsigned long long) id;
	unsigned long long left = i < m ? i * (p + 1) : m * (p + 1) + (i - m) * p;
	unsigned long long right = left + p - (i < m ? 0 : 1);
	unsigned long long res = 0;

	for (unsigned long long x = left; x <= right; ++x) {
		unsigned long long y = ceil(sqrtl(r_squared - x * x));
		res += y;
		// if (res >= k) {
		// 	res -= k;
		// }
	}
	res %= k;

	pthread_exit((void*) res);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long NUM_THREADS = CPU_COUNT(&cpuset);

	r = atoll(argv[1]);
	k = atoll(argv[2]);
	p = r / NUM_THREADS;
	m = r % NUM_THREADS;
	r_squared = r * r;

	unsigned long long pixels = 0;

	pthread_t threads[NUM_THREADS];

	for (unsigned long long i = 0; i < NUM_THREADS; ++i) {
		pthread_create(&threads[i], NULL, calc, (void*) i);
	}

	for (unsigned long long i = 0; i < NUM_THREADS; ++i) {
		void* res;
		pthread_join(threads[i], &res);
		pixels += (unsigned long long) res;
		// if (pixels >= k) {
		// 	pixels -= k;
		// }
	}
	pixels %= k;

	printf("%llu\n", (4 * pixels) % k);
	pthread_exit(NULL);
}
