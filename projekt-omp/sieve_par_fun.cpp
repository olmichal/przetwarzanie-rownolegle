#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

const int Number = 1000000001;
bool primes[Number];

inline int max(int a, int b) {
	if(a > b) {
		return a;
	} else {
		return b;
	}
}

inline int findProduct(int a, int b) {
	int res = 0;
	if (a%b != 0) {
		res = b;
	}
	return a - a%b + res;
}

void findPrimes(bool *primes, int m, int n, int threads) {
	int sqrtN = floor(sqrt(n));

	if (n > 4) {
		findPrimes(primes, 2, sqrtN, threads);
	} else {
		primes[0] = primes[1] = 1;
		if (n == 4) {
			primes[4] = 1;
		}
		return;
	}

	omp_set_num_threads(threads);
	#pragma omp parallel for
	for (int i = 2; i <= sqrtN; i++) {
		if (primes[i] == 1) {
			continue;
		}

		int j = max(findProduct(sqrtN + 1, i), findProduct(m, i));
		for (j = max(i*i, j); j <= n; j += i) {
			primes[j] = 1;
		}
	}
}

unsigned long countPrimes(bool *primes, unsigned long m, unsigned long n) {
	int counter = 0;
	for (unsigned long i = m; i <= n; i++) {
		if (primes[i] == 0) {
			counter++;
		}
	}
	return counter;
}

void parallel_functional_sieve(unsigned long m, unsigned long n, int threads) {
	double t1 = omp_get_wtime();
	findPrimes(primes, m, n, threads);
	double t2 = omp_get_wtime();
	unsigned long counter = countPrimes(primes, m, n);

	printf("%.6lf\n", t2 - t1);
	printf("Primes between <%ld,%ld>: %ld\n", m, n, counter);
}

int main(int argc, char *argv[]) {

	if(argc != 4) {
		return 0;
	}
	unsigned long m,n;
	int threads;
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	threads = atoi(argv[3]);
	
	parallel_functional_sieve(m, n, threads);

	return 0;
}