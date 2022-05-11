#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

const int Number = 1000000001;
bool primes[Number];

void findPrimes(bool *primes, int m, int n) {
	for (int i = 2; i*i <= n; i++) {
		if (primes[i] == 0) {
			for (int j = i*i; j <= n; j+=i) {
				primes[j] = 1;
			}
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

void sequential_sieve(unsigned long m, unsigned long n) {
	double t1 = omp_get_wtime();
	findPrimes(primes, m, n);
	double t2 = omp_get_wtime();

	unsigned long counter = countPrimes(primes, m, n);

	printf("%.6lf\n", t2 - t1);
	printf("Primes between <%ld,%ld>: %lu\n", m, n, counter);
}

int main(int argc, char *argv[]) {

	if(argc != 3) {
		return 0;
	}
	unsigned long m,n;
	m = atoi(argv[1]);
	n = atoi(argv[2]);

	sequential_sieve(m, n);

	return 0;
}