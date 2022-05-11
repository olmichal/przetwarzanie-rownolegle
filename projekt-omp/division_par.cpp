#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

int isPrime(unsigned long x) {
    for(unsigned long i = 2; i * i <= x; i++) {
        if(x % i == 0) {
			return 0;
		}
    }
    return 1;
}

unsigned long countPrimes(char* tab, unsigned long n, unsigned long m, int threads) {
	unsigned long counter = 0;
	double t1 = omp_get_wtime();

	omp_set_num_threads(threads);
	#pragma omp parallel for reduction (+:counter)
		for(unsigned long i = m; i <= n; i++) {
        	if(!isPrime(i)) {
				tab[i-m] = '0';
			} else {
				counter++;
			}
    	}

	double t2 = omp_get_wtime();
	printf("%.6lf\n", t2 - t1);
	return counter;
}

void runCode(unsigned long n, unsigned long m, int threads) {
	char *tab=(char*)malloc(n - m + 1);
	memset(tab, '1', n - m + 1);
	
	unsigned long counter = countPrimes(tab, n, m, threads);

	printf("Primes between <%ld,%ld>: %ld\n", m, n, counter);
	free(tab);
}

int main(int argc, char* argv[]) {
	
	if(argc != 4) {
		return 0;
	}
	unsigned long m,n;
	int threads;
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	threads = atoi(argv[3]);

	runCode(n, m, threads);
	
    return 0;
}