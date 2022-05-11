#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int isPrime(unsigned long x) {
    for(unsigned long i = 2; i * i <= x; i++) {
        if(x % i == 0) {
			return 0;
		}  
    }
    return 1;
}

unsigned long countPrimes(char* tab, unsigned long n, unsigned long m) {
	unsigned long counter = 0;
	double t1 = omp_get_wtime();

	for(unsigned long i = m; i <= n ; i++) {
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

void runCode(unsigned long n, unsigned long m) {
	char *tab=(char*)malloc(n - m + 1);
	memset(tab, '1', n - m + 1);
	
	unsigned long counter = countPrimes(tab, n, m);
    
	printf("Primes between <%ld,%ld>: %ld\n", m, n, counter);
	free(tab);
}

int main(int argc, char* argv[]) {
	
	if(argc != 3) {
		return 0;
	}
	unsigned long m,n;
	m = atoll(argv[1]);
	n = atoll(argv[2]);
	
	runCode(n, m);
	
    return 0;
}