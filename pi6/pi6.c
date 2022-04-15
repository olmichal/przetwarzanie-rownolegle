#include <stdio.h>
#include <time.h>
#include <omp.h>

long long num_steps = 1000000000;
double step;

int threads[3] = {16,8,4};
int max_threads = 16;

int main(){
	printf("------------------------------------------------\n");
    clock_t ppstart, ppstop;
    double pswtime, pewtime;
	double fractional_sums[max_threads];
    int i;

    step = 1./(double)num_steps;

	for(i = 0; i < 3; i++) {

		double x, pi, sum = 0.0;

		for(int j = 0; j < max_threads; j++) {
			fractional_sums[j] = 0.0;
		}

		omp_set_num_threads(threads[i]);
		pswtime = omp_get_wtime();
        ppstart = clock();

		#pragma omp parallel private(x) 
        {
                int id = omp_get_thread_num();

                #pragma omp for
                for (i = 0; i < num_steps; i++) {
                        x = (i + .5) * step;
                        
                        fractional_sums[id] +=  (4.0 / (1. + x * x));
                }
        }
        for(int j = 0; j < threads[i]; j++) {
            sum += fractional_sums[j];
        }
        pi = sum * step;
        
        pewtime = omp_get_wtime();
		ppstop = clock();

		printf("%15.12f Wartosc liczby PI rownolegle %d\n",pi, threads[i]);
        printf("Czas procesorów przetwarzania równoleglego  %f sekund \n", ((double)(ppstop - ppstart)/CLOCKS_PER_SEC));
        printf("Czas trwania obliczen rownoleglych - wallclock %f sekund \n", pewtime-pswtime);
		printf("------------------------------------------------\n");
	}
        
    return 0;
}

// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle 16
// Czas procesorów przetwarzania równoleglego  1.974791 sekund 
// Czas trwania obliczen rownoleglych - wallclock 0.126623 sekund 
// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle 8
// Czas procesorów przetwarzania równoleglego  1.151469 sekund 
// Czas trwania obliczen rownoleglych - wallclock 0.195453 sekund 
// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle 4
// Czas procesorów przetwarzania równoleglego  0.979279 sekund 
// Czas trwania obliczen rownoleglych - wallclock 0.245385 sekund 
// ------------------------------------------------
