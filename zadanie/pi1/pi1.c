#include <stdio.h>
#include <time.h>
#include <omp.h>

long long num_steps = 1000000000;
double step;

int main()
{
	clock_t spstart, spstop;
	double sswtime, sewtime;
	double x, pi, sum=0.0;
	int i;

	sswtime = omp_get_wtime();
	spstart = clock();

	step = 1./(double)num_steps;

	for (i=0; i<num_steps; i++) {
		x = (i + .5)*step;
		sum = sum + 4.0/(1.+ x*x);
	}
	
	pi = sum*step;

	spstop = clock();
    sewtime = omp_get_wtime();
    printf("------------------------------------------------\n");
    printf("%15.12f wartosc liczby PI sekwencyjnie \n", pi);
    printf("------------------------------------------------\n");
    printf("Czas procesorów przetwarzania sekwencyjnego  %f sekund \n", ((double)(spstop - spstart)/CLOCKS_PER_SEC));
    printf("------------------------------------------------\n");
    printf("Czas trwania obliczen sekwencyjnych - wallclock %f sekund \n",  sewtime-sswtime);
    printf("------------------------------------------------\n");
    return 0;
}

// ------------------------------------------------
//  3.141592653590 wartosc liczby PI sekwencyjnie 
// ------------------------------------------------
// Czas procesorów przetwarzania sekwencyjnego  0.944504 sekund 
// ------------------------------------------------
// Czas trwania obliczen sekwencyjnych - wallclock 0.944600 sekund 
// ------------------------------------------------