#include <stdio.h>
#include <omp.h>

long long num_steps = 1000000000;
double step;

int threads[3] = {16,8,1};

int main() {
    printf("------------------------------------------------\n");
    double pswtime, pewtime;
	int i;

    step = 1./(double)num_steps;

    for(i = 0; i < 3; i++) {
        double sum=0.0;
        omp_set_num_threads(threads[i]);
        pswtime = omp_get_wtime();

        #pragma omp parallel for reduction(+: sum)
        for(i = 0; i < num_steps; i++) {
            double x = (i + .5)*step;
            sum = sum + 4.0/(1.+ x*x);
        }
        double pi = sum*step;

        pewtime = omp_get_wtime();
        printf("%15.12f Wartosc liczby PI rownolegle dla %d wątków\n",pi, threads[i]);
        printf("Czas trwania obliczen rownoleglych dla %d wątków - wallclock %f sekund \n", threads[i], pewtime-pswtime);
        printf("------------------------------------------------\n");
    }
    return 0;
}