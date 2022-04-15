#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

long long num_steps = 1000000000;
double step;
const int experiments = 50;

int main() {
    clock_t ppstart, ppstop;
    double pswtime, pewtime, start_time, stop_time;
    double x, pi, sum = 0.0;
    int i,j;

    omp_set_num_threads(2);

    volatile double *sum_array;
    sum_array = malloc(sizeof(double) * experiments);

    ppstart = clock();
    pswtime = omp_get_wtime();
    
    step = 1./(double)num_steps;

    for (j = 0; j < experiments - 1; j++)
    {
        sum_array[j] = 0;
        sum_array[j+1] = 0;

        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            #pragma omp for
            for (i = 0; i < num_steps; i++)
            {
                double x = (i+.5) * step;
                sum_array[j+id] += 4.0 / (1.+x*x);
            }
        }
        stop_time = omp_get_wtime();
        pi = (sum_array[j] + sum_array[j+1]) * step;
        printf("Para: %d | Pi: %15.12f | Czas: %.4f\n", j, pi, stop_time - start_time);
    }

    ppstop = clock();
    pewtime = omp_get_wtime();

    printf("------------------------------------------------\n");
    printf("%15.12f Wartosc liczby PI rownolegle \n", pi);
    printf("Czas procesorów przetwarzania równoleglego  %f sekund \n", ((double)(ppstop - ppstart) / CLOCKS_PER_SEC));
    printf("Czas trwania obliczen rownoleglych - wallclock %f sekund \n", pewtime - pswtime);
    printf("------------------------------------------------\n");
    return 0;
}

// Para: 0 | Pi:  3.141592653590 | Czas: 9.7352
// Para: 1 | Pi:  3.141592653590 | Czas: 10.0346
// Para: 2 | Pi:  3.141592653590 | Czas: 10.1091
// Para: 3 | Pi:  3.141592653590 | Czas: 1.2576
// Para: 4 | Pi:  3.141592653590 | Czas: 9.7568
// Para: 5 | Pi:  3.141592653590 | Czas: 9.4945
// Para: 6 | Pi:  3.141592653590 | Czas: 8.7238
// Para: 7 | Pi:  3.141592653590 | Czas: 9.0793
// Para: 8 | Pi:  3.141592653590 | Czas: 10.3304
// Para: 9 | Pi:  3.141592653590 | Czas: 9.6248
// Para: 10 | Pi:  3.141592653590 | Czas: 8.9517
// Para: 11 | Pi:  3.141592653590 | Czas: 1.2834
// Para: 12 | Pi:  3.141592653590 | Czas: 9.8739
// Para: 13 | Pi:  3.141592653590 | Czas: 10.1439
// Para: 14 | Pi:  3.141592653590 | Czas: 10.7067
// Para: 15 | Pi:  3.141592653590 | Czas: 10.1674
// Para: 16 | Pi:  3.141592653590 | Czas: 10.5122
// Para: 17 | Pi:  3.141592653590 | Czas: 10.6863
// Para: 18 | Pi:  3.141592653590 | Czas: 10.3452
// Para: 19 | Pi:  3.141592653590 | Czas: 1.2650
// Para: 20 | Pi:  3.141592653590 | Czas: 9.4499
// Para: 21 | Pi:  3.141592653590 | Czas: 8.6889
// Para: 22 | Pi:  3.141592653590 | Czas: 9.7834
// Para: 23 | Pi:  3.141592653590 | Czas: 10.0449
// Para: 24 | Pi:  3.141592653590 | Czas: 10.6548
// Para: 25 | Pi:  3.141592653590 | Czas: 9.8561
// Para: 26 | Pi:  3.141592653590 | Czas: 10.1345
// Para: 27 | Pi:  3.141592653590 | Czas: 1.3045
// Para: 28 | Pi:  3.141592653590 | Czas: 10.5150
// Para: 29 | Pi:  3.141592653590 | Czas: 9.3684
// Para: 30 | Pi:  3.141592653590 | Czas: 9.7445
// Para: 31 | Pi:  3.141592653590 | Czas: 9.0054
// Para: 32 | Pi:  3.141592653590 | Czas: 9.5981
// Para: 33 | Pi:  3.141592653590 | Czas: 9.9683
// Para: 34 | Pi:  3.141592653590 | Czas: 10.2625
// Para: 35 | Pi:  3.141592653590 | Czas: 1.2537
// Para: 36 | Pi:  3.141592653590 | Czas: 11.1410
// Para: 37 | Pi:  3.141592653590 | Czas: 9.9988
// Para: 38 | Pi:  3.141592653590 | Czas: 9.6985
// Para: 39 | Pi:  3.141592653590 | Czas: 8.9607
// Para: 40 | Pi:  3.141592653590 | Czas: 9.9797
// Para: 41 | Pi:  3.141592653590 | Czas: 9.4954
// Para: 42 | Pi:  3.141592653590 | Czas: 9.6314
// Para: 43 | Pi:  3.141592653590 | Czas: 1.2557
// Para: 44 | Pi:  3.141592653590 | Czas: 9.7065
// Para: 45 | Pi:  3.141592653590 | Czas: 9.5191
// Para: 46 | Pi:  3.141592653590 | Czas: 9.4794
// Para: 47 | Pi:  3.141592653590 | Czas: 8.0071
// Para: 48 | Pi:  3.141592653590 | Czas: 7.2814
// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle 
// Czas procesorów przetwarzania równoleglego  841.763824 sekund 
// Czas trwania obliczen rownoleglych - wallclock 425.869907 sekund 
// ------------------------------------------------
