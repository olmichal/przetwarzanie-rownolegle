#include <stdio.h>
#include <omp.h>
#include <time.h>

long long num_steps = 1000000000;
double step;

int threads[3] = {16,8,4};

int main() {
    printf("------------------------------------------------\n");
    double pswtime, pewtime;
    clock_t ppstart, ppstop;
	int i;

    step = 1./(double)num_steps;

    for(i = 0; i < 3; i++) {
        double sum = 0.0;
        omp_set_num_threads(threads[i]);
        ppstart = clock();
        pswtime = omp_get_wtime();

        #pragma omp parallel
        {
            double thread_sum = 0;
            
            #pragma omp for
            for(i = 0; i < num_steps; i++) {
                double x = (i + .5)*step;
                thread_sum = thread_sum + 4.0/(1.+ x*x);
            }

            #pragma omp atomic
            sum = sum + thread_sum;
        }
        
        double pi = sum*step;

        ppstop = clock();
        pewtime = omp_get_wtime();
        printf("%15.12f Wartosc liczby PI rownolegle dla %d wątków\n",pi, threads[i]);
        printf("Czas procesorów przetwarzania równoleglego dla %d wątków - %f sekund \n", threads[i], ((double)(ppstop - ppstart)/CLOCKS_PER_SEC));
        printf("Czas trwania obliczen rownoleglych dla %d wątków - wallclock %f sekund \n", threads[i], pewtime-pswtime);
        printf("------------------------------------------------\n");
    }
    return 0;
}

// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle dla 16 wątków
// Czas procesorów przetwarzania równoleglego dla 16 wątków - 1.970853 sekund 
// Czas trwania obliczen rownoleglych dla 16 wątków - wallclock 0.129373 sekund 
// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle dla 8 wątków
// Czas procesorów przetwarzania równoleglego dla 8 wątków - 1.335895 sekund 
// Czas trwania obliczen rownoleglych dla 8 wątków - wallclock 0.206839 sekund 
// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle dla 4 wątków
// Czas procesorów przetwarzania równoleglego dla 4 wątków - 0.973428 sekund 
// Czas trwania obliczen rownoleglych dla 4 wątków - wallclock 0.244853 sekund 
// ------------------------------------------------

// Zmienne globalne: sum
// Zmienne lokalne: i, x, thread_sum

// Wyniki są poprawne. W tym wariancie ćwiczenia procesory logiczne modyfikują lokalną sumę, a dopiero na końcu dodają jej wartość do globalnej sumy przy użyciu dyrektywy atomic.
// Poprawa czasów spowodowana jest znacznym zmniejszeniem liczby blokowań.

// Przyspieszenie:
// Dla 16 wątków: 7.301368
// Dla 8 wątków:  4.566837
// Dla 4 wątków:  3.857824
