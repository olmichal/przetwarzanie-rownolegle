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
        double x, pi, sum=0.0;
        omp_set_num_threads(threads[i]);
        ppstart = clock();
        pswtime = omp_get_wtime();

        #pragma omp parallel for
        for(i = 0; i < num_steps; i++) {
            x = (i + .5)*step;
            sum = sum + 4.0/(1.+ x*x);
        }
        pi = sum*step;

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
//  0.214559085307 Wartosc liczby PI rownolegle dla 16 wątków
// Czas procesorów przetwarzania równoleglego dla 16 wątków - 1.939707 sekund 
// Czas trwania obliczen rownoleglych dla 16 wątków - wallclock 0.124455 sekund 
// ------------------------------------------------
//  0.455168028575 Wartosc liczby PI rownolegle dla 8 wątków
// Czas procesorów przetwarzania równoleglego dla 8 wątków - 1.383529 sekund 
// Czas trwania obliczen rownoleglych dla 8 wątków - wallclock 0.216154 sekund 
// ------------------------------------------------
//  0.874675783496 Wartosc liczby PI rownolegle dla 4 wątków
// Czas procesorów przetwarzania równoleglego dla 4 wątków - 0.960492 sekund 
// Czas trwania obliczen rownoleglych dla 4 wątków - wallclock 0.240853 sekund 
// ------------------------------------------------

// Zmienne globalne: x, sum
// Zmienne lokalne: i

// Wynik jest niepoprawny, ponieważ wszystkie procesory logiczne mają dostęp do zmiennych x oraz sum, a zatem jednocześnie modyfikują te zmienne (wyścig).

// Występuje unieważnienie kopii linii pamięci - dane są modyfikowane.

// Przyspieszenie:
// Dla 16 wątków: 7.589891
// Dla 8 wątków:  4.370032
// Dla 4 wątków:  3.921894