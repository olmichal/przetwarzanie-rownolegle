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
        double sum=0.0;
        omp_set_num_threads(threads[i]);
        ppstart = clock();
        pswtime = omp_get_wtime();

        #pragma omp parallel for
        for(i = 0; i < num_steps; i++) {
            double x = (i + .5)*step;
            #pragma omp atomic
            sum = sum + 4.0/(1.+ x*x);
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
//  3.141592653591 Wartosc liczby PI rownolegle dla 16 wątków
// Czas procesorów przetwarzania równoleglego dla 16 wątków - 1807.658712 sekund 
// Czas trwania obliczen rownoleglych dla 16 wątków - wallclock 124.723466 sekund 
// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle dla 8 wątków
// Czas procesorów przetwarzania równoleglego dla 8 wątków - 441.338305 sekund 
// Czas trwania obliczen rownoleglych dla 8 wątków - wallclock 58.453435 sekund 
// ------------------------------------------------
//  3.141592653590 Wartosc liczby PI rownolegle dla 4 wątków
// Czas procesorów przetwarzania równoleglego dla 4 wątków - 185.190490 sekund 
// Czas trwania obliczen rownoleglych dla 4 wątków - wallclock 47.094037 sekund 
// ------------------------------------------------

// Zmienne lokalne: i, x, sum

// Wynik wykonanych działań jest poprawny, a tak ogromne spowolnienie spowodowane jest atomowym dostępem do sum. 
// Podczas aktualizowania wartości sumy, nakładane są blokady przez co inne wątki nie mają do niej dostępu.

// Występuje unieważnienie kopii linii pamięci - dane są blokowane i modyfikowane.

// Dyrektywa #pragma omp atomic umożliwia atomowy dostęp do określonej lokalizacji w pamięci. 
// Zapewnia to uniknięcie sytuacji wyścigu poprzez bezpośrednią kontrolę współbieżnych wątków, które mogą odczytywać lub zapisywać do lub z określonej lokalizacji w pamięci.

// Niepodzielność możemy zapewnić dyrektywą: #pragma omp critical (ogranicza wykonanie powiązanej struktury do jednego wątku na raz)

// Przyspieszenie:
// Dla 16 wątków: 0.007573
// Dla 8 wątków:  0.016159
// Dla 4 wątków:  0.020057