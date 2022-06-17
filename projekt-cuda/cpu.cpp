#include <stdio.h>
#include <cstdlib>
#include <omp.h>

constexpr int N = 3072;         // liczba kolumn i wierszy

float* A = new float[N * N];    // lewy operand
float* B = new float[N * N];    // pracy operand
float* C = new float[N * N];    // wynik

void initialize_matrices() {
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)rand() / RAND_MAX;
            B[i * N + j] = (float)rand() / RAND_MAX;
            C[i * N + j] = 0.0;
        }
    }
}

void multiply_matrices_IKJ() {
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    double start, stop;
    printf("liczba watkow = %d\n\n", omp_get_max_threads());
    initialize_matrices();

    start = omp_get_wtime();
    multiply_matrices_IKJ();
    stop = omp_get_wtime();
    printf("IKJ: %1.8f\n", stop - start);
    return(0);
}