#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "driver_types.h"
#include "device_functions.h"
#include <omp.h>
#include <cstdio>
#include <random>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cassert>

#define SIZE  (1 << 10) * 3												// Rozmiar		
#define THREADS 32														// Liczba wątków na blok
#define BYTES SIZE * SIZE * sizeof(float)								// Liczba bytów do alokacji
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void fillMatrix(float* matrix, const int N);
bool verify(float* A, float* B, float* C, const int N);

// Kernel
__global__ void multiplyKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int N);

int main() {

	srand(time(NULL));
	float* h_a = nullptr;
	float* h_b = nullptr;
	float* h_c = nullptr;

	// Allocate memory on host for A, B, C matrices
	// Paged memory - prepared
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
	checkCudaErrors(cudaHostAlloc(&h_a, BYTES, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&h_b, BYTES, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&h_c, BYTES, cudaHostAllocMapped));

	fillMatrix(h_a, SIZE);
	fillMatrix(h_b, SIZE);

	// Set C-matrix to 0s
	checkCudaErrors(cudaMemset(h_c, 0, BYTES));

	float* d_a = nullptr;
	float* d_b = nullptr;
	float* d_c = nullptr;

	// Zero copy
	// Sync hosts and devices
	checkCudaErrors(cudaHostGetDevicePointer(&d_a, h_a, 0));
	checkCudaErrors(cudaHostGetDevicePointer(&d_b, h_b, 0));
	checkCudaErrors(cudaHostGetDevicePointer(&d_c, h_c, 0));

	int WARP = THREADS; // 8 || 16 || 32
	int BLOCKS = (int)ceil(SIZE / WARP); // sufit z 3072 / WARP

	dim3 threads(WARP, WARP);
	dim3 blocks(BLOCKS, BLOCKS);

	std::cout << "Rozpoczynam przetwarzanie\n";
	multiplyKernel <<<blocks, threads>>> (d_a, d_b, d_c, SIZE);
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "Koncze...\n";

	//  UNCOMMENT THIS TO CHECK THAT YOUR GPU CALCULATED CORRECTLY - IT MAY TAKE A LITTLE BIT LONGER THAN NORMAL
	/*std::cout << "[CPU]: Verify CPU started\n";
	bool correct = verify(h_a, h_b, h_c, SIZE);
	std::cout << "[CPU]: Verify CPU finished\n";

	if (correct) {
		std::cout << "Success\n";
	}
	else {
		std::cout << "Failure\n";
	}*/
	

	// Free memory
	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}

__global__ void multiplyKernel(const float* __restrict__ A, 
const float* __restrict__ B, 
float* __restrict__ C, const int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	C[row * N + col] = 0;
	for (int k = 0; k < N; k++) {
		C[row * N + col] += A[row * N + k] * B[k * N + col];
	}
}

void fillMatrix(float* matrix, const int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			matrix[i * N + j] = (float)rand() / RAND_MAX;
		}
	}
}

bool verify(float* A, float* B, float* C, const int N) {
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			float tmp = 0;
			for (int element = 0; element < N; ++element) {
				tmp += A[row * N + element] * B[element * N + col];
			}
			// Check against the CPU result
			if (tmp != C[row * N + col]) {
				return false;
			}
		}
	}
	return true;
}