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
#define SHARED_MEMORY THREADS * THREADS * sizeof(float)					// Rozmiar pamięci współdzielonej

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void fillMatrix(float* matrix, const int size);
bool verify(float* A, float* B, float* C, const int N);

// Kernel
__global__ void multiplyKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, const int size);

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

	int WARP = THREADS;
	int BLOCKS = (int)ceil(SIZE / WARP);

	dim3 threads(WARP, WARP);
	dim3 blocks(BLOCKS, BLOCKS);

	std::cout << "Rozpoczynam przetwarzanie\n";
	multiplyKernel <<<blocks, threads>>> (d_a, d_b, d_c, WARP);
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "Koncze...\n";

	//UNCOMMENT THIS TO CHECK THAT YOUR GPU CALCULATED CORRECTLY - IT MAY TAKE A LITTLE BIT LONGER THAN NORMAL
	/*std::cout << "[CPU]: Verify CPU started\n";
	bool correct = verify(h_a, h_b, h_c, SIZE);
	std::cout << "[CPU]: Verify CPU finished\n";

	if (correct) {
		std::cout << "Succes\n";
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
float* __restrict__ C, 
const int tile_size) {
	int row = blockIdx.y * tile_size + threadIdx.y;
	int col = blockIdx.x * tile_size + threadIdx.x;

	__shared__ float shared_A[SHARED_MEMORY];
	__shared__ float shared_B[SHARED_MEMORY];

	float sum = 0;
	for (int i = 0; i < (SIZE / tile_size); i++) {
		int tile = threadIdx.y * tile_size + threadIdx.x;
		shared_A[tile] = A[row * SIZE + (i * tile_size + threadIdx.x)];
		shared_B[tile] = B[(i * tile_size * SIZE + threadIdx.y * SIZE) + col];

		__syncthreads();

		for (int j = 0; j < tile_size; j++) {
			sum += shared_A[(threadIdx.y * tile_size) + j] * shared_B[(j * tile_size) + threadIdx.x];
		}
		__syncthreads();
	}
	C[row * SIZE + col] = sum;
}

void fillMatrix(float* matrix, const int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			matrix[i * size + j] = (float)rand() / RAND_MAX;
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