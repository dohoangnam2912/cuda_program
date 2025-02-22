/*
	File: naive_sgemm_cuda.cu
	Author: Tran Ba Thanh
	Created: 17/12/2024
	Last update: 18/12/2024
	Purpose: to demonstate naive matrix multiplicaiton on GPU
				compare CPU copy data then compute time with
				CPU copy data then compute.
				Also validate data between the two.
	
	More info: 
		A x B = C 
		With A, B and C contain all float values.

	Todo:
		- Funtion turning matrix from MxN into 1xN

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Devide and round up to the nearest multiplicant of N 
// Chia và làm tròn lên phép chia M/N
#define CEIL_DIV(M, N) ((M + N -1)/N)

#define M 256		// Number of row for mat A and mat C - số hàng của ma trận A và C 
#define N 512		// Number of col for Mat B and mat C - số cột của ma trận B và C
#define K 256		// Number of col for mat A, row for mat B - số cột cho ma trận A, số hàng cho ma trận B

#define BLOCK_SIZE 32	// number of thread in one side of the square block

#define T_TEST 30

void initMatrix(float *mat, int row, int col);
void sgemm_naive_cpu(float *A, float* B, float *C, int m, int n, int k);
__global__ void matmul_gpu_v1(float *A, float *B, float *C, int m, int n, int k);
__global__ void matmul_gpu_v2(float *A, float *B, float *C, int m, int n, int k);
void printMatrix(float *C, int m, int n);
double get_time();

int main()
{
	printf("Testing matmul on CPU and CUDA with matrix N = %d, M = %d, K = %d.\n", (int)N, (int)M, (int)K);

	float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
	float *d_A, *d_B, *d_C;
	int sizeA = sizeof(float) * M * K;
	int sizeB = sizeof(float) * K * N;
	int sizeC = sizeof(float) * M * N;

	// Allocate host memory
	h_A = (float*)malloc(sizeA);
	h_B = (float*)malloc(sizeB);
	h_C_cpu = (float*)malloc(sizeC);
	h_C_gpu = (float*)malloc(sizeC);

	// Set the seed of rand to current time
	srand(time(NULL));

	// Initialize matrices
	initMatrix(h_A, M, K);
	initMatrix(h_B, K, N);

	// Allocate device memory
	cudaMalloc(&d_A, sizeA);
	cudaMalloc(&d_B, sizeB);
	cudaMalloc(&d_C, sizeC);

	// Get the transfer h to d time
	printf("Measuring time for transfering data from host to device...\n");
	double data_transfer_total_time = 0.0;
	for (int i = 0; i < T_TEST; i++)
	{
		double start_time = get_time();
		cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
		double end_time = get_time();
		data_transfer_total_time += end_time - start_time;

		// Free memory
		cudaFree(d_A);
		cudaFree(d_B);

		// Reallocated memory
		cudaMalloc(&d_A, sizeA);
		cudaMalloc(&d_B, sizeB);
	}
	double data_transfer_h_to_d_avg_time = data_transfer_total_time / (float)T_TEST;

	// Actually transfer to device
	cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

	// Define grid and block dim
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));

    // Warm-up runs
    printf("Performing CPU warm-up runs...\n");
    for (int i = 0; i < 5; i++) 
    {
        sgemm_naive_cpu(h_A, h_B, h_C_cpu, M, N, K);
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < T_TEST; i++)
    {
        double start_time = get_time();
        sgemm_naive_cpu(h_A, h_B, h_C_cpu, M, N, K);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / (float)T_TEST;

    // Warm-up runs
    printf("Performing GPU warm-up runs...\n");
    for (int i = 0; i < 5; i++) 
    {
        matmul_gpu_v1<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < T_TEST; i++)
    {
        double start_time = get_time();
        matmul_gpu_v1<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / (float)T_TEST;

    printf("Measuring time for transfering data from device to host...\n");
	data_transfer_total_time = 0.0;
	for (int i = 0; i < T_TEST; i++)
	{
		double start_time = get_time();
		cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);
		double end_time = get_time();
		data_transfer_total_time += end_time - start_time;

		// Free memory
		free(h_C_gpu);

		// Reallocated memory
		h_C_gpu = (float*)malloc(sizeC);
	}
	cudaMemcpy(h_C_gpu	, d_C, sizeC, cudaMemcpyDeviceToHost);
	double data_transfer_d_to_h_avg_time = data_transfer_total_time / (float)T_TEST;

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average execution time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("GPU average transfer h to d time: %f microseconds\n", (data_transfer_h_to_d_avg_time * 1e6f));
    printf("GPU average transfer d to h time: %f microseconds\n", (data_transfer_d_to_h_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", ((data_transfer_h_to_d_avg_time + data_transfer_d_to_h_avg_time + gpu_avg_time) * 1e6f));
    printf("Speedup: %fx (not include transfer time)\n", cpu_avg_time / gpu_avg_time);

	// Free memory
	free(h_A);
	free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

	return 0;
}


/*
	Hàm để khởi tạo ma trận - Matrix init function
	Input:
		mat: ma trận ngang (MxN = 1xn)
		row: number of row that matrix has - số hàng của ma trận
		col: number of colum that matrix has - số cột của ma trận 
*/
void initMatrix(float *mat, int row, int col)
{
	for (int i = 0; i < row * col; i++)
	{
		mat[i] = (float)rand() / RAND_MAX;

		// for testing: generate integer value
		// mat[i] = (int)(rand() % 11);
	}
}

/*
	Hàm nhân ma trận - Funciton doing matrix multiplicaiton
	Input:
		C: ma trận kết quả của AB
		A: ma trận 1
		B: ma trận 2
		m: Number of row for mat A and mat C - số hàng của ma trận A và C 
		n: Number of col for Mat B and mat C - số cột của ma trận B và C
		k: Number of col for mat A, row for mat B - số cột cho ma trận A, số hàng cho ma trận B
*/
void sgemm_naive_cpu(float *A, float* B, float *C, int m, int n, int k)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float tmp = 0;
			for (int l = 0; l < k; l++)
			{
				tmp += A[i*k + l] * B[l*n + j];
			}
			C[i*n + j] = tmp;
		}
	}
}

__global__ void matmul_gpu_v1(float *A, float *B, float *C, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }

}

/*
	Function to display matrix
*/
void printMatrix(float *C, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%f ", C[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
