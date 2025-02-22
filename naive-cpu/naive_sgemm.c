/*
	File: naive_sgemm.c
	Author: Tran Ba Thanh
	Created: 17/12/2024
	Last update: 17/12/2024
	Purpose: to demonstate naive matrix multiplicaiton on CPU
	
	More info: 
		A x B = C 
		With A, B and C contain all float values.

	Todo:
		- Funtion turning matrix from MxN into 1xN

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Devide and round up to the nearest multiplicant of N 
// Chia và làm tròn lên phép chia M/N
#define CEIL_DIV(M, N) ((M + N -1)/N)

#define M 3	// Number of row for mat A and mat C - số hàng của ma trận A và C 
#define N 2	// Number of col for Mat B and mat C - số cột của ma trận B và C
#define K 2	// Number of col for mat A, row for mat B - số cột cho ma trận A, số hàng cho ma trận B

void initMatrix(float *mat, int row, int col);
void sgemm_naive_cpu(float *A, float* B, float *C, int m, int n, int k);
void printMatrix(float *C, int m, int n);

int main()
{
	float *A, *B, *C;
	int sizeA = sizeof(float) * M * K;
	int sizeB = sizeof(float) * K * N;
	int sizeC = sizeof(float) * M * N;

	
	A = (float*)malloc(sizeA);
	B = (float*)malloc(sizeB);
	C = (float*)malloc(sizeC);

	initMatrix(A, M, K);
	initMatrix(B, K, N);
	
	sgemm_naive_cpu(A, B, C, M, N, K);

	printf("A = \n");
	printMatrix(A, M, K);

	printf("B = \n");
	printMatrix(B, K, N);

	printf("C = \n");
	printMatrix(C, M, N);

	// Free memory
	free(A);
	free(B);
	free(C);

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
	float tmp = 0;
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
