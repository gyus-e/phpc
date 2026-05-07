#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.hpp"

#define BLOCK_SIZE 32

__global__ void cudaMatMatNaive(double *A, double *B, double *C, const unsigned int N, const unsigned int M, const unsigned int K){
    int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    if (rowIdx < N && colIdx < M){
        for (int k = 0; k < K; k++){
            sum += A[rowIdx * K + k] * B[k * N + colIdx];
        }
        C[rowIdx * N + colIdx] = sum;
    }
  }

__global__ void cudaMatMat(double *A, double *B, double *C, const unsigned int N, const unsigned int M, const unsigned int K){
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;
    if (rowIdx < N && colIdx < M){
        for (int k = 0; k < K; k++){
            sum += A[rowIdx * K + k] * B[k * N + colIdx];
        }
        C[rowIdx * N + colIdx] = sum;
    }
}


int main() {
  const unsigned int N = 8192;
  const unsigned int M = 8192;
  const unsigned int K = 8192;
  const unsigned int LD = 8192;

  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double *h_A, *d_A; // N x K
  double *h_B, *d_B; // K x M
  double *h_C, *d_C; // N x M

  dim3 DimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

  h_A = (double *)malloc(N * LD * sizeof(double));
  h_B = (double *)malloc(K * LD * sizeof(double));
  h_C = (double *)calloc(N * LD, sizeof(double));
  cudaMalloc((void **)&d_A, N * LD * sizeof(double));
  cudaMalloc((void **)&d_B, K * LD * sizeof(double));
  cudaMalloc((void **)&d_C, N * LD * sizeof(double));

  init_mat(h_A, N, K, LD);
  init_mat(h_B, K, M, LD);

  cudaMemcpy(d_A, h_A, N * LD * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * LD * sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

  cudaMatMatNaive<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, N, M, K);
  cudaEventRecord(stop, 0);
  cudaMemcpy(h_C, d_C, N * LD * sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed time: %f ms\n", milliseconds);

  print_mat(h_C, 5, 5, LD);



  init_mat(h_A, N, K, LD);
  init_mat(h_B, K, M, LD);
  for (int i=0;i<N;i++){
    for (int j=0;j<M;j++){
      h_C[i * LD + j] = 0.0;
    }
  }

  cudaMemcpy(d_A, h_A, N * LD * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * LD * sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  cudaMatMat<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, N, M, K);
  cudaEventRecord(stop, 0);
  cudaMemcpy(h_C, d_C, N * LD * sizeof(double), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed time: %f ms\n", milliseconds);


  print_mat(h_C, 5, 5, LD);


  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);
  free(h_C);
  free(h_B);
  free(h_A);

  return 0;
}
