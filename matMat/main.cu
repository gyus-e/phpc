#include "mat_kernels.hpp"
#include "mat_utils.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void benchmarkKernel(matmatkernel_t kernel, const char *kernelName,
                     const dim3 gridDim, const dim3 blockDim, const size_t sharedMem, 
                     const double *__restrict__ dA, const double *__restrict__ dB, 
                     double *__restrict__ dC, double *hC,
                     const size_t N, const size_t K, const size_t M,
                     const size_t ldA, const size_t ldB, const size_t ldC) {
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemset(dC, 0, N * M * sizeof(double));

  cudaEventRecord(start);
  kernel<<<gridDim, blockDim, sharedMem>>>(dA, dB, dC, N, K, M, ldA, ldB, ldC);
  cudaEventRecord(stop);
  checkCUDAError(kernelName);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("%s time: %f ms\n", kernelName, milliseconds);

  cudaMemcpy2D(hC, N * sizeof(double), dC, ldC * sizeof(double),
               N * sizeof(double), M, cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy2D");

  print_mat(hC, 4, 4, M);
}

int main(int argc, char *argv[]) {
  const size_t N = 4069;
  const size_t K = 4069;
  const size_t M = 4069;

  double *hA, *dA;
  double *hB, *dB;
  double *hC, *dC;
  size_t pitchA, pitchB, pitchC;

  dim3 blockDim(32, 32);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  hA = (double *)malloc(N * K * sizeof(double));
  hB = (double *)malloc(K * M * sizeof(double));
  hC = (double *)malloc(N * M * sizeof(double));

  cudaMallocPitch((void **)&dA, &pitchA, N * sizeof(double), K);
  cudaMallocPitch((void **)&dB, &pitchB, K * sizeof(double), M);
  cudaMallocPitch((void **)&dC, &pitchC, N * sizeof(double), M);
  checkCUDAError("cudaMallocPitch");

  const size_t ldA = pitchA / sizeof(double);
  const size_t ldB = pitchB / sizeof(double);
  const size_t ldC = pitchC / sizeof(double);

  init_mat(hA, N, K, K);
  init_mat(hB, K, M, M);

  cudaMemcpy2D(dA, pitchA, hA, N * sizeof(double), N * sizeof(double), K,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(dB, pitchB, hB, K * sizeof(double), K * sizeof(double), M,
               cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy2D");

  benchmarkKernel(hadamard, "hadamard", gridDim, blockDim, 0, 
                  dA, dB, dC, hC, N, K, M, ldA, ldB, ldC);

  benchmarkKernel(matmat_uncoalesced, "matmat_uncoalesced", gridDim, blockDim, 0,
                  dA, dB, dC, hC, N, K, M, ldA, ldB, ldC);

  benchmarkKernel(matmat_coalesced, "matmat_coalesced", gridDim, blockDim, 0,
                  dA, dB, dC, hC, N, K, M, ldA, ldB, ldC);
                  
  benchmarkKernel(matmat_tiled, "matmat_tiled", gridDim, blockDim, 0, 
                  dA, dB, dC, hC, N, K, M, ldA, ldB, ldC);

  free(hA);
  free(hB);
  free(hC);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
