#include "kernels.hpp"
#include "utils.hpp"
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

int main(int argc, char *argv[]) {
  const size_t N = 4069;
  const size_t K = 4069;
  const size_t M = 4069;

  double *hA, *dA;
  double *hB, *dB;
  double *hC, *dC;
  size_t pitchA, pitchB, pitchC;

  dim3 blockDim(32, 2);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

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

  for (int i = 0; i <= 1; i++) {
    cudaMemset(dC, 0, N * M * sizeof(double));

    if (i == 1) {
      cudaEventRecord(start);
      matmat_coalesced<<<gridDim, blockDim>>>(dA, dB, dC, N, K, M, ldA, ldB,
                                              ldC);
      cudaEventRecord(stop);
      checkCUDAError("matmat_coalesced");
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("matmat_coalesced time: %f ms\n", milliseconds);
    } else {
      cudaEventRecord(start);
      matmat_uncoalesced<<<gridDim, blockDim>>>(dA, dB, dC, N, K, M, ldA, ldB,
                                                ldC);
      cudaEventRecord(stop);
      checkCUDAError("matmat_uncoalesced");
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("matmat_uncoalesced time: %f ms\n", milliseconds);
    }

    cudaMemcpy2D(hC, N * sizeof(double), dC, pitchC, N * sizeof(double), M,
                 cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy2D");
    print_mat(hC, 4, 4, M);
  }

  free(hA);
  free(hB);
  free(hC);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
