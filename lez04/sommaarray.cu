#include <stdio.h>
#include <stdlib.h>

/**
 * Functions which execute on the GPU which can be invoked from the host are called kernels. 
 * Kernels are written to be run by many parallel threads simultaneously. 
 * The code for a kernel is specified using the __global__ declaration specifier. 
 * This indicates to the compiler that this function will be compiled for the GPU in a way that allows it to be invoked from a kernel launch. 
 * A kernel launch is an operation which starts a kernel running, usually from the CPU. 
 * Kernels are functions with a void return type.
 */
__global__ void sommaarray(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  int N, size;
  int i;

  N = 10;
  size = N * sizeof(float);

  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  for (i = 0; i < N; i++) {
    h_A[i] = i + 1.0f;
    h_B[i] = i + 1.0f;
  }

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  /**
   * The number of threads that will execute the kernel in parallel is specified as part of the kernel launch. 
   * This is called the execution configuration.
   * Different invocations of the same kernel may use different execution configurations, 
   * such as a different number of threads or thread blocks.
   * Triple chevron notation is a CUDA C++ Language Extension which is used to launch kernels. 
   * Execution configuration parameters are specified as a comma separated list inside the chevrons, 
   * similar to parameters to a function call. 
   * The first two parameters to the triple chevron notation are the grid dimensions and the thread block dimensions, respectively. 
   * When using 1-dimensional thread blocks or grids, integers can be used to specify dimensions. 
   * In the case of 2- or 3-dimensional thread blocks or grids, the dimensions are specified using the dim3 type.
   * The code below launches a single thread block containing N threads.
   * Each thread will execute the exact same kernel code. 
   * Each thread can use its index within the thread block and grid to change the data it operates on.
   */
  dim3 DimGrid(1, 1);
  dim3 DimBlock(N, 1, 1);
  sommaarray<<<DimGrid, DimBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for (i = 0; i < N; i++) {
    printf("%.2f ", h_C[i]);
  }
  printf("\n");

  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);

  free(h_C);
  free(h_B);
  free(h_A);
}
