#ifndef KERNELS_HPP
#define KERNELS_HPP

__global__ void hadamard(const double *__restrict__ A,
                         const double *__restrict__ B, double *__restrict__ C,
                         const size_t N, const size_t K, const size_t M,
                         const size_t ldA, const size_t ldB, const size_t ldC);

__global__ void matmat_coalesced(const double *__restrict__ A,
                                 const double *__restrict__ B,
                                 double *__restrict__ C, const size_t N,
                                 const size_t M, const size_t K,
                                 const size_t ldA, const size_t ldB,
                                 const size_t ldC);

__global__ void matmat_uncoalesced(const double *__restrict__ A,
                                   const double *__restrict__ B,
                                   double *__restrict__ C, const size_t N,
                                   const size_t M, const size_t K,
                                   const size_t ldA, const size_t ldB,
                                   const size_t ldC);

#endif /* KERNELS_HPP */