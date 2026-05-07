#include "mat_utils.hpp"
#include <string.h>
#include <stdio.h>

void print_mat(const double *mat, const size_t rows, const size_t cols, const size_t ld) {
  unsigned int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%.1f \t", mat[i * ld + j]);
    }
    printf("\n");
  }
}

void init_mat(double *mat, const size_t rows, const size_t cols, const size_t ld) {
  unsigned int i, j;
  double x = 0.0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      mat[i * ld + j] = ++x;
    }
  }
}

void matmatikj(const double *A, const double *B, double *C, 
               const size_t N, const size_t K, const size_t M, 
               const size_t ldA, const size_t ldB, const size_t ldC) {
  unsigned int i, j, k;
  for (i = 0; i < N; i++) {
    for (k = 0; k < K; k++) {
      for (j = 0; j < M; j++) {
        C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
      }
    }
  }
}

void matmatblock(const double *A, const double *B, double *C, 
                 const size_t N, const size_t K, const size_t M, 
                 const size_t ldA, const size_t ldB, const size_t ldC, 
                 const size_t dbA, const size_t dbB, const size_t dbC) {
  unsigned int row_A, col_B;
  unsigned int idxA, idxB, idxC;

  unsigned int ii, jj, kk;
  for (ii = 0; ii < N / dbA; ii++) {
    row_A = ii * dbA;

    for (jj = 0; jj < M / dbC; jj++) {
      col_B = jj * dbC;
      idxC = row_A * ldC + col_B;

      for (kk = 0; kk < K / dbB; kk++) {
        idxA = (row_A * ldA) + (kk * dbB);
        idxB = (kk * dbB) * (ldB + col_B);

        matmatikj(&A[idxA], &B[idxB], &C[idxC], dbA, dbB, dbC, ldA, ldB, ldC);
      }
    }
  }
}
