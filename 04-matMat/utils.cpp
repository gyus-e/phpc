#include "utils.hpp"
#include <stdio.h>

void print_mat(const double *mat, const size_t rows, const size_t cols,
               const size_t ld) {
  unsigned int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%.1f \t", mat[i * ld + j]);
    }
    printf("\n");
  }
}

void init_mat(double *mat, const size_t rows, const size_t cols,
              const size_t ld) {
  unsigned int i, j;
  double x = 0.0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      mat[i * ld + j] = ++x;
    }
  }
}
