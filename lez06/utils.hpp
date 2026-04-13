#include <math.h>

#ifndef UTILS_HPP
#define UTILS_HPP

#define WARP_SIZE 32
#define MAX_BLKSZ 1024
#define FULL_MASK 0xffffffff

inline __device__ __host__ double f(const double x) { return x*exp(-x)*cos(2*x); }
// integrale tra 0 e 2pi uguale a −0.12212260462

#endif /* UTILS_HPP */
