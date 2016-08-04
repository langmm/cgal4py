#include <vector>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>

double* max_pts(double *pts, uint64_t n, uint32_t m);
double* min_pts(double *pts, uint64_t n, uint32_t m);
void quickSort(double *pts, uint64_t *idx,
               uint32_t ndim, uint32_t d,
               int64_t l, int64_t r);
void insertSort(double *pts, uint64_t *idx,
                uint32_t ndim, uint32_t d,
                int64_t l, int64_t r);
int64_t pivot(double *pts, uint64_t *idx,
              uint32_t ndim, uint32_t d,
              int64_t l, int64_t r);
int64_t partition(double *pts, uint64_t *idx,
                  uint32_t ndim, uint32_t d,
                  int64_t l, int64_t r, int64_t p);
int64_t select(double *pts, uint64_t *idx,
               uint32_t ndim, uint32_t d,
               int64_t l, int64_t r, int64_t n);
