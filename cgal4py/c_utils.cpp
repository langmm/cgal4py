#include <vector>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include "c_utils.hpp"

double* max_pts(double *pts, uint64_t n, uint32_t m)
{
  double* max = (double*)malloc(m*sizeof(double));
  uint32_t d;
  for (d = 0; d < m; d++) max[d] = pts[d];
  for (uint64_t i = 0; i < n; i++) {
    for (d = 0; d < m; d++) {
      if (pts[m*i + d] > max[d])
        max[d] = pts[m*i + d];
    }
  }
  return max;
}

double* min_pts(double *pts, uint64_t n, uint32_t m)
{
  double* min = (double*)malloc(m*sizeof(double));
  uint32_t d;
  for (d = 0; d < m; d++) min[d] = pts[d];
  for (uint64_t i = 0; i < n; i++) {
    for (d = 0; d < m; d++) {
      if (pts[m*i + d] < min[d])
        min[d] = pts[m*i + d];
    }
  }
  return min;
}

// http://www.comp.dit.ie/rlawlor/Alg_DS/sorting/quickSort.c 
void quickSort(double *pts, uint64_t *idx,
               uint32_t ndim, uint32_t d,
               int64_t l, int64_t r)
{
  int64_t j;
  if( l < r )
    {
      // divide and conquer
      j = partition(pts, idx, ndim, d, l, r, (l+r)/2);
      quickSort(pts, idx, ndim, d, l, j-1);
      quickSort(pts, idx, ndim, d, j+1, r);
    }
}

void insertSort(double *pts, uint64_t *idx,
                uint32_t ndim, uint32_t d,
                int64_t l, int64_t r)
{
  int64_t i, j;
  uint64_t t;

  if (l == r) return;
  for (i = l+1; i <= r; i++) {
    t = idx[i];
    j = i - 1;
    while ((j >= l) && (pts[ndim*idx[j]+d] > pts[ndim*t+d])) {
      idx[j+1] = idx[j];
      j--;
    }
    idx[j+1] = t;
  }
}

int64_t pivot(double *pts, uint64_t *idx,
              uint32_t ndim, uint32_t d,
              int64_t l, int64_t r)
{ 
  if ((r - l) < 5) {
    insertSort(pts, idx, ndim, d, l, r);
    return (l+r)/2;
  }

  int64_t i, subr, m5;
  uint64_t t;
  int64_t nsub = 0;
  for (i = l; i <= r; i+=5) {
    subr = i + 4;
    if (subr > r) subr = r;

    insertSort(pts, idx, ndim, d, i, subr);
    m5 = (i+subr)/2;
    t = idx[m5]; idx[m5] = idx[l + nsub]; idx[l + nsub] = t;

    nsub++;
  }
  return select(pts, idx, ndim, d, l, l+nsub-1, l+(nsub-1)/2);
}

int64_t partition(double *pts, uint64_t *idx,
                  uint32_t ndim, uint32_t d,
                  int64_t l, int64_t r, int64_t p)
{ 
  double pivot;
  int64_t i, j;
  uint64_t t;
  pivot = pts[ndim*idx[p]+d];
  t = idx[p]; idx[p] = idx[l]; idx[l] = t;

  for (i = l+1, j = r; i <= j; ) {
    if ((pts[ndim*idx[i]+d] > pivot) && (pts[ndim*idx[j]+d] <= pivot)) {
      t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    if (pts[ndim*idx[i]+d] <= pivot) i++;
    if (pts[ndim*idx[j]+d] > pivot) j--;
  }

  t = idx[l]; idx[l] = idx[j]; idx[j] = t;

  return j;
}

// https://en.wikipedia.org/wiki/Median_of_medians
int64_t select(double *pts, uint64_t *idx,
               uint32_t ndim, uint32_t d,
               int64_t l, int64_t r, int64_t n)
{
  int64_t p;

  while ( 1 ) {
    if (l == r) return l;

    p = pivot(pts, idx, ndim, d, l, r);
    p = partition(pts, idx, ndim, d, l, r, p);
    if (n == p) {
      return n;
    } else if (n < p) {
      r = p - 1;
    } else {
      l = p + 1;
    }
  }
}





