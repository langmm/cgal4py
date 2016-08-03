cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t


cdef extern from "c_utils.hpp":
    double* max_pts(double *pts, uint64_t n, uint64_t m)
    double* min_pts(double *pts, uint64_t n, uint64_t m)
    void quickSort(double *pts, uint64_t *idx,
                   uint32_t ndim, uint32_t d,
                   int64_t l, int64_t r)
    int64_t partition(double *pts, uint64_t *idx,
                      uint32_t ndim, uint32_t d,
                      int64_t l, int64_t r, int64_t p)
    int64_t select(double *pts, uint64_t *idx,
                   uint32_t ndim, uint32_t d,
                   int64_t l, int64_t r, int64_t n)
    int64_t pivot(double *pts, uint64_t *idx,
                  uint32_t ndim, uint32_t d,
                  int64_t l, int64_t r)
    void insertSort(double *pts, uint64_t *idx,
                    uint32_t ndim, uint32_t d,
                    int64_t l, int64_t r)
