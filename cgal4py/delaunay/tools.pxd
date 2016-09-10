cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

cdef extern from "c_tools.hpp":
    bool tEQ[I](I *cells, uint32_t ndim, int64_t i1, int64_t i2) nogil
    bool tGT[I](I *cells, uint32_t ndim, int64_t i1, int64_t i2) nogil
    bool tLT[I](I *cells, uint32_t ndim, int64_t i1, int64_t i2) nogil
    int64_t arg_partition[I](I *arr, I *idx, uint32_t ndim,
                             int64_t l, int64_t r, int64_t p) nogil
    void arg_quickSort[I](I *arr, I *idx, uint32_t ndim,
                          int64_t l, int64_t r) nogil
    int64_t partition_tess[I](I *cells, I *neigh, I *idx, uint32_t ndim,
                              int64_t l, int64_t r, int64_t p) nogil
    void quickSort_tess[I](I *cells, I *neigh, I *idx, uint32_t ndim,
                           int64_t l, int64_t r) nogil
    void sortCellVerts[I](I *cells, I *neigh, uint64_t ncells, uint32_t ndim) nogil
    void sortSerializedTess[I](I *cells, I *neigh, 
                               uint64_t ncells, uint32_t ndim) nogil
