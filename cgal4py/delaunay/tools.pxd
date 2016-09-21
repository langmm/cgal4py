cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

cdef extern from "c_tools.hpp":
    bool arg_tLT[I](I *cells, uint32_t *idx_verts, uint32_t ndim, uint64_t i1, uint64_t i2) nogil
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
    int64_t arg_partition_tess[I](I *cells, uint32_t *idx_verts, uint64_t *idx_cells, 
                                  uint32_t ndim, int64_t l, int64_t r, int64_t p) nogil
    void arg_quickSort_tess[I](I *cells, uint32_t *idx_verts, uint64_t *idx_cells, 
                               uint32_t ndim, int64_t l, int64_t r) nogil
    void arg_sortCellVerts[I](I *cells, uint32_t *idx_verts, uint64_t ncells, uint32_t ndim) nogil
    void arg_sortSerializedTess[I](I *cells, uint64_t ncells, uint32_t ndim,
                                   uint32_t *idx_verts, uint64_t *idx_cells) nogil
    cdef cppclass SerializedLeaf[I]:
        SerializedLeaf() except +
        SerializedLeaf(int _id, uint32_t _ndim, int64_t _ncells,
                       I _start_idx, I _stop_idx, I _idx_inf,
                       I *_verts, I *_neigh,
                       uint32_t *_sort_verts, uint64_t *_sort_cells) except +
        # ~SerializedLeaf() except +
    cdef cppclass ConsolidatedLeaves[I,leafI]:
        ConsolidatedLeaves() except +
        ConsolidatedLeaves(uint32_t _ndim, uint64_t _num_leaves, I _idx_inf,
                           I *_verts, I *_neigh, 
                           vector[SerializedLeaf[leafI]] _leaves) except +
        int64_t ncells
        

ctypedef SerializedLeaf[uint32_t] sLeaf32
ctypedef SerializedLeaf[uint64_t] sLeaf64
ctypedef vector[sLeaf32] sLeaves32
ctypedef vector[sLeaf64] sLeaves64
