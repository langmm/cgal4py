cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

cdef extern from "c_tools.hpp":
    bool intersect_sph_box(uint32_t ndim, double *c, double r, double *le, double *re) nogil
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
    void swap_cells[I](I *verts, I *neigh, uint32_t ndim, uint64_t i1, uint64_t i2) nogil
    cdef cppclass SerializedLeaf[I] nogil:
        SerializedLeaf() except +
        SerializedLeaf(int _id, uint32_t _ndim, int64_t _ncells, I _idx_inf,
                       I *_verts, I *_neigh,
                       uint32_t *_sort_verts, uint64_t *_sort_cells,
                       uint64_t idx_start, uint64_t idx_stop) except +
        int id
        uint32_t ndim
        int64_t ncells
        uint64_t idx_start
        uint64_t idx_stop
        I idx_inf
        I *verts
        I *neigh
        uint32_t *sort_verts
        uint64_t *sort_cells
        bool init_from_file
        void write_to_file(const char* filename)
        int64_t read_from_file(const char* filename)
        void cleanup()
    cdef cppclass ConsolidatedLeaves[I] nogil:
        ConsolidatedLeaves() except +
        ConsolidatedLeaves(uint32_t _ndim, I _idx_inf, int64_t _max_ncells,
                           I *_verts, I *_neigh) except +
        ConsolidatedLeaves(uint32_t _ndim, int64_t _ncells, I _idx_inf,
                           int64_t _max_ncells, I *_verts, I *_neigh) except +
        ConsolidatedLeaves(uint32_t _ndim, int64_t _ncells, I _idx_inf,
                           int64_t _max_ncells, I *_verts, I *_neigh,
                           uint64_t n_split_map, I *key_split_map, uint64_t *val_split_map,
                           uint64_t n_inf_map, I *key_inf_map, uint64_t *val_inf_map) except +
        int64_t ncells
        int64_t max_ncells
        I *allverts
        I *allneigh
        uint64_t size_split_map()
        uint64_t size_inf_map()
        void get_split_map(I *keys, uint64_t *vals) 
        void get_inf_map(I *keys, uint64_t *vals)
        void cleanup()
        void add_leaf[leafI](SerializedLeaf[leafI] leaf)
        void add_leaf_fromfile(const char *filename)
        int64_t count_inf()
        void add_inf()

ctypedef SerializedLeaf[uint32_t] sLeaf32
ctypedef SerializedLeaf[uint64_t] sLeaf64
ctypedef vector[sLeaf32] sLeaves32
ctypedef vector[sLeaf64] sLeaves64
