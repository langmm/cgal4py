cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool as cbool
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t, uint8_t


cdef extern from "c_parallel_delaunay.hpp":

    cdef cppclass CParallelDelaunay[Info] nogil:
        CParallelDelaunay()
        CParallelDelaunay(uint32_t ndim0, double *le0, double *re0,
                          cbool *periodic0)

        int rank
        int size
        uint32_t ndim
        uint64_t npts_total
        uint64_t *idx_total
        Info *info_total
        double *pts_total

        void insert(uint64_t npts, double *pts)

        uint64_t num_cells()
        uint64_t consolidate_tess(uint64_t tot_ncells_total, Info *tot_idx_inf,
                                  Info *allverts, Info *allneigh)
