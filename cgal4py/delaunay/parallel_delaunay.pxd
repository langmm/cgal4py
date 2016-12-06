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

        uint32_t ndim

        void insert(uint64_t npts, double *pts)
        
        void consolidate_tess()
