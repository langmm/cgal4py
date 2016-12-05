cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool as cbool
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t, uint8_t

cdef extern from "c_parallel_delaunay.hpp":

    cdef cppclass CParallelDelaunay nogil:
        CParallelDelaunay()
        void run(uint64_t npts, uint32_t ndim, double *pts,
                 double *le, double *re, cbool *periodic)
        
