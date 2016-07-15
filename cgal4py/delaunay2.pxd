cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t

cdef extern from "c_delaunay2.hpp":
    cdef cppclass Delaunay_with_info_2[Info]:
        Delaunay_with_info_2() except +
        Delaunay_with_info_2(double *pts, Info *val, uint32_t n) except +
        void write_to_file(const char* filename) except +
        void read_from_file(const char* filename) except +
        void insert(double *, Info *val, uint32_t n)
        uint32_t num_verts()
        # uint32_t num_edges()
        uint32_t num_cells()
        void edge_info(vector[pair[Info,Info]]& edges)
        void outgoing_points(double *left_edge, double *right_edge, bool periodic,
                             vector[Info]& lx, vector[Info]& ly,
                             vector[Info]& rx, vector[Info]& ry,
                             vector[Info]& alln)


cdef class Delaunay2:
    cdef int n
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef void _insert(self, np.ndarray[double, ndim=2, mode="c"] pts)
    cdef object _edge_info(self, int max_incl, np.uint64_t[:])
    cdef object _outgoing_points(self,
                                 np.ndarray[double, ndim=1] left_edge,
                                 np.ndarray[double, ndim=1] right_edge,
                                 bool periodic, object neighbors, int num_leaves)

