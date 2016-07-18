# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
"""
delaunay2.pyx

Wrapper for CGAL 2D Delaunay Triangulation
"""

import cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from cython.operator cimport dereference
from cython.operator cimport preincrement
from libc.stdint cimport uint32_t, uint64_t

cdef class Delaunay2_vertex:
    cdef Delaunay_with_info_2[uint32_t].All_verts_iter v

    def __richcmp__(Delaunay2_vertex self, Delaunay2_vertex solf, int op):
        if (op == 2): 
            return <pybool>(self.v == solf.v)
        elif (op == 3):
            return <pybool>(self.v != solf.v)
        else:
            raise NotImplementedError

    def increment(self):
        preincrement(self.v)

    property point:
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
            self.v.point(&out[0])
            return out

    property index:
        def __get__(self):
            cdef np.uint64_t out = self.v.info()
            return out

cdef class Delaunay2_vertex_range:
    cdef Delaunay2_vertex v
    cdef Delaunay2_vertex vstop
    def __cinit__(self, Delaunay2_vertex vstart, Delaunay2_vertex vstop):
        self.v = vstart
        self.vstop = vstop

    def __iter__(self):
        return self

    def __next__(self):
        self.v.increment()
        cdef Delaunay2_vertex out = self.v
        if self.v != self.vstop:
            return out
        else:
            raise StopIteration()


cdef class Delaunay2:
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.n = 0
        self.T = new Delaunay_with_info_2[uint32_t]()

    def write_to_file(self, fname):
        cdef char* cfname = fname
        self.T.write_to_file(cfname)

    def read_from_file(self, fname):
        cdef char* cfname = fname
        self.T.read_from_file(cfname)
        self.n = self.num_verts()

    def num_verts(self): return self.T.num_verts()
    # def num_edges(self): return self.T.num_edges()
    def num_cells(self): return self.T.num_cells()

    def insert(self, np.ndarray[double, ndim=2, mode="c"] pts not None):
        if pts.shape[0] != 0:
            self._insert(pts)
    cdef void _insert(self, np.ndarray[double, ndim=2, mode="c"] pts):
        cdef int Nold, Nnew, m
        Nold = self.n
        Nnew, m = pts.shape[0], pts.shape[1]
        cdef np.ndarray[uint32_t, ndim=1] idx
        idx = np.arange(Nold, Nold+Nnew).astype('uint32')
        assert(m == 2)
        self.T.insert(&pts[0,0], &idx[0], <uint32_t>Nnew)
        self.n += Nnew
        if self.n != self.num_verts():
            print "There were {} duplicates".format(self.n-self.num_verts())
        # assert(self.n == self.num_verts())

    def all_verts_begin(self):
        cdef Delaunay_with_info_2[uint32_t].All_verts_iter it_begin
        it_begin = self.T.all_verts_begin()
        cdef Delaunay2_vertex v_begin = Delaunay2_vertex()
        v_begin.v = it_begin
        return v_begin
    def all_verts_end(self):
        cdef Delaunay_with_info_2[uint32_t].All_verts_iter it_end
        it_end = self.T.all_verts_end()
        cdef Delaunay2_vertex v_end = Delaunay2_vertex()
        v_end.v = it_end
        return v_end
    def all_verts(self):
        return Delaunay2_vertex_range(self.all_verts_begin(), 
                                      self.all_verts_end())

    def edge_info(self, max_incl, idx):
        return self._edge_info(max_incl, idx)
    cdef object _edge_info(self, int max_incl, np.uint64_t[:] idx):
        cdef object edge_list = []
        cdef vector[pair[uint32_t,uint32_t]] edge_vect
        cdef int j
        cdef uint32_t i1, i2
        self.T.edge_info(edge_vect)
        for j in xrange(<int>edge_vect.size()):
            i1 = edge_vect[j].first
            i2 = edge_vect[j].second
            if idx[i2] < idx[i1]:
                i1, i2 = i2, i1
            if i1 < (<uint32_t>max_incl):
                edge_list.append(np.array([i1,i2],'int64'))
        return np.array(edge_list)

