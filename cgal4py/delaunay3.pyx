# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
"""
delaunay3.pyx

Wrapper for CGAL 3D Delaunay Triangulation
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

cdef class Delaunay3_vertex:
    r"""Wrapper class for a triangulation vertex.
 
    Attributes:
        v (:obj:`Delaunay_with_info_3[uint32_t].All_verts_iter`): C++ vertex 
            object. Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_3[uint32_t].All_verts_iter v

    def __richcmp__(Delaunay3_vertex self, Delaunay3_vertex solf, int op):
        if (op == 2):
            return <pybool>(self.v == solf.v)
        elif (op == 3):
            return <pybool>(self.v != solf.v)
        else:
            raise NotImplementedError

    def increment(self):
        r"""Advance to the next vertex in the triangulation."""
        preincrement(self.v)

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
        of the vertex."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
            self.v.point(&out[0])
            return out

    property index:
        r"""uint64: The index of the vertex point in the input array."""
        def __get__(self):
            cdef np.uint64_t out = self.v.info()
            return out


cdef class Delaunay3_vertex_range:
    r"""Wrapper class for iterating over a range of triangulation vertices

    Args:
        vstart (Delaunay3_vertex): The starting vertex.  
        vstop (Delaunay3_vertex): Final vertex that will end the iteration. 

    Attributes:
        v (Delaunay3_vertex): The current vertex. 
        vstop (Delaunay3_vertex): Final vertex that will end the iteration. 

    """
    cdef Delaunay3_vertex v
    cdef Delaunay3_vertex vstop
    def __cinit__(self, Delaunay3_vertex vstart, Delaunay3_vertex vstop):
        self.v = vstart
        self.vstop = vstop

    def __iter__(self):
        return self

    def __next__(self):
        self.v.increment()
        cdef Delaunay3_vertex out = self.v
        if self.v != self.vstop:
            return out
        else:
            raise StopIteration()


cdef class Delaunay3:
    r"""Wrapper class for a 3D Delaunay triangulation.

    Attributes:
        n (int): The number of points inserted into the triangulation.
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 

    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.n = 0
        self.T = new Delaunay_with_info_3[uint32_t]()

    def write_to_file(self, fname):
        r"""Write the serialized tessellation information to a file. 

        Args:
            fname (str): The full path to the file that the tessellation should 
                be written to.

        """
        cdef char* cfname = fname
        self.T.write_to_file(cfname)

    def read_from_file(self, fname):
        r"""Read serialized tessellation information from a file.

        Args:
            fname (str): The full path to the file that the tessellation should 
                be read from.

        """
        cdef char* cfname = fname
        self.T.read_from_file(cfname)
        self.n = self.num_verts

    @property
    def num_verts(self): 
        r"""int: The number of unique vertices in the triangulation."""
        return self.T.num_verts()
    @property
    def num_cells(self):
        r"""int: The number of tetrahedral cells in the triangulation."""
        return self.T.num_cells()

    def insert(self, np.ndarray[double, ndim=2, mode="c"] pts not None):
        r"""Insert points into the triangulation.

        Args:
            pts (:obj:`ndarray` of :obj:`float64`): Array of 3D cartesian 
                points to insert into the triangulation. 

        """
        if pts.shape[0] != 0:
            self._insert(pts)
    cdef void _insert(self, np.ndarray[double, ndim=2, mode="c"] pts):
        cdef int Nold, Nnew, m
        Nold = self.n
        Nnew, m = pts.shape[0], pts.shape[1]
        cdef np.ndarray[uint32_t, ndim=1] idx
        idx = np.arange(Nold, Nold+Nnew).astype('uint32')
        assert(m == 3)
        self.T.insert(&pts[0,0], &idx[0], <uint32_t>Nnew)
        self.n += Nnew
        if self.n != self.num_verts:
            print "There were {} duplicates".format(self.n-self.num_verts)
        # assert(self.n == self.num_verts)

    @property
    def all_verts_begin(self):
        r"""Delaunay3_vertex: Starting vertex for the triangulation."""
        cdef Delaunay_with_info_3[uint32_t].All_verts_iter it_begin
        it_begin = self.T.all_verts_begin()
        cdef Delaunay3_vertex v_begin = Delaunay3_vertex()
        v_begin.v = it_begin
        return v_begin
    @property
    def all_verts_end(self):
        r"""Delaunay3_vertex: Final vertex in the triangulation."""
        cdef Delaunay_with_info_3[uint32_t].All_verts_iter it_end
        it_end = self.T.all_verts_end()
        cdef Delaunay3_vertex v_end = Delaunay3_vertex()
        v_end.v = it_end
        return v_end
    @property
    def all_verts(self):
        r"""Delaunay3_vertex_range: Iterable for all vertices in the 
        triangulation."""
        return Delaunay3_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end)

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

