"""
delaunay2.pyx

Wrapper for CGAL 2D Delaunay Triangulation
"""

import cython

import numpy as np
cimport numpy as np
import struct

from cgal4py import PY_MAJOR_VERSION
from cgal4py import plot
from cgal4py.delaunay.tools cimport sortSerializedTess

from functools import wraps

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from cython.operator cimport dereference
from cython.operator cimport preincrement, predecrement
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

ctypedef uint32_t info_t
cdef object np_info = np.uint32
ctypedef np.uint32_t np_info_t

def is_valid():
    if (VALID == 1):
        return True
    else:
        return False

cdef class PeriodicDelaunay2_vertex:
    r"""Wrapper class for a triangulation vertex.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
            object that this vertex belongs to.
        x (:obj:`PeriodicDelaunay_with_info_2[info_t].Vertex`): C++ vertex 
            object. Direct interaction with this object is not recommended.

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef PeriodicDelaunay_with_info_2[info_t].Vertex x

    cdef void assign(self, PeriodicDelaunay_with_info_2[info_t] *T,
                     PeriodicDelaunay_with_info_2[info_t].Vertex x):
        r"""Assign C++ objects to attributes.

        Args:
            T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
                object that this vertex belongs to.
            x (:obj:`PeriodicDelaunay_with_info_2[info_t].Vertex`): C++ vertex 
                object. Direct interaction with this object is not recommended.

        """
        self.T = T
        self.x = x

    def __repr__(self):
        return "PeriodicDelaunay2_vertex[{} at {:+7.2e},{:+7.2e}]".format(
            self.index, *list(self.point))

    def __richcmp__(PeriodicDelaunay2_vertex self, 
                    PeriodicDelaunay2_vertex solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the vertex is the infinite vertex.

        Returns:
            bool: True if the vertex is the infinite vertex, False otherwise.

        """
        return self.T.is_infinite(self.x)

    def set_point(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Set this vertex's coordinates.

        Args:
            pos (:obj:`ndarray` of float64): new x,y coordinates for this 
                vertex.

        """
        self.T.updated = <cbool>True
        assert(len(pos) == 2)
        self.x.set_point(&pos[0])

    def set_cell(self, PeriodicDelaunay2_cell c):
        r"""Assign this vertex's designated cell.

        Args:
            c (PeriodicDelaunay2_cell): Cell that will be assigned as designated 
                cell.

        """
        self.T.updated = <cbool>True
        self.x.set_cell(c.x)

    property has_offset:
        r""":obj:`bool`: True if the vertex has a periodic offset (not including 
        any cell offset. False otherwise."""
        def __get__(self):
            return <pybool>self.T.has_offset(self.x)

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
        the vertex including the periodic offset."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
            self.T.point(self.x, &out[0])
            return out

    property periodic_point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
        the vertex, not including the periodic offset."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
            self.T.periodic_point(self.x, &out[0])
            return out

    property periodic_offset:
        r""":obj:`ndarray` of :obj:`int`: The number of wrappings applied in 
        each dimension to create this periodic point."""
        def __get__(self):
            cdef np.ndarray[np.int32_t] out = np.zeros(2, 'int32')
            self.T.periodic_offset(self.x, &out[0])
            return out

    property index:
        r"""info_t: The index of the vertex point in the input array."""
        def __get__(self):
            global np_info
            if self.is_infinite():
                out = np.iinfo(np_info).max
            else:
                out = self.x.info()
            return out

    property dual_volume:
        r"""float64: The area of the dual Voronoi cell. If the area is 
        infinite, -1.0 is returned."""
        def __get__(self):
            cdef np.float64_t out = self.T.dual_area(self.x)
            return out

    property cell:
        r"""PeriodicDelaunay2_cell: The cell assigned to this vertex."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_2[info_t].Cell c
            c = self.x.cell()
            cdef PeriodicDelaunay2_cell out = PeriodicDelaunay2_cell()
            out.assign(self.T, c)
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this vertex.

        Returns:
            PeriodicDelaunay2_vertex_vector: Iterator over vertices incident to 
                this vertex.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef PeriodicDelaunay2_vertex_vector out
        out = PeriodicDelaunay2_vertex_vector()
        out.assign(self.T, it)
        return out
        
    def incident_edges(self):
        r"""Find edges that are incident to this vertex.

        Returns:
            PeriodicDelaunay2_edge_vector: Iterator over edges incident to this 
                vertex.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef PeriodicDelaunay2_edge_vector out = PeriodicDelaunay2_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this vertex.

        Returns:
            PeriodicDelaunay2_cell_vector: Iterator over cells incident to this 
                vertex.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef PeriodicDelaunay2_cell_vector out = PeriodicDelaunay2_cell_vector()
        out.assign(self.T, it)
        return out


cdef class PeriodicDelaunay2_vertex_iter:
    r"""Wrapper class for a triangulation vertex iterator.

    Args:
        T (PeriodicDelaunay2): Triangulation that this vertex belongs to.
        vert (:obj:`str`, optional): String specifying the vertex that 
            should be referenced. Valid options include:
                'all_begin': The first vertex in an iteration over all vertices.
                'all_end': The last vertex in an iteration over all vertices.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
            object that this vertex belongs to.
        x (:obj:`PeriodicDelaunay_with_info_2[info_t].All_verts_iter`): C++ 
            vertex iteration object. Direct interaction with this object is not 
            recommended.

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef PeriodicDelaunay_with_info_2[info_t].All_verts_iter x
    
    def __cinit__(self, PeriodicDelaunay2 T, str vert = None):
        self.T = T.T
        if vert == 'all_begin':
            self.x = self.T.all_verts_begin()
        elif vert == 'all_end':
            self.x = self.T.all_verts_end()

    def __richcmp__(PeriodicDelaunay2_vertex_iter self, 
                    PeriodicDelaunay2_vertex_iter solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def increment(self):
        r"""Advance to the next vertex in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous vertex in the triangulation."""
        predecrement(self.x)

    property vertex:
        r"""PeriodicDelaunay2_vertex: The corresponding vertex object."""
        def __get__(self):
            cdef PeriodicDelaunay2_vertex out = PeriodicDelaunay2_vertex()
            out.assign(self.T, 
                       PeriodicDelaunay_with_info_2[info_t].Vertex(self.x))
            return out

cdef class PeriodicDelaunay2_vertex_vector:
    r"""Wrapper class for a vector of vertices. 

    Attributes: 
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended. 
        v (:obj:`vector[PeriodicDelaunay_with_info_2[info_t].Vertex]`): Vector 
            of C++ vertices. 
        n (int): The number of vertices in the vector. 
        i (int): The index of the currect vertex. 

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef vector[PeriodicDelaunay_with_info_2[info_t].Vertex] v
    cdef int n
    cdef int i

    cdef void assign(self, PeriodicDelaunay_with_info_2[info_t] *T,
                     vector[PeriodicDelaunay_with_info_2[info_t].Vertex] v):
        r"""Assign C++ attributes. 

        Args: 
            T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ triangulation 
                object. Direct interaction with this object is not recommended. 
            v (:obj:`vector[PeriodicDelaunay_with_info_2[info_t].Vertex]`): 
                Vector of C++ vertices. 

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay2_vertex out
        if self.i < self.n:
            out = PeriodicDelaunay2_vertex()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef PeriodicDelaunay2_vertex out
        if isinstance(i, int):
            out = PeriodicDelaunay2_vertex()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("PeriodicDelaunay2_vertex_vector indices must be "+
                            "integers, not {}".format(type(i)))


cdef class PeriodicDelaunay2_vertex_range:
    r"""Wrapper class for iterating over a range of triangulation vertices
    
    Args:
        vstart (PeriodicDelaunay2_vertex_iter): The starting vertex.
        vstop (PeriodicDelaunay2_vertex_iter): Final vertex that will end the 
            iteration.
        finite (:obj:`bool`, optional): If True, only finite verts are
            iterated over. Otherwise, all verts are iterated over. Defaults 
            to False.

    Attributes:
        x (PeriodicDelaunay2_vertex_iter): The current vertex.
        xstop (PeriodicDelaunay2_vertex_iter): Final vertex that will end the 
            iteration.
        finite (bool): If True, only finite verts are iterater over. Otherwise 
            all verts are iterated over.

    """
    cdef PeriodicDelaunay2_vertex_iter x
    cdef PeriodicDelaunay2_vertex_iter xstop
    cdef pybool finite
    def __cinit__(self, PeriodicDelaunay2_vertex_iter xstart, 
                  PeriodicDelaunay2_vertex_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.T.is_infinite(self.x.x):
                self.x.increment()
        cdef PeriodicDelaunay2_vertex out
        if self.x != self.xstop:
            out = self.x.vertex
            self.x.increment()
            return out
        else:
            raise StopIteration()


cdef class PeriodicDelaunay2_edge:
    r"""Wrapper class for a triangulation edge.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
            object that this edge belongs to.
        x (:obj:`PeriodicDelaunay_with_info_2[info_t].Edge`): C++ edge 
            object. Direct interaction with this object is not recommended.

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef PeriodicDelaunay_with_info_2[info_t].Edge x

    cdef void assign(self, PeriodicDelaunay_with_info_2[info_t] *T,
                     PeriodicDelaunay_with_info_2[info_t].Edge x):
        r"""Assign C++ objects to attributes.

        Args:
            T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
                object that this edge belongs to.
            x (:obj:`PeriodicDelaunay_with_info_2[info_t].Edge`): C++ edge 
                object. Direct interaction with this object is not recommended.

        """
        self.T = T
        self.x = x

    @staticmethod
    def from_cell(PeriodicDelaunay2_cell c, int i):
        r"""Construct an edges from a cell and index of the vertex opposite the 
        edge.

        Args:
            c (PeriodicDelaunay2_cell): Cell
            i (int): Index of vertex opposite the desired edge in c.

        Returns:
            PeriodicDelaunay2_edge: Edge incident to c and opposite vertex i of 
                c.

        """
        cdef PeriodicDelaunay2_edge out = PeriodicDelaunay2_edge()
        cdef PeriodicDelaunay_with_info_2[info_t].Edge e
        e = PeriodicDelaunay_with_info_2[info_t].Edge(c.x, i)
        out.assign(c.T, e)
        return out

    def __repr__(self):
        return "PeriodicDelaunay2_edge[{},{}]".format(repr(self.vertex1), 
                                              repr(self.vertex2))

    def __richcmp__(PeriodicDelaunay2_edge self, 
                    PeriodicDelaunay2_edge solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    property has_offset:
        r""":obj:`bool`: True if any of the incident vertices has a periodic 
        offset (not including any cell offset. False otherwise."""
        def __get__(self):
            return <pybool>self.T.has_offset(self.x)

    def is_infinite(self):
        r"""Determine if the edge is incident to the infinite vertex.

        Returns:
            bool: True if the edge is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def vertex(self, int i):
        r"""Return a vertex incident to this edge.

        Args:
            i (int): Index of vertex incident to this edge.

        Returns:
            PeriodicDelaunay2_vertex: Vertex i of this edge.

        """
        if (i % 2) == 0:
            return self.vertex1
        else:
            return self.vertex2

    def point(self, int i):
        r"""Return the (x, y) coordinates of the ith vertex incident to this 
        edge including the periodic offset to provide an unwrapped triangle.

        Args:
            i (int): Index of vertex incident to this edge.

        Returns:
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
                the vertex including the periodic offset.

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
        self.T.point(self.x, i, &out[0])
        return out

    def periodic_point(self, int i):
        r"""Return the (x, y) coordinates of the ith vertex incident to this 
        edge, not including the periodic offset.

        Args:
            i (int): Index of vertex incident to this edge.

        Returns:
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
                the vertex including the periodic offset.

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
        self.T.periodic_point(self.x, i, &out[0])
        return out

    def periodic_offset(self, int i):
        r"""Return the number of wrappings in (x, y) applied to the ith vertex 
        of this edge.

        Args:
            i (int): Index of vertex incident to this edge.

        Returns:
            :obj:`ndarray` of :obj:`int32`: The number of wrappings in (x, y)
                applied to the vertex.

        """
        cdef np.ndarray[np.int32_t] out = np.zeros(2, 'int32')
        self.T.periodic_offset(self.x, i, &out[0])
        return out

    property vertex1:
        r"""PeriodicDelaunay2_vertex: The first vertex in the edge."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_2[info_t].Vertex x = self.x.v1()
            cdef PeriodicDelaunay2_vertex out = PeriodicDelaunay2_vertex()
            out.assign(self.T, x)
            return out

    property vertex2:
        r"""PeriodicDelaunay2_vertex: The second vertex in the edge."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_2[info_t].Vertex x = self.x.v2()
            cdef PeriodicDelaunay2_vertex out = PeriodicDelaunay2_vertex()
            out.assign(self.T, x)
            return out

    property center:
        r""":obj:`ndarray` of float64: x,y coordinates of edge center."""
        def __get__(self):
            if self.is_infinite():
                return np.float('inf')*np.ones(2, 'float64')
            else:
                return (self.point(0) + self.point(1))/2.0

    property midpoint:
        r""":obj:`ndarray` of float64: x,y coordinates of edge midpoint."""
        def __get__(self):
            return self.center

    property length:
        r"""float64: The length of the edge. If infinite, -1 is returned"""
        def __get__(self):
            cdef np.float64_t out = self.T.length(self.x)
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this edge.

        Returns:
            PeriodicDelaunay2_vertex_vector: Iterator over vertices incident to 
                this edge.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        # it.push_back(self.x.v1())
        # it.push_back(self.x.v2())
        cdef PeriodicDelaunay2_vertex_vector out
        out = PeriodicDelaunay2_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this edge.

        Returns:
            PeriodicDelaunay2_edge_vector: Iterator over edges incident to this 
                edge.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef PeriodicDelaunay2_edge_vector out = PeriodicDelaunay2_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this edge.

        Returns:
            PeriodicDelaunay2_cell_vector: Iterator over cells incident to this 
                edge.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef PeriodicDelaunay2_cell_vector out = PeriodicDelaunay2_cell_vector()
        out.assign(self.T, it)
        return out

    def flip(self):
        r"""Flip this edge to the other diagonal of the quadrilateral formed by 
        the two cells incident to this edge after first testing that the edge 
        can be flipped.

        Returns:
            bool: True if the edge could be flipped, False otherwise.

        """
        self.T.updated = <cbool>True
        return self.T.flip(self.x)

    def flip_flippable(self):
        r"""Flip this edge to the other diagonal of the quadrilateral formed by 
        the two cells incident to this edge. The edge is assumed flippable to 
        save time.
        """
        self.T.updated = <cbool>True
        self.T.flip_flippable(self.x)

cdef class PeriodicDelaunay2_edge_iter:
    r"""Wrapper class for a triangulation edge iterator.

    Args:
        T (PeriodicDelaunay2): Triangulation that this edge belongs to.
        edge (:obj:`str`, optional): String specifying the edge that 
            should be referenced. Valid options include:
                'all_begin': The first edge in an iteration over all edges.
                'all_end': The last edge in an iteration over all edges.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
            object that this edge belongs to.
        x (:obj:`PeriodicDelaunay_with_info_2[info_t].All_edges_iter`): C++ edge 
            iteration object. Direct interaction with this object is not 
            recommended.

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef PeriodicDelaunay_with_info_2[info_t].All_edges_iter x
    
    def __cinit__(self, PeriodicDelaunay2 T, str edge = None):
        self.T = T.T
        if edge == 'all_begin':
            self.x = self.T.all_edges_begin()
        elif edge == 'all_end':
            self.x = self.T.all_edges_end()

    def __richcmp__(PeriodicDelaunay2_edge_iter self, 
                    PeriodicDelaunay2_edge_iter solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def increment(self):
        r"""Advance to the next edge in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous edge in the triangulation."""
        predecrement(self.x)

    property edge:
        r"""PeriodicDelaunay2_edge: The corresponding edge object."""
        def __get__(self):
            cdef PeriodicDelaunay2_edge out = PeriodicDelaunay2_edge()
            out.assign(self.T, 
                       PeriodicDelaunay_with_info_2[info_t].Edge(self.x))
            return out

cdef class PeriodicDelaunay2_edge_vector:
    r"""Wrapper class for a vector of edges.

    Attributes: 
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended. 
        v (:obj:`vector[PeriodicDelaunay_with_info_2[info_t].Edge]`): Vector of 
            C++ edges.
        n (int): The number of edges in the vector. 
        i (int): The index of the currect edge.

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef vector[PeriodicDelaunay_with_info_2[info_t].Edge] v
    cdef int n
    cdef int i

    cdef void assign(self, PeriodicDelaunay_with_info_2[info_t] *T,
                     vector[PeriodicDelaunay_with_info_2[info_t].Edge] v):
        r"""Assign C++ attributes. 

        Args: 
            T (:obj:`Delaunay_with_info_3[info_t]`): C++ triangulation object. 
                Direct interaction with this object is not recommended. 
            v (:obj:`vector[Delaunay_with_info_3[info_t].Edge]`): Vector of 
                C++ edges. 

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay2_edge out
        if self.i < self.n:
            out = PeriodicDelaunay2_edge()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef PeriodicDelaunay2_edge out
        if isinstance(i, int):
            out = PeriodicDelaunay2_edge()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("PeriodicDelaunay2_edge_vector indices must be "+
                            "integers, not {}".format(type(i)))


cdef class PeriodicDelaunay2_edge_range:
    r"""Wrapper class for iterating over a range of triangulation edges.
    
    Args:
        vstart (PeriodicDelaunay2_edge_iter): The starting edge.
        vstop (PeriodicDelaunay2_edge_iter): Final edge that will end the 
            iteration.
        finite (:obj:`bool`, optional): If True, only finite edges are
            iterated over. Otherwise, all edges are iterated over. Defaults 
            to False.

    Attributes:
        x (PeriodicDelaunay2_edge_iter): The current edge.
        xstop (PeriodicDelaunay2_edge_iter): Final edge that will end the 
            iteration.
        finite (bool): If True, only finite edges are iterater over. Otherwise 
            all edges are iterated over.

    """
    cdef PeriodicDelaunay2_edge_iter x
    cdef PeriodicDelaunay2_edge_iter xstop
    cdef pybool finite
    def __cinit__(self, PeriodicDelaunay2_edge_iter xstart, 
                  PeriodicDelaunay2_edge_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.T.is_infinite(self.x.x):
                self.x.increment()
        cdef PeriodicDelaunay2_edge out
        if self.x != self.xstop:
            out = self.x.edge
            self.x.increment()
            return out
        else:
            raise StopIteration()


cdef class PeriodicDelaunay2_cell:
    r"""Wrapper class for a triangulation cell.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
            object that this cell belongs to.
        x (:obj:`PeriodicDelaunay_with_info_2[info_t].Cell`): C++ cell object.
            Direct interaction with this object is not recommended.

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef PeriodicDelaunay_with_info_2[info_t].Cell x
    
    cdef void assign(self, PeriodicDelaunay_with_info_2[info_t] *T,
                     PeriodicDelaunay_with_info_2[info_t].Cell x):
        r"""Assign C++ objects to attributes.

        Args:
            T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
                object that this edge belongs to.
            x (:obj:`PeriodicDelaunay_with_info_2[info_t].Cell`): C++ cell 
                object. Direct interaction with this object is not recommended.

        """
        self.T = T
        self.x = x

    def __richcmp__(PeriodicDelaunay2_cell self, 
                    PeriodicDelaunay2_cell solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def __repr__(self):
        return "PeriodicDelaunay2_cell[{},{},{}]".format(repr(self.vertex(0)),
                                                 repr(self.vertex(1)),
                                                 repr(self.vertex(2)))

    property has_offset:
        r""":obj:`bool`: True if any of the incident vertices has a periodic 
        offset (not including any cell offset. False otherwise."""
        def __get__(self):
            return <pybool>self.T.has_offset(self.x)

    def is_infinite(self):
        r"""Determine if the cell is incident to the infinite vertex.

        Returns:
            bool: True if the cell is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def mirror_index(self, int i):
        r"""Get the index of this cell with respect to its ith neighbor.

        Args:
            i (int): Index of neighbor that should be used to determine the 
                mirrored index.

        Returns:
            int: Index of this cell with respect to its ith neighbor.

        """
        cdef int out = self.T.mirror_index(self.x, i)
        return out

    def mirror_vertex(self, int i):
        r"""Get the vertex of this cell's ith neighbor that is opposite to this 
        cell.

        Args:
            i (int): Index of neighbor that should be used to determine the
                mirrored vertex.

        Returns:
            PeriodicDelaunay2_vertex: Vertex in the ith neighboring cell of this 
                cell that is opposite to this cell.

        """
        cdef PeriodicDelaunay_with_info_2[info_t].Vertex vc
        vc = self.T.mirror_vertex(self.x, i)
        cdef PeriodicDelaunay2_vertex out = PeriodicDelaunay2_vertex()
        out.assign(self.T, vc)
        return out

    def edge(self, int i):
        r"""Find the edge incident to this cell that is opposite the ith vertex.

        Args:
            i (int): The index of the vertex opposite the edge that should be 
                returned.

        Returns:
            PeriodicDelaunay2_edge: Edge incident to this cell and opposite 
                vertex i.

        """
        return PeriodicDelaunay2_edge.from_cell(self, i)

    def vertex(self, int i):
        r"""Find the ith vertex that is incident to this cell.

        Args:
            i (int): The index of the vertex that should be returned.

        Returns:
            PeriodicDelaunay2_vertex: The ith vertex incident to this cell.

        """
        cdef PeriodicDelaunay_with_info_2[info_t].Vertex v
        v = self.x.vertex(i)
        cdef PeriodicDelaunay2_vertex out = PeriodicDelaunay2_vertex()
        out.assign(self.T, v)
        return out

    def point(self, int i):
        r"""Return the (x, y) coordinates of the ith vertex incident to this 
        cell including the periodic offset to provide and unwrapped triangle.

        Args:
            i (int): Index of vertex incident to this cell.

        Returns:
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
                the vertex including the periodic offset.

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
        self.T.point(self.x, i, &out[0])
        return out

    def periodic_point(self, int i):
        r"""Return the (x, y) coordinates of the ith vertex incident to this 
        cell, not including the periodic offset.

        Args:
            i (int): Index of vertex incident to this cell.

        Returns:
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
                the vertex including the periodic offset.

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
        self.T.periodic_point(self.x, i, &out[0])
        return out

    def periodic_offset(self, int i):
        r"""Return the number of wrappings in (x, y) applied to this vertex.

        Args:
            i (int): Index of vertex incident to this cell.

        Returns:
            :obj:`ndarray` of :obj:`int`: The number of wrappings in (x,y) 
                applied to the vertex.

        """
        cdef np.ndarray[np.int32_t] out = np.zeros(2, 'int32')
        self.T.periodic_offset(self.x, i, &out[0])
        return out

    def has_vertex(self, PeriodicDelaunay2_vertex v, 
                   pybool return_index = False):
        r"""Determine if a vertex belongs to this cell.

        Args:
            v (PeriodicDelaunay2_vertex): Vertex to test ownership for.
            return_index (:obj:`bool`, optional): If True, the index of the 
                vertex within the cell is returned in the event that it is a 
                vertex of the cell. Otherwise, the index is not returned.

        Returns:
            bool: True if the vertex is part of the cell, False otherwise. In 
                the event that `return_index = True` and the vertex is a part of 
                the cell, an integer specifying the index of the vertex within 
                the cell will be returned instead.

        """
        cdef int i = -1
        cdef cbool out
        if return_index:
            out = self.x.has_vertex(v.x, &i)
            return i
        else:
            out = self.x.has_vertex(v.x)
            return <pybool>out

    def ind_vertex(self, PeriodicDelaunay2_vertex v):
        r"""Determine the index of a vertex within a cell.

        Args:
            v (PeriodicDelaunay2_vertex): Vertex to find index for.
        
        Returns:
            int: Index of vertex within the cell.

        """
        return self.x.ind(v.x)

    def neighbor(self, int i):
        r"""Find the neighboring cell opposite the ith vertex of this cell. 

        Args:
            i (int): The index of the neighboring cell that should be returned.

        Returns:
            PeriodicDelaunay2_cell: The neighboring cell opposite the ith 
                vertex.

        """
        cdef PeriodicDelaunay_with_info_2[info_t].Cell v
        v = self.x.neighbor(i)
        cdef PeriodicDelaunay2_cell out = PeriodicDelaunay2_cell()
        out.assign(self.T, v)
        return out

    def has_neighbor(self, PeriodicDelaunay2_cell v, 
                     pybool return_index = False):
        r"""Determine if a cell is a neighbor to this cell.

        Args:
            v (PeriodicDelaunay2_cell): Cell to test as a neighbor.
            return_index (:obj:`bool`, optional): If True, the index of the 
                neighbor within the cell is returned in the event that it is a 
                neighbor of the cell. Otherwise, the index is not returned.

        Returns:
            bool: True if the given cell is a neighbor, False otherwise. In 
                the event that `return_index = True` and the v is a neighbor of 
                this cell, an integer specifying the index of the neighbor to 
                will be returned instead.

        """
        cdef int i = -1
        cdef cbool out = self.x.has_neighbor(v.x, &i)
        if out and return_index:
            return i
        else:
            return <pybool>out

    def ind_neighbor(self, PeriodicDelaunay2_cell v):
        r"""Determine the index of a neighboring cell.

        Args:
            v (PeriodicDelaunay2_cell): Neighboring cell to find index for.
        
        Returns:
            int: Index of vertex opposite to neighboring cell.

        """
        return self.x.ind(v.x)

    def set_vertex(self, int i, PeriodicDelaunay2_vertex v):
        r"""Set the ith vertex of this cell.

        Args:
            i (int): Index of this cell's vertex that should be set.
            v (Delauany2_vertex): Vertex to set ith vertex of this cell to.

        """
        self.T.updated = <cbool>True
        self.x.set_vertex(i, v.x)

    def set_vertices(self, PeriodicDelaunay2_vertex v1, 
                     PeriodicDelaunay2_vertex v2, 
                     PeriodicDelaunay2_vertex v3):
        r"""Set this cell's vertices.

        Args:
            v1 (PeriodicDelaunay2_vertex): 1st vertex of cell.
            v2 (PeriodicDelaunay2_vertex): 2nd vertex of cell.
            v3 (PeriodicDelaunay2_vertex): 3rd vertex of cell.

        """
        self.T.updated = <cbool>True
        self.x.set_vertices(v1.x, v2.x, v3.x)

    def reset_vertices(self):
        r"""Reset all of this cell's vertices."""
        self.T.updated = <cbool>True
        self.x.set_vertices()

    def set_neighbor(self, int i, PeriodicDelaunay2_cell n):
        r"""Set the ith neighboring cell of this cell.

        Args:
            i (int): Index of this cell's neighbor that should be set.
            n (PeriodicDelaunay2_cell): Cell to set ith neighbor of this cell 
                to.

        """
        self.T.updated = <cbool>True
        self.x.set_neighbor(i, n.x)

    def set_neighbors(self, PeriodicDelaunay2_cell c1, 
                      PeriodicDelaunay2_cell c2, 
                      PeriodicDelaunay2_cell c3):
        r"""Set this cell's neighboring cells.

        Args:
            c1 (PeriodicDelaunay2_cell): 1st neighboring cell.
            c2 (PeriodicDelaunay2_cell): 2nd neighboring cell.
            c3 (PeriodicDelaunay2_cell): 3rd neighboring cell.

        """
        self.T.updated = <cbool>True
        self.x.set_neighbors(c1.x, c2.x, c3.x)

    def reset_neighbors(self):
        r"""Reset all of this cell's neighboring cells."""
        self.T.updated = <cbool>True
        self.x.set_neighbors()

    def reorient(self):
        r"""Change the vertex order so that ccw and cw are switched."""
        self.T.updated = <cbool>True
        self.x.reorient()

    def ccw_permute(self):
        r"""Bring the last vertex to the front of the vertex order."""
        self.T.updated = <cbool>True
        self.x.ccw_permute()
        
    def cw_permute(self):
        r"""Put the 1st vertex at the end of the vertex order."""
        self.T.updated = <cbool>True
        self.x.cw_permute()

    property center:
        r""":obj:`ndarray` of float64: x,y coordinates of cell center."""
        def __get__(self):
            if self.is_infinite():
                return np.float('inf')*np.ones(2, 'float64')
            else:
                return (self.point(0) + \
                        self.point(1) + \
                        self.point(2))/3.0

    property circumcenter:
        r""":obj:`ndarray` of float64: x,y coordinates of cell circumcenter."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
            self.T.circumcenter(self.x, &out[0])
            return out

    property dimension:
        r"""int: The number of dimensions that this cell occupies."""
        def __get__(self):
            return self.x.dimension()

    def incident_vertices(self):
        r"""Find vertices that are incident to this cell.

        Returns:
            PeriodicDelaunay2_vertex_vector: Iterator over vertices incident to 
                this cell.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef PeriodicDelaunay2_vertex_vector out
        out = PeriodicDelaunay2_vertex_vector()
        out.assign(self.T, it)
        return out
        
    def incident_edges(self):
        r"""Find edges that are incident to this cell.

        Returns:
            PeriodicDelaunay2_edge_vector: Iterator over edges incident to this 
                cell.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef PeriodicDelaunay2_edge_vector out = PeriodicDelaunay2_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this cell.

        Returns:
            PeriodicDelaunay2_cell_vector: Iterator over cells incident to this 
                cell.

        """
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef PeriodicDelaunay2_cell_vector out = PeriodicDelaunay2_cell_vector()
        out.assign(self.T, it)
        return out

    def side(self, np.ndarray[np.float64_t, ndim=1] p):
        r"""Determine if a point is inside, outside or on this cell's edge.

        Args:
            p (np.ndarray of np.float64): x,y coordinates.
        
        Returns:
            int: -1 if p is inside this cell, 0 if p is on one of this cell's 
                vertices or edges, and 1 if p is outside this cell.

        """
        return self.T.oriented_side(self.x, &p[0])

    def side_of_circle(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Determine where a point is with repect to this cell's 
            circumcircle.

        Args:
            pos (:obj:`ndarray` of np.float64): x,y coordinates.

        Returns:
            int: -1, 0, or 1 if `pos` is within, on, or inside this cell's 
                circumcircle respectively.

        """
        return self.T.side_of_oriented_circle(self.x, &pos[0])


cdef class PeriodicDelaunay2_cell_iter:
    r"""Wrapper class for a triangulation cell.

    Args:
        T (Delaunay2): Triangulation that this cell belongs to.
        cell (:obj:`str`, optional): String specifying the cell that 
            should be referenced. Valid options include:
                'all_begin': The first cell in an iteration over all cells.
                'all_end': The last cell in an iteration over all cells.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ Triangulation 
            object that this cell belongs to.
        x (:obj:`PeriodicDelaunay_with_info_2[info_t].All_cells_iter`): C++ cell 
            object. Direct interaction with this object is not recommended.

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef PeriodicDelaunay_with_info_2[info_t].All_cells_iter x
    
    def __cinit__(self, PeriodicDelaunay2 T, str cell = None):
        self.T = T.T
        if cell == 'all_begin':
            self.x = self.T.all_cells_begin()
        elif cell == 'all_end':
            self.x = self.T.all_cells_end()

    def __richcmp__(PeriodicDelaunay2_cell_iter self, 
                    PeriodicDelaunay2_cell_iter solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def increment(self):
        r"""Advance to the next cell in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous cell in the triangulation."""
        predecrement(self.x)

    property cell:
        r"""PeriodicDelaunay2_cell: Corresponding cell object."""
        def __get__(self):
            cdef PeriodicDelaunay2_cell out = PeriodicDelaunay2_cell()
            out.T = self.T
            out.x = PeriodicDelaunay_with_info_2[info_t].Cell(self.x)
            return out


cdef class PeriodicDelaunay2_cell_vector:
    r"""Wrapper class for a vector of cells. 

    Attributes: 
        T (:obj:`Delaunay_with_info_3[info_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 
        v (:obj:`vector[Delaunay_with_info_3[info_t].Cell]`): Vector of C++ 
            cells. 
        n (int): The number of cells in the vector. 
        i (int): The index of the currect cell. 

    """
    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef vector[PeriodicDelaunay_with_info_2[info_t].Cell] v
    cdef int n
    cdef int i

    cdef void assign(self, PeriodicDelaunay_with_info_2[info_t] *T,
                     vector[PeriodicDelaunay_with_info_2[info_t].Cell] v):
        r"""Assign C++ attributes. 

        Args: 
            T (:obj:`Delaunay_with_info_3[info_t]`): C++ triangulation object. 
                Direct interaction with this object is not recommended. 
            v (:obj:`vector[Delaunay_with_info_3[info_t].Cell]`): Vector of 
                C++ cells. 

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay2_cell out
        if self.i < self.n:
            out = PeriodicDelaunay2_cell()
            out.T = self.T
            out.x = self.v[self.i]
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef PeriodicDelaunay2_cell out
        if isinstance(i, int):
            out = PeriodicDelaunay2_cell()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("PeriodicDelaunay2_cell_vector indices must be "+
                            "integers, not {}".format(type(i)))


cdef class PeriodicDelaunay2_cell_range:
    r"""Wrapper class for iterating over a range of triangulation cells.
    
    Args:
        xstart (PeriodicDelaunay2_cell_iter): The starting cell.
        xstop (PeriodicDelaunay2_cell_iter): Final cell that will end the 
            iteration.
        finite (:obj:`bool`, optional): If True, only finite cells are
            iterated over. Otherwise, all cells are iterated over. Defaults 
            to False.

    Attributes:
        x (PeriodicDelaunay2_cell_iter): The current cell.
        xstop (PeriodicDelaunay2_cell_iter): Final cell that will end the 
            iteration.
        finite (bool): If True, only finite cells are iterated over. Otherwise, 
            all cells are iterated over.

    """
    cdef PeriodicDelaunay2_cell_iter x
    cdef PeriodicDelaunay2_cell_iter xstop
    cdef pybool finite
    def __cinit__(self, PeriodicDelaunay2_cell_iter xstart, 
                  PeriodicDelaunay2_cell_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.T.is_infinite(self.x.x):
                self.x.increment()
        cdef PeriodicDelaunay2_cell out
        if self.x != self.xstop:
            out = self.x.cell
            self.x.increment()
            return out
        else:
            raise StopIteration()


cdef class PeriodicDelaunay2:
    r"""Wrapper class for a 2D Delaunay triangulation.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_2[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended.
        n (int): The number of points inserted into the triangulation.
        n_per_insert (list of int): The number of points inserted at each 
            insert.

    """

    cdef PeriodicDelaunay_with_info_2[info_t] *T
    cdef readonly int n
    cdef public object n_per_insert
    cdef readonly pybool _locked
    cdef public object _cache_to_clear_on_update

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray[np.float64_t] left_edge = None,
                  np.ndarray[np.float64_t] right_edge = None):
        cdef np.ndarray[np.float64_t] domain = np.empty(2*2, 'float64')
        if left_edge is None or right_edge is None:
            domain[:2] = [0,0]
            domain[2:] = [1,1]
        else:
            domain[:2] = left_edge
            domain[2:] = right_edge
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T = new PeriodicDelaunay_with_info_2[info_t](&domain[0])
        self.n = 0
        self.n_per_insert = []
        self._locked = False
        self._cache_to_clear_on_update = {}

    def is_equivalent(PeriodicDelaunay2 self, PeriodicDelaunay2 solf):
        r"""Determine if two triangulations are equivalent. Currently only 
        checks that the triangulations have the same numbers of vertices, cells, 
        and edges.

        Args: 
            solf (:class:`cgal4py.delaunay.PeriodicDelaunay2`): Triangulation 
                this one should be compared to.

        Returns: 
            bool: True if the two triangulations are equivalent. 

        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.is_equal(dereference(solf.T))
        return <pybool>(out)

    @classmethod
    def from_file(cls, fname):
        r"""Create a triangulation from one saved to a file.

        Args:
            fname (str): Full path to file where triangulation is saved.

        Returns:
            :class:`cgal4py.delaunay.PeriodicDelaunay2`: Triangulation
                constructed from saved information.

        """
        T = cls()
        T.read_from_file(fname)
        return T

    @classmethod
    def from_serial(cls, *args):
        r"""Create a triangulation from serialized information.

        Args:
            *args: All arguments are passed to 
                :meth:`cgal4py.delaunay.PeriodicDelaunay2.deserialize`.

        Returns:
            :class:`cgal4py.delaunay.PeriodicDelaunay2`: Triangulation 
                constructed from deserialized information.

        """
        T = cls()
        T.deserialize(*args)
        return T

    @classmethod
    def from_serial_buffer(cls, *args, **kwargs):
        r"""Create a triangulation from serialized information in a buffer.

        Args:
            See :meth:`cgal4py.delaunay.PeriodicDelaunay2.deserialize_from_buffer`.

        Returns:
            :class:`cgal4py.delaunay.PeriodicDelaunay2`: Triangulation
                constructed from deserialized information.

        """
        T = cls()
        T.deserialize_from_buffer(*args, **kwargs)
        return T

    def serialize_to_buffer(self, buf, pos=None):
        r"""Write serialized triangulation to a buffer.

        Args:
            buf (file): File buffer.
            pos (np.ndarray, optional): Positions to be written. If not
                provided, positions are not included and must be provided
                during any subsequent read. Defaults to None.

        """
        cdef np.ndarray[np_info_t, ndim=2] cells
        cdef np.ndarray[np_info_t, ndim=2] neighbors
        cdef info_t idx_inf
        cells, neighbors, idx_inf = self.serialize()
        cdef str ifmt, ffmt
        ifmt = cells.dtype.char
        ffmt = 'd'
        if pos is not None:
            ffmt = pos.dtype.char
        if PY_MAJOR_VERSION == 2:
            buf.write(struct.pack('cc', ffmt, ifmt))
        else:
            buf.write(struct.pack('cc', ffmt.encode('ascii'),
                                  ifmt.encode('ascii')))
        buf.write(struct.pack(ifmt, idx_inf))
        if pos is not None:
            buf.write(struct.pack(2*ifmt, pos.shape[0], pos.shape[1]))
            buf.write(pos.tobytes())
        buf.write(struct.pack(2*ifmt, cells.shape[0], cells.shape[1]))
        buf.write(cells.tobytes())
        buf.write(neighbors.tobytes())

    def deserialize_from_buffer(self, buf, pos=None):
        r"""Read a serialized triangulation from the buffer.

        Args:
            buf (file): File buffer.
            pos (np.ndarray, optional): Positions to be used for deserializing
                the triangulation if the positions are not in the file. If
                not provided, the file is assumed to contain the positions.
                Defaults to None.

       """
        cdef int ndim = 2
        cdef np.ndarray[np_info_t, ndim=2] cells
        cdef np.ndarray[np_info_t, ndim=2] neighbors
        cdef info_t idx_inf
        cdef str ifmt, ffmt
        cdef int isiz, fsiz
        cdef int nx, ny
        cdef bytes ibfmt, fbfmt
        if PY_MAJOR_VERSION == 2:
            (ffmt, ifmt) = struct.unpack('cc', buf.read(struct.calcsize('cc')))
        else:
            (fbfmt, ibfmt) = struct.unpack('cc', buf.read(struct.calcsize('cc')))
            ffmt = fbfmt.decode()
            ifmt = ibfmt.decode()
        fsiz = struct.calcsize(ffmt)
        isiz = struct.calcsize(ifmt)
        (idx_inf,) = struct.unpack(ifmt, buf.read(isiz))
        nx, ny = struct.unpack(2*ifmt, buf.read(2*isiz))
        assert(ny == ndim)
        pos = np.frombuffer(
            bytearray(buf.read(nx*ny*fsiz)), dtype=np.dtype(ffmt),
            count=nx*ny).reshape(nx, ny)
        nx, ny = struct.unpack(2*ifmt, buf.read(2*isiz))
        assert(ny == (ndim+1))
        cells = np.frombuffer(
            bytearray(buf.read(nx*ny*isiz)), dtype=np.dtype(ifmt),
            count=nx*ny).reshape(nx, ny)
        neigh = np.frombuffer(
            bytearray(buf.read(nx*ny*isiz)), dtype=np.dtype(ifmt),
            count=nx*ny).reshape(nx, ny)
        self.deserialize(pos, cells, neigh, idx_inf)

    def _lock(self):
        self._locked = True
    def _unlock(self):
        self._locked = False
    def _set_updated(self):
        self.T.updated = <cbool>True
    def _unset_updated(self):
        self.T.updated = <cbool>False

    def _update_tess(self):
        if self.T.updated:
            self._cache_to_clear_on_update.clear()
            self.T.updated = <cbool>False

    @staticmethod
    def _update_to_tess(func):
        def wrapped_func(solf, *args, **kwargs):
            solf._lock()
            out = func(solf, *args, **kwargs)
            solf._unlock()
            solf._set_updated()
            solf._update_tess()
            return out
        return wrapped_func

    @staticmethod
    def _dependent_property(fget):
        attr = '_'+fget.__name__
        def wrapped_fget(solf):
            if solf._locked:
                raise RuntimeError("Cannot get dependent property "+
                                   "'{}'".format(attr)+
                                   " while triangulation is locked.")
            solf._update_tess()
            if attr not in solf._cache_to_clear_on_update:
                solf._cache_to_clear_on_update[attr] = fget(solf)
            return solf._cache_to_clear_on_update[attr]
        return property(wrapped_fget, None, None, fget.__doc__)

    def is_valid(self):
        r"""Determine if the triangulation is a valid Delaunay triangulation.

        Returns:
            bool: True if the triangulation is valid, False otherwise.
        
        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.is_valid()
        return <pybool>out

    def write_to_file(self, fname):
        r"""Write the serialized tessellation information to a file.

        Args:
            fname (str): The full path to the file that the tessellation should 
                be written to.

        """
        cdef char* cfname
        cdef bytes pyfname
        if PY_MAJOR_VERSION < 3:
            cfname = fname
        else:
            pyfname = bytes(fname, encoding="ascii")
            cfname = pyfname
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.write_to_file(cfname)

    @_update_to_tess
    def read_from_file(self, fname):
        r"""Read serialized tessellation information from a file.

        Args:
            fname (str): The full path to the file that the tessellation should 
                be read from.

        """
        cdef char* cfname
        cdef bytes pyfname
        if PY_MAJOR_VERSION < 3:
            cfname = fname
        else:
            pyfname = bytes(fname, encoding="ascii")
            cfname = pyfname
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.read_from_file(cfname)
        self.n = self.T.num_finite_verts()
        self.n_per_insert.append(self.n)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def serialize(self, pybool sort = False):
        r"""Serialize triangulation.

        Args:
            sort (bool, optional): If True, cells info is sorted so that the 
                verts are in descending order for each cell and ascending order 
                overall. Defaults to False.
        
        Returns: 
            tuple containing:
                domain (np.ndarray of float64): Min/max bounds of domain in each 
                    dimension (e.g. [xmin, ymin, xmax, ymax]).
                cover (np.ndarray of int32): Number of times points are 
                    replicated in each dimension to allow wrapping.
                cells (np.ndarray of info_t): (n,m) Indices for m vertices in 
                    each of the n cells. A value of np.iinfo(np_info).max 
                    indicates the infinite vertex.
                neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of 
                    the m neighbors to each of the n cells.
                offsets (np.ndarray of int32): (n,m) Offset of m vertices in 
                    each of the ncells.
                idx_inf (info_t): Value representing the infinite vertex and or 
                    a missing neighbor.
        
        """
        cdef info_t n, m, i
        cdef int32_t d, j
        cdef np.ndarray[np.float64_t, ndim=1] domain
        cdef np.ndarray[np.int32_t, ndim=1] cover
        cdef np.ndarray[np_info_t, ndim=2] cells
        cdef np.ndarray[np_info_t, ndim=2] neighbors
        cdef np.ndarray[np.int32_t, ndim=2] offsets
        # Initialize arrays based on properties
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(n == self.num_finite_verts)
        assert(m == self.num_cells)
        d = 2
        domain = np.zeros(2*d, np.float64)
        cover = self.num_sheets
        cells = np.zeros((m, d+1), np_info)
        neighbors = np.zeros((m, d+1), np_info)
        offsets = np.zeros((m, d+1), np.int32)
        # Serialize and convert to original vertex order
        cdef info_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_idxinfo[info_t](
                n, m, d, &domain[0], &cover[0], 
                &cells[0,0], &neighbors[0,0], &offsets[0,0])
        # Sort if desired
        # TODO: Sort offsets?
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[info_t](&cells[0,0], &neighbors[0,0], m, d+1)
        return domain, cover, cells, neighbors, offsets, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_int32(self, info_t max_info,
                                  np.ndarray[np.int32_t] idx,
                                  pybool sort=False):
        cdef int32_t n, m
        cdef int32_t d
        cdef np.ndarray[np.float64_t, ndim=1] domain
        cdef np.ndarray[np.int32_t, ndim=1] cover
        cdef np.ndarray[np.int32_t, ndim=2] cells
        cdef np.ndarray[np.int32_t, ndim=2] neighbors
        cdef np.ndarray[np.int32_t, ndim=2] offsets
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = 2
        domain = np.zeros(2*d, 'float64')
        cover = np.zeros(d, 'int32')
        cells = np.zeros((m, d+1), 'int32')
        neighbors = np.zeros((m, d+1), 'int32')
        offsets = np.zeros((m, d+1), 'int32')
        cdef int32_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[int32_t](
                n, m, d, &domain[0], &cover[0],
                &cells[0,0], &neighbors[0,0], &offsets[0,0],
                max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        offsets.resize(m, d+1, refcheck=False)
        # TODO: Sort offsets
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[int32_t](&cells[0,0], &neighbors[0,0], 
                                            m, d+1)
        return domain, cover, cells, neighbors, offsets, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_uint32(self, info_t max_info,
                                   np.ndarray[np.uint32_t] idx,
                                   pybool sort=False):
        cdef uint32_t n, m
        cdef int32_t d
        cdef np.ndarray[np.float64_t, ndim=1] domain
        cdef np.ndarray[np.int32_t, ndim=1] cover
        cdef np.ndarray[np.uint32_t, ndim=2] cells
        cdef np.ndarray[np.uint32_t, ndim=2] neighbors
        cdef np.ndarray[np.int32_t, ndim=2] offsets
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = 2
        domain = np.zeros(2*d, 'float64')
        cover = np.zeros(d, 'int32')
        cells = np.zeros((m, d+1), 'uint32')
        neighbors = np.zeros((m, d+1), 'uint32')
        offsets = np.zeros((m, d+1), 'int32')
        cdef uint32_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[uint32_t](
                n, m, d, &domain[0], &cover[0],
                &cells[0,0], &neighbors[0,0], &offsets[0,0],
                max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        offsets.resize(m, d+1, refcheck=False)
        # TODO: Sort offsets
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[uint32_t](&cells[0,0], &neighbors[0,0], 
                                             m, d+1)
        return domain, cover, cells, neighbors, offsets, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_int64(self, info_t max_info,
                                  np.ndarray[np.int64_t] idx,
                                  pybool sort=False):
        cdef int64_t n, m
        cdef int32_t d
        cdef np.ndarray[np.float64_t, ndim=1] domain
        cdef np.ndarray[np.int32_t, ndim=1] cover
        cdef np.ndarray[np.int64_t, ndim=2] cells
        cdef np.ndarray[np.int64_t, ndim=2] neighbors
        cdef np.ndarray[np.int32_t, ndim=2] offsets
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = 2
        domain = np.zeros(2*d, 'float64')
        cover = np.zeros(d, 'int32')
        cells = np.zeros((m, d+1), 'int64')
        neighbors = np.zeros((m, d+1), 'int64')
        offsets = np.zeros((m, d+1), 'int32')
        cdef int64_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[int64_t](
                n, m, d, &domain[0], &cover[0],
                &cells[0,0], &neighbors[0,0], &offsets[0,0],
                max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        offsets.resize(m, d+1, refcheck=False)
        # TODO: Sort offsets
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[int64_t](&cells[0,0], &neighbors[0,0], 
                                            m, d+1)
        return domain, cover, cells, neighbors, offsets, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_uint64(self, info_t max_info,
                                   np.ndarray[np.uint64_t] idx, 
                                   pybool sort=False):
        cdef uint64_t n, m
        cdef int32_t d
        cdef np.ndarray[np.float64_t, ndim=1] domain
        cdef np.ndarray[np.int32_t, ndim=1] cover
        cdef np.ndarray[np.uint64_t, ndim=2] cells
        cdef np.ndarray[np.uint64_t, ndim=2] neighbors
        cdef np.ndarray[np.int32_t, ndim=2] offsets
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = 2
        domain = np.zeros(2*d, 'float64')
        cover = np.zeros(d, 'int32')
        cells = np.zeros((m, d+1), 'uint64')
        neighbors = np.zeros((m, d+1), 'uint64')
        offsets = np.zeros((m, d+1), 'int32')
        cdef uint64_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[uint64_t](
                n, m, d, &domain[0], &cover[0],
                &cells[0,0], &neighbors[0,0], &offsets[0,0],
                max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        offsets.resize(m, d+1, refcheck=False)
        # TODO: Sort offsets
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[uint64_t](&cells[0,0], &neighbors[0,0], 
                                             m, d+1)
        return domain, cover, cells, neighbors, offsets, idx_inf

    def serialize_info2idx(self, max_info, idx, pybool sort = False):
        r"""Serialize triangulation, only including some vertices and 
        translating the indices.

        Args:
            max_info (info_t): Maximum value of info for verts that will be
                included in the serialization.
            idx (np.ndarray of I): Indices that should be used to map from 
                vertex info.
            sort (bool, optional): If True, cells info is sorted so that the 
                verts are in descending order for each cell and ascending order 
                overall. Defaults to False.
        
        Returns: 
            tuple containing:
                domain (np.ndarray of float64): Min/max bounds of domain in each 
                    dimension (e.g. [xmin, ymin, xmax, ymax]).
                cover (np.ndarray of int32): Number of times points are 
                    replicated in each dimension to allow wrapping.
                cells (np.ndarray of I): (n,m) Indices for m vertices in 
                    each of the n cells. A value of np.iinfo(np_info).max 
                    indicates the infinite vertex.
                neighbors (np.ndarray of I): (n,m) Indices in `cells` of 
                    the m neighbors to each of the n cells.
                offsets (np.ndarray of int32): (n,m) Offset of m vertices in 
                    each of the ncells.
                idx_inf (I): Value representing the infinite vertex and or 
                    a missing neighbor.
        
        """
        if idx.dtype == np.int32:
            return self._serialize_info2idx_int32(<info_t>max_info, idx, sort)
        elif idx.dtype == np.uint32:
            return self._serialize_info2idx_uint32(<info_t>max_info, idx, sort)
        elif idx.dtype == np.int64:
            return self._serialize_info2idx_int64(<info_t>max_info, idx, sort)
        elif idx.dtype == np.uint64:
            return self._serialize_info2idx_uint64(<info_t>max_info, idx, sort)
        else:
            raise TypeError("idx.dtype = {} ".format(idx.dtype)+
                            "is not supported.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @_update_to_tess
    def deserialize(self, np.ndarray[np.float64_t, ndim=2] pos,
                    np.ndarray[np.float64_t, ndim=1] domain,
                    np.ndarray[np.int32_t, ndim=1] cover,
                    np.ndarray[np_info_t, ndim=2] cells,
                    np.ndarray[np_info_t, ndim=2] neighbors,
                    np.ndarray[np.int32_t, ndim=2] offsets,
                    info_t idx_inf):
        r"""Deserialize triangulation.

        Args:
            pos (np.ndarray of float64): Coordinates of points.
            domain (np.ndarray of float64): Min/max bounds of domain in each 
                dimension (e.g. [xmin, ymin, xmax, ymax]).
            cover (np.ndarray of int32): Number of times points are replicated 
                in each dimension to allow wrapping.
            cells (np.ndarray of info_t): (n,m) Indices for m vertices in each 
                of the n cells. A value of np.iinfo(np_info).max A value of 
                np.iinfo(np_info).max indicates the infinite vertex.
            neighbors (np.ndarray of info_t): (n,m) Indices in `cells` of the m
                neighbors to each of the n cells.
            offsets (np.ndarray of int32): (n,m) Offset of m vertices in each of 
                the ncells.
            idx_inf (info_t): Index indicating a vertex is infinite.

        """
        cdef info_t n = pos.shape[0]
        cdef info_t m = cells.shape[0]
        cdef int32_t d = neighbors.shape[1]-1
        if (n == 0) or (m == 0):
            return
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.deserialize_idxinfo[info_t](n, m, d, &domain[0], &cover[0],
                                               &pos[0,0], &cells[0,0], 
                                               &neighbors[0,0], &offsets[0,0], 
                                               idx_inf)
                                               
        self.n = n
        self.n_per_insert.append(n)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @_update_to_tess
    def deserialize_with_info(self, np.ndarray[np.float64_t, ndim=2] pos,
                              np.ndarray[np_info_t, ndim=1] info,
                              np.ndarray[np.float64_t, ndim=1] domain,
                              np.ndarray[np.int32_t, ndim=1] cover,
                              np.ndarray[np_info_t, ndim=2] cells,
                              np.ndarray[np_info_t, ndim=2] neighbors,
                              np.ndarray[np.int32_t, ndim=2] offsets,
                              info_t idx_inf):
        r"""Deserialize triangulation.

        Args:
            pos (np.ndarray of float64): Coordinates of points.
            info (np.ndarray of info_t): Info for points.
            domain (np.ndarray of float64): Min/max bounds of domain in each 
                dimension (e.g. [xmin, ymin, xmax, ymax]).
            cover (np.ndarray of int32): Number of times points are replicated 
                in each dimension to allow wrapping.
            cells (np.ndarray of info_t): (n,m) Indices for m vertices in each 
                of the n cells. A value of np.iinfo(np_info).max A value of 
                np.iinfo(np_info).max indicates the infinite vertex.
            neighbors (np.ndarray of info_t): (n,m) Indices in `cells` of the m
                neighbors to each of the n cells.
            offsets (np.ndarray of int32): (n,m) Offset of m vertices in each of 
                the ncells.
            idx_inf (info_t): Index indicating a vertex is infinite.

        """
        cdef info_t n = pos.shape[0]
        cdef info_t m = cells.shape[0]
        cdef int32_t d = neighbors.shape[1]-1
        if (n == 0) or (m == 0):
            return
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.deserialize[info_t](n, m, d, &domain[0], &cover[0], 
                                       &pos[0,0], &info[0],
                                       &cells[0,0], &neighbors[0,0], 
                                       &offsets[0,0], idx_inf)
        self.n = n
        self.n_per_insert.append(n)

    def plot(self, *args, **kwargs):
        r"""Plot the triangulation.

        Args:
            *args: All arguments are passed to :func:`plot.plot2D`.
            **kwargs: All keyword arguments are passed to :func:`plot.plot2D`.

        """
        plot.plot2D(self, *args, **kwargs)

    @_dependent_property
    def num_sheets(self):
        r"""np.ndarray of int32: The number of times the original domain is 
        replicated in each dimension to allow wrapping around periodic 
        boundaries."""
        cdef np.ndarray[np.int32_t] ns = np.empty(2,'int32')
        self.T.num_sheets(&ns[0])
        return ns
    @_dependent_property
    def num_sheets_total(self):
        r"""int: The number of times the original domain is replicated to allow 
        wrapping around periodic boundaries."""
        return self.T.num_sheets_total()
    @_dependent_property
    def num_finite_verts(self): 
        r"""int: The number of finite vertices in the triangulation."""
        return self.T.num_finite_verts()
    @_dependent_property
    def num_finite_edges(self): 
        r"""int: The number of finite edges in the triangulation."""
        return self.T.num_finite_edges()
    @_dependent_property
    def num_finite_cells(self): 
        r"""int: The number of finite cells in the triangulation."""
        return self.T.num_finite_cells()
    @_dependent_property
    def num_infinite_verts(self):
        r"""int: The number of infinite vertices in the triangulation."""
        return self.T.num_infinite_verts()
    @_dependent_property
    def num_infinite_edges(self):
        r"""int: The number of infinite edges in the triangulation."""
        return self.T.num_infinite_edges()
    @_dependent_property
    def num_infinite_cells(self): 
        r"""int: The number of infinite cells in the triangulation."""
        return self.T.num_infinite_cells()
    @_dependent_property
    def num_verts(self): 
        r"""int: The total number of vertices (Finite + infinite) in the 
        triangulation."""
        return self.T.num_verts()
    @_dependent_property
    def num_edges(self): 
        r"""int: The total number of edges (finite + infinite) in the 
        triangulation."""
        return self.T.num_edges()
    @_dependent_property
    def num_cells(self): 
        r"""int: The total number of cells (finite + infinite) in the 
        triangulation."""
        return self.T.num_cells()
    @_dependent_property
    def num_stored_verts(self): 
        r"""int: The total number of vertices (Finite + infinite) in the 
        triangulation including duplicates made to allow periodic 
        wrapping."""
        return self.T.num_stored_verts()
    @_dependent_property
    def num_stored_edges(self): 
        r"""int: The total number of edges (Finite + infinite) in the 
        triangulation including duplicates made to allow periodic 
        wrapping."""
        return self.T.num_stored_edges()
    @_dependent_property
    def num_stored_cells(self): 
        r"""int: The total number of cells (Finite + infinite) in the 
        triangulation including duplicates made to allow periodic 
        wrapping."""
        return self.T.num_stored_cells()

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_domain(self, np.ndarray[np.float64_t] left_edge,
                   np.ndarray[np.float64_t] right_edge):
        r"""Set the bounds on the periodic domain.

        Args:
            left_edge (:obj:`ndarray` of :obj:`float64`): Minimum bounds 
                on domain in each dimension.
            right_edge (:obj:`ndarray` of :obj:`float64`): Maximum bounds 
                on domain in each dimension.

        """
        cdef np.ndarray[np.float64_t] domain = np.empty(2*2, 'float64')
        domain[:2] = left_edge
        domain[2:] = right_edge
        self.T.set_domain(&domain[0])

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def insert(self, np.ndarray[double, ndim=2, mode="c"] pts not None):
        r"""Insert points into the triangulation.

        Args:
            pts (:obj:`ndarray` of :obj:`float64`): Array of 2D cartesian 
                points to insert into the triangulation.

        """
        global np_info, np_info_t
        if pts.shape[0] == 0:
            return
        cdef int Nold, Nnew, m
        Nold = self.n
        Nnew, m = pts.shape[0], pts.shape[1]
        if Nnew == 0:
            return
        assert(m == 2)
        cdef np.ndarray[np_info_t, ndim=1] idx
        idx = np.arange(Nold, Nold+Nnew).astype(np_info)
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.insert(&pts[0,0], &idx[0], <info_t>Nnew)
        self.n += Nnew
        self.n_per_insert.append(Nnew)

    @_update_to_tess
    def clear(self):
        r"""Removes all vertices and cells from the triangulation."""
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.clear()

    @_dependent_property
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def vertices(self):
        r"""ndarray: The x,y coordinates of the vertices"""
        cdef np.ndarray[np.float64_t, ndim=2] out
        out = np.zeros([self.n, 2], 'float64')
        if self.n == 0:
            return out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.info_ordered_vertices(&out[0,0])
        return out

    @_dependent_property
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def edges(self):
        r""":obj:`ndarray` of info_t: Vertex index pairs for edges."""
        global np_info, np_info_t
        cdef np.ndarray[np_info_t, ndim=2] out
        out = np.zeros([self.num_finite_edges, 2], np_info)
        if out.shape[0] == 0:
            return out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.edge_info(&out[0,0])
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def voronoi_volumes(self):
        r"""np.ndarray of float64: Array of voronoi cell volumes for vertices in 
        the triangulation. The volumes are in the order in which the vertices 
        were added to the triangulation."""
        cdef np.ndarray[np.float64_t, ndim=1] out
        out = np.empty(self.num_finite_verts, 'float64')
        if self.n == 0:
            return out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.dual_areas(&out[0])
        return out
        
    @_update_to_tess
    def remove(self, PeriodicDelaunay2_vertex x):
        r"""Remove a vertex from the triangulation.

        Args:
            x (PeriodicDelaunay2_vertex): Vertex that should be removed.

        """
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.remove(x.x)

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def move(self, PeriodicDelaunay2_vertex x, 
             np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location. If there is a vertex at the given 
        given coordinates, return that vertex and remove the one that was being 
        moved.

        Args:
            x (PeriodicDelaunay2_vertex): Vertex that should be moved.
            pos (:obj:`ndarray` of float64): x,y coordinates that the vertex 
                be moved to.

        Returns:
            PeriodicDelaunay2_vertex: Vertex at the new position.

        """
        # As of CGAL 4.9, this dosn't do as described in the documentation.
        # Rather than moving the point, the CGAL method deletes the old one and 
        # creates a new one if the point does not exist. Otherwise it just 
        # returns the provided point.
        assert(pos.shape[0] == 2)
        cdef PeriodicDelaunay_with_info_2[info_t].Vertex v
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            v = self.T.move(x.x, &pos[0])
        if v == x.x:
            self.remove(x)
            x = self.locate(pos)
            return x
        else:
            x.assign(self.T, v)
            return x

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def move_if_no_collision(self, PeriodicDelaunay2_vertex x, 
                             np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location only if there is not already a 
        vertex at the given coordinates. If there is a vertex there, it is 
        returned and the vertex being moved remains at its original location.

        Args:
            x (PeriodicDelaunay2_vertex): Vertex that should be moved.
            pos (:obj:`ndarray` of float64): x,y coordinates that the vertex 
                be moved to.

        Returns:
            PeriodicDelaunay2_vertex: Vertex at the new position.

        """
        # As of CGAL 4.9, this dosn't do as described in the documentation.
        # Rather than moving the point, the CGAL method creates a new point 
        # if the point does not already exist. Otherwise it just 
        # returns the provided point.
        assert(pos.shape[0] == 2)
        cdef PeriodicDelaunay_with_info_2[info_t].Vertex v
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            v = self.T.move_if_no_collision(x.x, &pos[0])
        if v == x.x:
            return self.locate(pos)
        else:
            self.remove(x)
            x.assign(self.T, v)
            return x

    @_update_to_tess
    def flip(self, PeriodicDelaunay2_cell x, int i):
        r"""Flip the edge incident to cell x and neighbor i of cell x. The 
        method first checks if the edge can be flipped. (In the 2D case, it 
        can always be flipped).

        Args:
            x (PeriodicDelaunay2_cell): Cell with edge that should be flipped.
            i (int): Integer specifying neighbor of x that is also incident 
                to the edge that should be flipped.

        Returns:
            bool: True if facet was flipped, False otherwise. (2D edges can 
                always be flipped).

        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.flip(x.x, i)
        return <pybool>out

    @_update_to_tess
    def flip_flippable(self, PeriodicDelaunay2_cell x, int i):
        r"""Same as :meth:`PeriodicDelaunay2.flip`, but assumes that facet is 
        flippable and does not check.

        Args:
            x (PeriodicDelaunay2_cell): Cell with edge that should be flipped.
            i (int): Integer specifying neighbor of x that is also incident 
                to the edge that should be flipped.

        """
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.flip_flippable(x.x, i)

    def get_vertex(self, info_t index):
        r"""Get the vertex object corresponding to the given index.

        Args:
            index (info_t): Index of vertex that should be found.

        Returns:
            PeriodicDelaunay2_vertex: Vertex corresponding to the given index. 
                If the index is not found, the infinite vertex is returned.

        """
        cdef PeriodicDelaunay_with_info_2[info_t].Vertex v
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            v = self.T.get_vertex(index)
        cdef PeriodicDelaunay2_vertex out = PeriodicDelaunay2_vertex()
        out.assign(self.T, v)
        return out

    def locate(self, np.ndarray[np.float64_t, ndim=1] pos,
               PeriodicDelaunay2_cell start = None):
        r"""Get the vertex/cell/edge that a given point is a part of.
        
        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.
            start (PeriodicDelaunay2_cell, optional): Cell to start the search 
                at.

        Returns:
            :obj:`PeriodicDelaunay2_vertex`, :obj:`PeriodicDelaunay2_cell` or 
                :obj:`PeriodicDelaunay2_edge`: Vertex, cell or edge that the 
                given point is a part of.

        """
        assert(len(pos) == 2)
        cdef int lt, li
        lt = li = 999
        cdef PeriodicDelaunay2_cell c = PeriodicDelaunay2_cell()
        if start is not None:
            c.assign(self.T, self.T.locate(&pos[0], lt, li, start.x))
        else:
            c.assign(self.T, self.T.locate(&pos[0], lt, li))
        assert(lt != 999)
        if lt < 2:
            assert(li != 999)
        if lt == 0:
            return c.vertex(li)
        elif lt == 1:
            return PeriodicDelaunay2_edge.from_cell(c, li)
        elif lt == 2:
            return c
        elif lt == 3:
            print("Point {} is outside the convex hull.".format(pos))
            return c
        elif lt == 4:
            print("Point {} is outside the affine hull.".format(pos))
            return 0
        else:
            raise RuntimeError("Value of {} ".format(lt)+
                               "not expected from CGAL locate.")

    @property
    def all_verts_begin(self):
        r"""PeriodicDelaunay2_vertex_iter: Starting vertex for all vertices in 
        the triangulation."""
        return PeriodicDelaunay2_vertex_iter(self, 'all_begin')
    @property
    def all_verts_end(self):
        r"""PeriodicDelaunay2_vertex_iter: Final vertex for all vertices in the 
        triangulation."""
        return PeriodicDelaunay2_vertex_iter(self, 'all_end')
    @property
    def all_verts(self):
        r"""PeriodicDelaunay2_vertex_range: Iterable for all vertices in the 
        triangulation."""
        return PeriodicDelaunay2_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end)
    @property
    def finite_verts(self):
        r"""PeriodicDelaunay2_vertex_range: Iterable for finite vertices in the 
        triangulation."""
        return PeriodicDelaunay2_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end, finite = True)

    @property
    def all_edges_begin(self):
        r"""PeriodicDelaunay2_edge_iter: Starting edge for all edges in the 
        triangulation."""
        return PeriodicDelaunay2_edge_iter(self, 'all_begin')
    @property
    def all_edges_end(self):
        r"""PeriodicDelaunay2_edge_iter: Final edge for all edges in the 
        triangulation."""
        return PeriodicDelaunay2_edge_iter(self, 'all_end')
    @property
    def all_edges(self):
        r"""PeriodicDelaunay2_edge_range: Iterable for all edges in the 
        triangulation."""
        return PeriodicDelaunay2_edge_range(self.all_edges_begin, 
                                    self.all_edges_end)
    @property
    def finite_edges(self):
        r"""PeriodicDelaunay2_edge_range: Iterable for finite edges in the 
        triangulation."""
        return PeriodicDelaunay2_edge_range(self.all_edges_begin, 
                                    self.all_edges_end, finite = True)

    @property
    def all_cells_begin(self):
        r"""PeriodicDelaunay2_cell_iter: Starting cell for all cells in the 
        triangulation."""
        return PeriodicDelaunay2_cell_iter(self, 'all_begin')
    @property
    def all_cells_end(self):
        r"""PeriodicDelaunay2_cell_iter: Final cell for all cells in the 
        triangulation."""
        return PeriodicDelaunay2_cell_iter(self, 'all_end')
    @property
    def all_cells(self):
        r"""PeriodicDelaunay2_cell_range: Iterable for all cells in the
        triangulation."""
        return PeriodicDelaunay2_cell_range(self.all_cells_begin,
                                    self.all_cells_end)
    @property
    def finite_cells(self):
        r"""PeriodicDelaunay2_cell_range: Iterable for finite cells in the
        triangulation."""
        return PeriodicDelaunay2_cell_range(self.all_cells_begin,
                                    self.all_cells_end, finite = True)

    def is_edge(self, PeriodicDelaunay2_vertex v1, PeriodicDelaunay2_vertex v2, 
                PeriodicDelaunay2_cell c = PeriodicDelaunay2_cell(), int i = 0):
        r"""Determine if two vertices form an edge in the triangulation.

        Args:
            v1 (PeriodicDelaunay2_vertex): First vertex.
            v2 (PeriodicDelaunay2_vertex): Second vertex.
            c (PeriodicDelaunay2_cell, optional): If provided and the two 
                vertices form an edge, the cell incident to the edge is stored 
                here.
            i (int, optional): If provided and the two vertices form an edge, 
                the index of the vertex opposite the edge in cell c is stored 
                here.
        Returns:
            bool: True if v1 and v2 form an edge, False otherwise.

        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.is_edge(v1.x, v2.x, c.x, i)
        return <pybool>out

    def is_cell(self, PeriodicDelaunay2_vertex v1, PeriodicDelaunay2_vertex v2, 
                PeriodicDelaunay2_vertex v3, 
                PeriodicDelaunay2_cell c = PeriodicDelaunay2_cell()):
        r"""Determine if three vertices form a cell in the triangulation.

        Args:
            v1 (PeriodicDelaunay2_vertex): First vertex.
            v2 (PeriodicDelaunay2_vertex): Second vertex.
            v3 (PeriodicDelaunay2_vertex): Third vertex.
            c (PeriodicDelaunay2_cell, optional): If provided and the three 
                vertices form a cell, the cell they form is stored here.

        Returns:
            bool: True if v1, v2, and v3 form a cell, False otherwise.

        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.is_cell(v1.x, v2.x, v3.x, c.x)
        return <pybool>out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def nearest_vertex(self, np.ndarray[np.float64_t, ndim=1] x):
        r"""Determine which vertex is closes to a given set of x,y coordinates

        Args:
            x (:obj:`ndarray` of float64): x,y coordinates.

        Returns:
            PeriodicDelaunay2_vertex: Vertex closest to x.

        """
        assert(x.shape[0] == 2)
        cdef PeriodicDelaunay_with_info_2[info_t].Vertex vc
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            vc = self.T.nearest_vertex(&x[0])
        cdef PeriodicDelaunay2_vertex v = PeriodicDelaunay2_vertex()
        v.assign(self.T, vc)
        return v

    def mirror_index(self, PeriodicDelaunay2_cell x, int i):
        r"""Get the index of a cell with respect to its ith neighbor.

        Args:
            x (PeriodicDelaunay2_cell): Cell to get index for.
            i (int): Index of neighbor that should be used to determine the 
                mirrored index.

        Returns:
            int: Index of cell x with respect to its ith neighbor.

        """
        return x.mirror_index(i)

    def mirror_vertex(self, PeriodicDelaunay2_cell x, int i):
        r"""Get the vertex of a cell's ith neighbor opposite to the cell.

        Args:
            x (PeriodicDelaunay2_cell): Cell.
            i (int): Index of neighbor that should be used to determine the
                mirrored vertex.

        Returns:
            PeriodicDelaunay2_vertex: Vertex in the ith neighboring cell of cell 
                x, that is opposite to cell x.

        """
        return x.mirror_vertex(i)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_boundary_of_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                                  PeriodicDelaunay2_cell start):
        r"""Get the edges of the cell in conflict with a given point.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.
            start (PeriodicDelaunay2_cell): Cell to start list of edges at.

        Returns:
            :obj:`list` of PeriodicDelaunay2_edge: Edges of the cell in conflict 
                 with pos.

        """
        assert(pos.shape[0] == 2)
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Edge] ev
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            ev = self.T.get_boundary_of_conflicts(&pos[0], start.x)
        cdef object out = []
        cdef np.uint32_t i
        cdef PeriodicDelaunay2_edge x
        for i in range(ev.size()):
            x = PeriodicDelaunay2_edge()
            x.assign(self.T, ev[i])
            out.append(x)
        return out
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                      PeriodicDelaunay2_cell start):
        r"""Get the cells that are in conflict with a given point.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates. 
            start (PeriodicDelaunay2_cell): Cell to start list of conflicts at.

        Returns:
            :obj:`list` of PeriodicDelaunay2_cell: Cells in conflict with pos.

        """
        assert(pos.shape[0] == 2)
        cdef vector[PeriodicDelaunay_with_info_2[info_t].Cell] cv
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            cv = self.T.get_conflicts(&pos[0], start.x)
        cdef object out = []
        cdef np.uint32_t i
        cdef PeriodicDelaunay2_cell x
        for i in range(cv.size()):
            x = PeriodicDelaunay2_cell()
            x.assign(self.T, cv[i])
            out.append(x)
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_conflicts_and_boundary(self, np.ndarray[np.float64_t, ndim=1] pos,
                                   PeriodicDelaunay2_cell start):
        r"""Get the cells and edges of cells that are in conflict with a given 
            point.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.
            start (PeriodicDelaunay2_cell): Cell to start list of conflicts at.  
        
        Returns:
            tuple: :obj:`list` of PeriodicDelaunay2_cells in conflict 
                with pos and :obj:`list` of PeriodicDelaunay2_edges 
                bounding the conflicting cells.

        """
        assert(pos.shape[0] == 2)
        cdef pair[vector[PeriodicDelaunay_with_info_2[info_t].Cell],
                  vector[PeriodicDelaunay_with_info_2[info_t].Edge]] cv
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            cv = self.T.get_conflicts_and_boundary(&pos[0], start.x)
        cdef object out_cells = []
        cdef object out_edges = []
        cdef np.uint32_t i
        cdef PeriodicDelaunay2_cell c
        cdef PeriodicDelaunay2_edge e
        for i in range(cv.first.size()):
            c = PeriodicDelaunay2_cell()
            c.assign(self.T, cv.first[i])
            out_cells.append(c)
        for i in range(cv.second.size()):
            e = PeriodicDelaunay2_edge()
            e.assign(self.T, cv.second[i])
            out_edges.append(e)
        return out_cells, out_edges

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def outgoing_points(self, 
                        np.ndarray[np.float64_t, ndim=2] left_edges,
                        np.ndarray[np.float64_t, ndim=2] right_edges): 
        r"""Get the indices of points in tets that intersect a set of boxes.

        Args:
            left_edges (np.ndarray of float64): (m, n) array of m box mins in n 
                dimensions.
            right_edges (np.ndarray of float64): (m, n) array of m box maxs in n 
                dimensions.

        Returns:
        
        """
        assert(left_edges.shape[1] == 2)
        assert(left_edges.shape[0] == right_edges.shape[0])
        assert(left_edges.shape[1] == right_edges.shape[1])
        cdef uint64_t nbox = <uint64_t>left_edges.shape[0]
        cdef vector[vector[info_t]] vout
        if (nbox > 0):
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                vout = self.T.outgoing_points(nbox,
                                              &left_edges[0,0], 
                                              &right_edges[0,0])
        assert(vout.size() == nbox)
        # Transfer values to array
        cdef uint64_t i, j
        cdef object out = [None for i in range(vout.size())]
        for i in range(vout.size()):
            out[i] = np.empty(vout[i].size(), np_info)
            for j in range(vout[i].size()):
                out[i][j] = vout[i][j]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def boundary_points(self, 
                        np.ndarray[np.float64_t, ndim=1] left_edge,
                        np.ndarray[np.float64_t, ndim=1] right_edge, 
                        pybool periodic):
        r"""Get the indices of points in tets that border a box.

        Args:
            left_edge (`np.ndarray` of `np.float64_t`): Minimum boundary of 
                box in each dimension.
            right_edge (`np.ndarray` of `np.float64_t`): Maximum boundary of 
                box in each dimension.
            periodic (bool): True if the domain is periodic, False otherwise.

        Returns:
            tuple: 3 groupings of indices:
                lind: list of np.ndarray indices of points in tets bordering the 
                    left edge of the box in each direction.
                rind: list of np.ndarray indices of points in tets bordering the 
                    right edge of the box in each direction.
                iind: indices of points in tets that are infinite. This will be 
                    empty if `periodic == True`.

        """
        assert(left_edge.shape[0]==2)
        assert(right_edge.shape[0]==2)
        global np_info
        cdef int i, j, k
        cdef vector[info_t] lr, lx, ly, lz, rx, ry, rz, alln
        cdef cbool cperiodic = <cbool>periodic
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.boundary_points(&left_edge[0], &right_edge[0], cperiodic,
                                   lx, ly, rx, ry, alln)
        # Get counts to preallocate
        cdef object lind = [None, None]
        cdef object rind = [None, None]
        cdef info_t iN = 0
        for i in range(2):
            if   i == 0: lr = lx
            elif i == 1: lr = ly
            iN = <info_t>lr.size()
            lind[i] = np.zeros(iN, np_info)
        for i in range(2):
            if   i == 0: lr = rx
            elif i == 1: lr = ry
            iN = <info_t>lr.size()
            rind[i] = np.zeros(iN, np_info)
        # Fill in
        cdef np.ndarray[info_t] iind
        cdef np.ndarray[info_t] lr_arr
        iN = alln.size()
        iind = np.array([alln[j] for j in range(<int>iN)], np_info)
        for i in range(2):
            if   i == 0: lr = lx
            elif i == 1: lr = ly
            iN = <info_t>lr.size()
            lr_arr = np.array([lr[j] for j in range(<int>iN)], np_info)
            lind[i] = lr_arr
        for i in range(2):
            if   i == 0: lr = rx
            elif i == 1: lr = ry
            iN = <info_t>lr.size()
            lr_arr = np.array([lr[j] for j in range(<int>iN)], np_info)
            rind[i] = lr_arr
        # Return
        return lind, rind, iind
