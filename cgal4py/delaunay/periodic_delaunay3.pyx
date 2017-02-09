"""
delaunay3.pyx

Wrapper for CGAL 3D PeriodicDelaunay Triangulation
"""

import cython

import numpy as np
cimport numpy as np
import struct

from cgal4py import PY_MAJOR_VERSION
from cgal4py import plot
from cgal4py.delaunay.tools cimport sortSerializedTess

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

cdef class PeriodicDelaunay3_vertex:
    r"""Wrapper class for a triangulation vertex.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
            object that this vertex belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].Vertex`): C++ vertex 
            object. Direct interaction with this object is not recommended. 

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].Vertex x

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     PeriodicDelaunay_with_info_3[info_t].Vertex x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
                object that this vertex belongs to. 
            x (:obj:`PeriodicDelaunay_with_info_3[info_t].Vertex`): C++ vertex 
                object. Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    def __repr__(self):
        return str("PeriodicDelaunay3_vertex[{} at ".format(self.index)+
                   "{:+7.2e},{:+7.2e},{:+7.2e}]".format(*list(self.point)))

    def __richcmp__(PeriodicDelaunay3_vertex self, 
                    PeriodicDelaunay3_vertex solf, int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    property is_unique:
        r""":obj:`bool`: True if the vertex is the unique unwrapped version."""
        def __get__(self):
            return <pybool>self.T.is_unique(self.x)

    property has_offset:
        r""":obj:`bool`: True if the vertex has a periodic offset (not including 
        any cell offset. False otherwise."""
        def __get__(self):
            return <pybool>self.T.has_offset(self.x)

    def is_infinite(self):
        r"""Determine if the vertex is the infinite vertex.
        
        Returns:
            bool: True if the vertex is the infinite vertex, False otherwise.

        """
        return self.T.is_infinite(self.x)

    def set_point(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Set this vertex's corrdinates.

        Args:
            pos (:obj:`ndarray` of float64): new x,y,z coordinates for vertex.

        """
        self.T.updated = <cbool>True
        assert(len(pos) == 3)
        self.x.set_point(&pos[0])

    def set_cell(self, PeriodicDelaunay3_cell c):
        r"""Set the designated cell for this vertex.

        Args:
            c (PeriodicDelaunay3_cell): Cell that should be set as the 
                designated cell.

        """
        self.T.updated = <cbool>True
        self.x.set_cell(c.x)

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
        of the vertex."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
            self.T.point(self.x, &out[0])
            return out

    property periodic_point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
        of the vertex, not including the periodic offset."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
            self.T.periodic_point(self.x, &out[0])
            return out

    property periodic_offset:
        r""":obj:`ndarray` of :obj:`int`: The number of wrappings applied in 
        each dimension to create this periodic point."""
        def __get__(self):
            cdef np.ndarray[np.int32_t] out = np.zeros(3, 'int32')
            self.T.periodic_offset(self.x, &out[0])
            # self.x.offset(&out[0]) # offset only stored on verts during copy 
            return out

    property index:
        r"""info_t: The index of the vertex point in the input array."""
        def __get__(self):
            cdef info_t out
            out = self.x.info()
            return out

    property dual_volume:
        r"""float64: The volume of the dual Voronoi cell. If the volume is 
        infinite, -1.0 is returned."""
        def __get__(self):
            cdef np.float64_t out = self.T.dual_volume(self.x)
            return out

    property cell:
        r"""PeriodicDelaunay3_cell: Designated cell for this vertex."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_3[info_t].Cell c
            c = self.x.cell()
            cdef PeriodicDelaunay3_cell out = PeriodicDelaunay3_cell()
            out.assign(self.T, c)
            return out

    def incident_vertices(self):
        r"""Find vertices that are adjacent to this vertex.

        Returns:
            PeriodicDelaunay3_vertex_vector: Iterator over vertices incident to 
                this vertex.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef PeriodicDelaunay3_vertex_vector out
        out = PeriodicDelaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this vertex.

        Returns:
            PeriodicDelaunay3_edge_vector: Iterator over edges incident to this 
                vertex.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef PeriodicDelaunay3_edge_vector out = PeriodicDelaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this vertex.

        Returns:
            PeriodicDelaunay3_facet_vector: Iterator over facets incident to 
                this vertex.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef PeriodicDelaunay3_facet_vector out 
        out = PeriodicDelaunay3_facet_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this vertex.

        Returns:
            PeriodicDelaunay3_cell_vector: Iterator over cells incident to this 
                vertex.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef PeriodicDelaunay3_cell_vector out = PeriodicDelaunay3_cell_vector()
        out.assign(self.T, it)
        return out


cdef class PeriodicDelaunay3_vertex_iter:
    r"""Wrapper class for a triangulation vertex iterator.

    Args:
        T (PeriodicDelaunay3): Triangulation that this vertex belongs to.
        vert (:obj:`str`, optional): String specifying the vertex that 
            should be referenced. Valid options include: 
                'all_begin': The first vertex in an iteration over all vertices.  
                'all_end': The last vertex in an iteration over all vertices. 
 
    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
            object that this vertex belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].All_verts_iter`): C++ 
           vertex object. Direct interaction with this object is not 
           recommended. 

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].All_verts_iter x

    def __cinit__(self, PeriodicDelaunay3 T, str vert = None):
        self.T = T.T
        if vert == 'all_begin':
            self.x = self.T.all_verts_begin()
        elif vert == 'all_end':
            self.x = self.T.all_verts_end()

    def __richcmp__(PeriodicDelaunay3_vertex_iter self, 
                    PeriodicDelaunay3_vertex_iter solf, 
                    int op):
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

    def increment(self):
        r"""Advance to the next vertex in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous vertex in the triangulation."""
        predecrement(self.x)

    property vertex:
        r"""PeriodicDelaunay3_vertex: Corresponding vertex object."""
        def __get__(self):
            cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
            
            out.assign(self.T, 
                       PeriodicDelaunay_with_info_3[info_t].Vertex(self.x)) 
            return out


cdef class PeriodicDelaunay3_vertex_range:
    r"""Wrapper class for iterating over a range of triangulation vertices

    Args:
        xstart (PeriodicDelaunay3_vertex_iter): The starting vertex.  
        xstop (PeriodicDelaunay3_vertex_iter): Final vertex that will end the 
            iteration. 
        finite (:obj:`bool`, optional): If True, only finite verts are 
            iterated over. Otherwise, all verts are iterated over. Defaults 
            False.

    Attributes:
        x (PeriodicDelaunay3_vertex_iter): The current vertex. 
        xstop (PeriodicDelaunay3_vertex_iter): Final vertex that will end the 
            iteration. 
        finite (bool): If True, only finite verts are iterater over. Otherwise
            all verts are iterated over. 

    """
    cdef PeriodicDelaunay3_vertex_iter x
    cdef PeriodicDelaunay3_vertex_iter xstop
    cdef pybool finite
    def __cinit__(self, PeriodicDelaunay3_vertex_iter xstart, 
                  PeriodicDelaunay3_vertex_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay3_vertex out
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        if self.x != self.xstop:
            out = self.x.vertex
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class PeriodicDelaunay3_vertex_vector:
    r"""Wrapper class for a vector of vertices.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended.
        v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Vertex]`): Vector 
            of C++ vertices.
        n (int): The number of vertices in the vector.
        i (int): The index of the currect vertex.

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef vector[PeriodicDelaunay_with_info_3[info_t].Vertex] v
    cdef int n
    cdef int i

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     vector[PeriodicDelaunay_with_info_3[info_t].Vertex] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
                object. Direct interaction with this object is not recommended.
            v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Vertex]`): 
                Vector of C++ vertices.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay3_vertex out
        if self.i < self.n:
            out = PeriodicDelaunay3_vertex()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef PeriodicDelaunay3_vertex out
        if isinstance(i, int):
            out = PeriodicDelaunay3_vertex()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("PeriodicDelaunay3_vertex_vector indices must be "+
                            "integers, not {}".format(type(i)))


cdef class PeriodicDelaunay3_edge:
    r"""Wrapper class for a triangulation edge.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation object 
            that this edge belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].Edge`): C++ edge object 
            Direct interaction with this object is not recommended. 

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].Edge x

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     PeriodicDelaunay_with_info_3[info_t].Edge x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
                object that this edge belongs to. 
            x (:obj:`PeriodicDelaunay_with_info_3[info_t].Edge`): C++ edge 
                object. Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    @staticmethod
    def from_cell(PeriodicDelaunay3_cell c, int i, int j):
        r"""Construct an edges from a cell and indices of the two vertices 
        in the cell that are incident to the edge.

        Args:
            c (PeriodicDelaunay3_cell): Cell 
            i (int): Index of one vertex in c, incident to the edge.
            j (int): Index of second vertex in c, incident to the edge.

        Returns:
            PeriodicDelaunay3_edge: Edge incident to c and vertices i & j of 
                cell c.

        """
        cdef PeriodicDelaunay3_edge out = PeriodicDelaunay3_edge()
        cdef PeriodicDelaunay_with_info_3[info_t].Edge e
        e = PeriodicDelaunay_with_info_3[info_t].Edge(c.x, i, j)
        out.assign(c.T, e)
        return out

    def __repr__(self):
        return "PeriodicDelaunay3_edge[{},{}]".format(repr(self.vertex1),
                                              repr(self.vertex2))

    def __richcmp__(PeriodicDelaunay3_edge self, PeriodicDelaunay3_edge solf, 
                    int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    property is_unique:
        r""":obj:`bool`: True if the edge is the unique unwrapped version."""
        def __get__(self):
            return <pybool>self.T.is_unique(self.x)

    property has_offset:
        r""":obj:`bool`: True if any of the incident vertices has a periodic
        offset (not including any cell offset. False otherwise."""
        def __get__(self):
            return <pybool>self.T.has_offset(self.x)

    property spans_wrap:
        r""":obj:`bool`: True if the incident vertices span the first and 
        last sheet in one dimension. False otherwise."""
        def __get__(self):
            return <pybool>self.T.spans_wrap(self.x)

    def is_infinite(self):
        r"""Determine if the edge is incident to the infinite vertex.
        
        Returns:
            bool: True if the edge is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def is_Gabriel(self):
        r"""Determines if the edge is Gabriel (does not contain any other 
            vertices in it's smallest circumsphere). 

        Returns: 
            bool: True if the edge is Gabriel, False otherwise. 

        """
        return <pybool>self.T.is_Gabriel(self.x)

    def is_equivalent(self, PeriodicDelaunay3_edge solf):
        r"""Determine if another edge has the same vertices as this edge.

        Args:
            solf (PeriodicDelaunay3_edge): Edge for comparison.

        Returns:
            bool: True if the two edges share the same vertices, False 
                otherwise.

        """
        return <pybool>self.T.are_equal(self.x, solf.x)

    def vertex(self, int i):
        r"""Get the ith vertex on this edge.

        Args:
            i (int): Index of vertex to return.

        Returns:
            PeriodicDelaunay3_vertex: ith vertex on this edge.

        """
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex x
        x = self.x.vertex(i)
        cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
        out.assign(self.T, x)
        return out

    def point(self, int i):
        r"""Return the (x, y, z) coordinates of the ith vertex incident to this 
        edge including the periodic offsets.

        Args: 
            i (int): Index of vertex incident to this edge.

        Returns: 
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
                of the vertex including the periodic offset. 

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
        self.T.point(self.x, i, &out[0])
        return out

    def periodic_point(self, int i):
        r"""Return the (x, y, z) coordinates of the ith vertex incident to this 
        edge, not including the periodic offset. 

        Args: 
            i (int): Index of vertex incident to this edge. 

        Returns: 
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
                of the vertex including the periodic offset. 

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
        self.T.periodic_point(self.x, i, &out[0])
        return out

    def periodic_offset(self, int i):
        r"""Return the number of wrappings in (x,y,z) applied to the ith vertex 
        of this edge. 

        Args: 
            i (int): Index of vertex incident to this edge. 

        Returns: 
            :obj:`ndarray` of :obj:`int32`: The number of wrappings in (x,y,z) 
                applied to the vertex. 

        """
        cdef np.ndarray[np.int32_t] out = np.zeros(3, 'int32')
        self.T.periodic_offset(self.x, i, &out[0])
        return out

    property vertex1:
        r"""PeriodicDelaunay3_vertex: The first vertex in the edge."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_3[info_t].Vertex x = self.x.v1()
            cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
            out.assign(self.T, x)
            return out

    property vertex2:
        r"""PeriodicDelaunay3_vertex: The second vertex in the edge."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_3[info_t].Vertex x = self.x.v2()
            cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
            out.assign(self.T, x)
            return out

    property cell:
        r"""PeriodicDelaunay3_cell: The cell this edge is assigned to."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_3[info_t].Cell c
            c = self.x.cell()
            cdef PeriodicDelaunay3_cell out = PeriodicDelaunay3_cell()
            out.assign(self.T, c)
            return out

    property ind1:
        r"""int: The index of the 1st vertex of this edge in its cell."""
        def __get__(self):
            return self.x.ind1()

    property ind2:
        r"""int: The index of the 2nd vertex of this edge in its cell."""
        def __get__(self):
            return self.x.ind2()

    property center:
        r""":obj:`ndarray` of float64: x,y,z coordinates of edge center."""
        def __get__(self):
            if self.is_infinite():
                return np.float('inf')*np.ones(3, 'float64')
            else:
                return (self.point(0) + self.point(1))/2.0

    property midpoint:
        r""":obj:`ndarray` of float64: x,y,z coordinates of edge midpoint."""
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
            PeriodicDelaunay3_vertex_vector: Iterator over vertices incident to 
                this edge.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef PeriodicDelaunay3_vertex_vector out
        out = PeriodicDelaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this edge.

        Returns:
            PeriodicDelaunay3_edge_vector: Iterator over edges incident to this 
                edge. 

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef PeriodicDelaunay3_edge_vector out = PeriodicDelaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this edge.

        Returns:
            PeriodicDelaunay3_facet_vector: Iterator over facets incident to 
                this edge. 

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef PeriodicDelaunay3_facet_vector out
        out = PeriodicDelaunay3_facet_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this edge.

        Returns:
            PeriodicDelaunay3_cell_vector: Iterator over cells incident to this 
                edge.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef PeriodicDelaunay3_cell_vector out = PeriodicDelaunay3_cell_vector()
        out.assign(self.T, it)
        return out


cdef class PeriodicDelaunay3_edge_iter:
    r"""Wrapper class for a triangulation edge iterator.

    Args:
        T (PeriodicDelaunay3): Triangulation that this edge belongs to.
        edge (:obj:`str`, optional): String specifying the edge that 
            should be referenced. Valid options include: 
                'all_begin': The first edge in an iteration over all edges.
                'all_end': The last edge in an iteration over all edges.
 
    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
            object that this edge belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].All_edges_iter`): C++ edge  
            object. Direct interaction with this object is not recommended. 

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].All_edges_iter x

    def __cinit__(self, PeriodicDelaunay3 T, str edge = None):
        self.T = T.T
        if edge == 'all_begin':
            self.x = self.T.all_edges_begin()
        elif edge == 'all_end':
            self.x = self.T.all_edges_end()

    def __richcmp__(PeriodicDelaunay3_edge_iter self, 
                    PeriodicDelaunay3_edge_iter solf, 
                    int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the edge is incident to the the infinite vertex.
        
        Returns:
            bool: True if the edge is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def increment(self):
        r"""Advance to the next edge in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous edge in the triangulation."""
        predecrement(self.x)

    property edge:
        r"""PeriodicDelaunay3_edge: Corresponding edge object."""
        def __get__(self):
            cdef PeriodicDelaunay3_edge out = PeriodicDelaunay3_edge()
            out.assign(self.T, 
                       PeriodicDelaunay_with_info_3[info_t].Edge(self.x))
            return out


cdef class PeriodicDelaunay3_edge_range:
    r"""Wrapper class for iterating over a range of triangulation edges.

    Args:
        xstart (PeriodicDelaunay3_edge_iter): The starting edge.  
        xstop (PeriodicDelaunay3_edge_iter): Final edge that will end the 
            iteration. 
        finite (:obj:`bool`, optional): If True, only finite edges are 
            iterated over. Otherwise, all edges are iterated over. Defaults 
            False.

    Attributes:
        x (PeriodicDelaunay3_edge_iter): The currentedge. 
        xstop (PeriodicDelaunay3_edge_iter): Final edge that will end the 
            iteration. 
        finite (bool): If True, only finite edges are iterater over. Otherwise
            all edges are iterated over. 

    """
    cdef PeriodicDelaunay3_edge_iter x
    cdef PeriodicDelaunay3_edge_iter xstop
    cdef pybool finite
    def __cinit__(self, PeriodicDelaunay3_edge_iter xstart, 
                  PeriodicDelaunay3_edge_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef PeriodicDelaunay3_edge out
        if self.x != self.xstop:
            out = self.x.edge
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class PeriodicDelaunay3_edge_vector:
    r"""Wrapper class for a vector of edges.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended.
        v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Edge]`): Vector of 
            C++ edges.
        n (int): The number of edges in the vector.
        i (int): The index of the currect edge.

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef vector[PeriodicDelaunay_with_info_3[info_t].Edge] v
    cdef int n
    cdef int i

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     vector[PeriodicDelaunay_with_info_3[info_t].Edge] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
                object. Direct interaction with this object is not recommended.
            v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Edge]`): Vector 
                of C++ edges.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay3_edge out
        if self.i < self.n:
            out = PeriodicDelaunay3_edge()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef PeriodicDelaunay3_edge out
        if isinstance(i, int):
            out = PeriodicDelaunay3_edge()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("PeriodicDelaunay3_edge_vector indices must be "+
                            "integers, not {}".format(type(i)))


cdef class PeriodicDelaunay3_facet:
    r"""Wrapper class for a triangulation facet.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
            object that this facet belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].Facet`): C++ facet object 
            Direct interaction with this object is not recommended. 

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].Facet x

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     PeriodicDelaunay_with_info_3[info_t].Facet x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
                object that this facet belongs to. 
            x (:obj:`PeriodicDelaunay_with_info_3[info_t].Facet`): C++ facet 
                object Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    @staticmethod
    def from_cell(PeriodicDelaunay3_cell c, int i):
        r"""Construct a facet from a cell and index of the vertex in the cell 
        opposite the desired facet.

        Args:
            c (PeriodicDelaunay3_cell): Cell 
            i (int): Index of vertex in c that is opposite the facet.

        Returns:
            PeriodicDelaunay3_facet: Facet incident to c and opposite vertex i 
                in c.

        """
        cdef PeriodicDelaunay3_facet out = PeriodicDelaunay3_facet()
        cdef PeriodicDelaunay_with_info_3[info_t].Facet e
        e = PeriodicDelaunay_with_info_3[info_t].Facet(c.x, i)
        out.assign(c.T, e)
        return out

    def __repr__(self):
        return "PeriodicDelaunay3_facet[{},{},{}]".format(repr(self.vertex(0)),
                                                  repr(self.vertex(1)),
                                                  repr(self.vertex(2)))

    def __richcmp__(PeriodicDelaunay3_facet self, PeriodicDelaunay3_facet solf, 
                    int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    property is_unique:
        r""":obj:`bool`: True if the facet is the unique unwrapped version."""
        def __get__(self):
            return <pybool>self.T.is_unique(self.x)

    property has_offset:
        r""":obj:`bool`: True if any of the incident vertices has a periodic 
        offset (not including any cell offset). False otherwise."""
        def __get__(self):
            return <pybool>self.T.has_offset(self.x)

    def is_infinite(self):
        r"""Determine if the facet is incident to the infinite vertex.
        
        Returns:
            bool: True if the facet is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def is_Gabriel(self):
        r"""Determines if the facet is Gabriel (does not contain any other 
            vertices in it's smallest circumsphere). 

        Returns: 
            bool: True if the facet is Gabriel, False otherwise. 

        """
        return <pybool>self.T.is_Gabriel(self.x)

    def is_equivalent(self, PeriodicDelaunay3_facet solf):
        r"""Determine if another facet has the same vertices as this facet.

        Args:
            solf (PeriodicDelaunay3_facet): Facet for comparison.

        Returns:
            bool: True if the two facets share the same vertices, False 
                otherwise.

        """
        return <pybool>self.T.are_equal(self.x, solf.x)

    def edge(self, int i):
        r"""Get the edge opposite the ith vertex incident to this facet.

        Args:
            i (int): Index of the edge that should be returned.
        
        Returns:
            PeriodicDelaunay3_edge: Edge opposite the ith vertex of this facet.

        """
        cdef PeriodicDelaunay_with_info_3[info_t].Edge e
        e = self.x.edge(i)
        cdef PeriodicDelaunay3_edge out = PeriodicDelaunay3_edge()
        out.assign(self.T, e)
        return out

    def vertex(self, int i):
        r"""Get the ith vertex incident to this facet.

        Args:
            i (int): Index of vertex that should be returned.

        Returns:
            PeriodicDelaunay_vertex: ith vertex of this facet.

        """
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex v
        v = self.x.vertex(i)
        cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
        out.assign(self.T, v)
        return out

    def point(self, int i):
        r"""Return the (x, y, z) coordinates of the ith vertex incident to this 
        facet including the periodic offsets.

        Args: 
            i (int): Index of vertex incident to this facet.

        Returns: 
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
                of the vertex including the periodic offset. 

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
        self.T.point(self.x, i, &out[0])
        return out

    def periodic_point(self, int i):
        r"""Return the (x, y, z) coordinates of the ith vertex incident to this 
        facet, not including the periodic offset. 

        Args: 
            i (int): Index of vertex incident to this facet. 

        Returns: 
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
                of the vertex including the periodic offset. 

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
        self.T.periodic_point(self.x, i, &out[0])
        return out

    def periodic_offset(self, int i):
        r"""Return the number of wrappings in (x,y,z) applied to the ith vertex 
        of this facet. 

        Args: 
            i (int): Index of vertex incident to this facet. 

        Returns: 
            :obj:`ndarray` of :obj:`int32`: The number of wrappings in (x,y,z) 
                applied to the vertex. 

        """
        cdef np.ndarray[np.int32_t] out = np.zeros(3, 'int32')
        self.T.periodic_offset(self.x, i, &out[0])
        return out

    property center:
        r""":obj:`ndarray` of float64: x,y,z coordinates of cell center."""
        def __get__(self):
            if self.is_infinite():
                return np.float('inf')*np.ones(3, 'float64')
            else:
                return (self.point(0) + \
                        self.point(1) + \
                        self.point(2))/3.0
                        
    property area:
        r"""float64: The area of the facet. If infinite, -1 is returned"""
        def __get__(self):
            raise NotImplementedError
            # return -1

    property cell:
        r"""PeriodicDelaunay3_cell: The cell this facet is assigned to."""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_3[info_t].Cell c
            c = self.x.cell()
            cdef PeriodicDelaunay3_cell out = PeriodicDelaunay3_cell()
            out.assign(self.T, c)
            return out

    property ind:
        r"""int: The index of the vertex this facet is opposite on its cell."""
        def __get__(self):
            return self.x.ind()

    property mirror:
        r"""PeriodicDelaunay3_facet: The same facet as this one, but referenced 
        from its other incident cell"""
        def __get__(self):
            cdef PeriodicDelaunay_with_info_3[info_t].Facet ec
            ec = self.T.mirror_facet(self.x)
            cdef PeriodicDelaunay3_facet out = PeriodicDelaunay3_facet()
            out.assign(self.T, ec)
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this facet.

        Returns:
            PeriodicDelaunay3_vertex_vector: Iterator over vertices incident to 
                this facet.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef PeriodicDelaunay3_vertex_vector out
        out = PeriodicDelaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this facet.

        Returns:
            PeriodicDelaunay3_edge_vector: Iterator over edges incident to this 
                facet. 

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef PeriodicDelaunay3_edge_vector out = PeriodicDelaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this facet.

        Returns:
            PeriodicDelaunay3_facet_vector: Iterator over facets incident to 
                this facet. 

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef PeriodicDelaunay3_facet_vector out
        out = PeriodicDelaunay3_facet_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this facet.

        Returns:
            PeriodicDelaunay3_cell_vector: Iterator over cells incident to this 
                facet.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef PeriodicDelaunay3_cell_vector out = PeriodicDelaunay3_cell_vector()
        out.assign(self.T, it)
        return out

    # Currently segfaults inside CGAL function
    # def side_of_circle(self, np.ndarray[np.float64_t, ndim=1] pos):
    #     r"""Determine where a point is with repect to this facet's 
    #         circumcircle. 
    #
    #     Args: 
    #         pos (:obj:`ndarray` of np.float64): x,y,z coordinates. 
    # 
    #     Returns: 
    #         int: -1, 0, or 1 if `pos` is within, on, or inside this facet's 
    #             circumcircle respectively. 
    # 
    #     """
    #     return self.T.side_of_circle(self.x, &pos[0])


cdef class PeriodicDelaunay3_facet_iter:
    r"""Wrapper class for a triangulation facet iterator.

    Args:
        T (PeriodicDelaunay3): Triangulation that this facet belongs to.
        facet (:obj:`str`, optional): String specifying the facet that 
            should be referenced. Valid options include: 
                'all_begin': The first facet in an iteration over all facets.
                'all_end': The last facet in an iteration over all facets.
 
    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
            object that this facet belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].All_facets_iter`): C++ 
            facet object. Direct interaction with this object is not 
            recommended. 

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].All_facets_iter x

    def __cinit__(self, PeriodicDelaunay3 T, str facet = None):
        self.T = T.T
        if facet == 'all_begin':
            self.x = self.T.all_facets_begin()
        elif facet == 'all_end':
            self.x = self.T.all_facets_end()

    def __richcmp__(PeriodicDelaunay3_facet_iter self, 
                    PeriodicDelaunay3_facet_iter solf, 
                    int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the facet is incident to the the infinite vertex.
        
        Returns:
            bool: True if the facet is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def increment(self):
        r"""Advance to the next facet in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous facet in the triangulation."""
        predecrement(self.x)

    property facet:
        r"""PeriodicDelaunay3_facet: Corresponding facet object."""
        def __get__(self):
            cdef PeriodicDelaunay3_facet out = PeriodicDelaunay3_facet()
            out.assign(self.T, 
                       PeriodicDelaunay_with_info_3[info_t].Facet(self.x))
            return out


cdef class PeriodicDelaunay3_facet_range:
    r"""Wrapper class for iterating over a range of triangulation facets.

    Args:
        xstart (PeriodicDelaunay3_facet_iter): The starting facet.  
        xstop (PeriodicDelaunay3_facet_iter): Final facet that will end the 
            iteration. 
        finite (:obj:`bool`, optional): If True, only finite facets are 
            iterated over. Otherwise, all facets are iterated over. Defaults 
            False.

    Attributes:
        x (PeriodicDelaunay3_facet_iter): The currentfacet. 
        xstop (PeriodicDelaunay3_facet_iter): Final facet that will end the 
            iteration. 
        finite (bool): If True, only finite facets are iterater over. Otherwise
            all facets are iterated over. 

    """
    cdef PeriodicDelaunay3_facet_iter x
    cdef PeriodicDelaunay3_facet_iter xstop
    cdef pybool finite
    def __cinit__(self, PeriodicDelaunay3_facet_iter xstart, 
                  PeriodicDelaunay3_facet_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef PeriodicDelaunay3_facet out
        if self.x != self.xstop:
            out = self.x.facet
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class PeriodicDelaunay3_facet_vector:
    r"""Wrapper class for a vector of facets.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended.
        v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Facet]`): Vector of 
            C++ facets.
        n (int): The number of facets in the vector.
        i (int): The index of the currect facet.

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef vector[PeriodicDelaunay_with_info_3[info_t].Facet] v
    cdef int n
    cdef int i

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     vector[PeriodicDelaunay_with_info_3[info_t].Facet] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
                object. Direct interaction with this object is not recommended.
            v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Facet]`): 
                Vector of C++ facets.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay3_facet out
        if self.i < self.n:
            out = PeriodicDelaunay3_facet()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef PeriodicDelaunay3_facet out
        if isinstance(i, int):
            out = PeriodicDelaunay3_facet()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("PeriodicDelaunay3_facet_vector indices must be "+
                            "integers, not {}".format(type(i)))


cdef class PeriodicDelaunay3_cell:
    r"""Wrapper class for a triangulation cell.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
            object that this cell belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].Cell`): C++ cell object. 
            Direct interaction with this object is not recommended.

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].Cell x

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     PeriodicDelaunay_with_info_3[info_t].Cell x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
                object that this cell belongs to. 
            x (:obj:`PeriodicDelaunay_with_info_3[info_t].Cell`): C++ cell 
                object. Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    def __repr__(self):
        return "PeriodicDelaunay2_cell[{},{},{},{}]".format(
            repr(self.vertex(0)), repr(self.vertex(1)),
            repr(self.vertex(2)), repr(self.vertex(3)))

    def __richcmp__(PeriodicDelaunay3_cell self, PeriodicDelaunay3_cell solf, 
                    int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    property is_unique:
        r""":obj:`bool`: True if the cell is the unique unwrapped version."""
        def __get__(self):
            return <pybool>self.T.is_unique(self.x)

    property has_offset:
        r""":obj:`bool`: True if any of the incident vertices has a periodic 
        offset (not including any cell offset). False otherwise."""
        def __get__(self):
            return <pybool>self.T.has_offset(self.x)

    property spans_wrap:
        r""":obj:`bool`: True if the incident vertices span the first and 
        last sheet in one dimension. False otherwise."""
        def __get__(self):
            return <pybool>self.T.spans_wrap(self.x)

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
            PeriodicDelaunay3_vertex: Vertex in the ith neighboring cell of this 
                cell, that is opposite to this cell. 

        """
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex vc
        vc = self.T.mirror_vertex(self.x, i)
        cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
        out.assign(self.T, vc)
        return out

    def facet(self, int i):
        r"""Find the facet opposite the ith vertex incident to this cell.

        Args:
            i (int): Index of vertex opposite the desired facet.

        Returns:
            PeriodicDelaunay3_facet: The facet opposite the ith vertex incident 
                to this cell.

        """
        cdef PeriodicDelaunay_with_info_3[info_t].Facet f
        f = self.x.facet(i)
        cdef PeriodicDelaunay3_facet out = PeriodicDelaunay3_facet()
        out.assign(self.T, f)
        return out

    def vertex(self, int i):
        r"""Find the ith vertex that is incident to this cell. 

        Args:
            i (int): The index of the vertex that should be returned.

        Returns:
            PeriodicDelaunay3_vertex: The ith vertex incident to this cell. 

        """
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex v
        v = self.x.vertex(i)
        cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
        out.assign(self.T, v)
        return out

    def point(self, int i):
        r"""Return the (x, y, z) coordinates of the ith vertex incident to this 
        facet including the periodic offsets.

        Args: 
            i (int): Index of vertex incident to this facet.

        Returns: 
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
                of the vertex including the periodic offset. 

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
        self.T.point(self.x, i, &out[0])
        return out

    def periodic_point(self, int i):
        r"""Return the (x, y, z) coordinates of the ith vertex incident to this 
        facet, not including the periodic offset. 

        Args: 
            i (int): Index of vertex incident to this facet. 

        Returns: 
            :obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
                of the vertex including the periodic offset. 

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
        self.T.periodic_point(self.x, i, &out[0])
        return out

    def periodic_offset(self, int i):
        r"""Return the number of wrappings in (x,y,z) applied to the ith vertex 
        of this facet. 

        Args: 
            i (int): Index of vertex incident to this facet. 

        Returns: 
            :obj:`ndarray` of :obj:`int32`: The number of wrappings in (x,y,z) 
                applied to the vertex. 

        """
        cdef np.ndarray[np.int32_t] out = np.zeros(3, 'int32')
        self.T.periodic_offset(self.x, i, &out[0])
        return out

    def has_vertex(self, PeriodicDelaunay3_vertex v, 
                   pybool return_index = False):
        r"""Determine if a vertex belongs to this cell.

        Args:
            v (PeriodicDelaunay3_vertex): Vertex to test ownership for. 
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
            
    def ind_vertex(self, PeriodicDelaunay3_vertex v):
        r"""Determine the index of a vertex within a cell. 

        Args: 
            v (PeriodicDelaunay3_vertex): Vertex to find index for. 

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
        cdef PeriodicDelaunay_with_info_3[info_t].Cell v
        v = self.x.neighbor(i)
        cdef PeriodicDelaunay3_cell out = PeriodicDelaunay3_cell()
        out.assign(self.T, v)
        return out

    def has_neighbor(self, PeriodicDelaunay3_cell v, 
                     pybool return_index = False):
        r"""Determine if a cell is a neighbor to this cell. 

        Args: 
            v (PeriodicDelaunay3_cell): Cell to test as a neighbor. 
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

    def ind_neighbor(self, PeriodicDelaunay3_cell v):
        r"""Determine the index of a neighboring cell. 

        Args: 
            v (PeriodicDelaunay3_cell): Neighboring cell to find index for. 

        Returns: 
            int: Index of vertex opposite to neighboring cell. 

        """
        return self.x.ind(v.x)

    def set_vertex(self, int i, PeriodicDelaunay3_vertex v):
        r"""Set the ith vertex of this cell. 

        Args: 
            i (int): Index of this cell's vertex that should be set. 
            v (PeriodicDelaunay3_vertex): Vertex to set ith vertex of this cell
                to. 

        """
        self.T.updated = <cbool>True
        self.x.set_vertex(i, v.x)

    def set_vertices(self, PeriodicDelaunay3_vertex v1, 
                     PeriodicDelaunay3_vertex v2,
                     PeriodicDelaunay3_vertex v3, PeriodicDelaunay3_vertex v4):
        r"""Set this cell's vertices. 

        Args: 
            v1 (PeriodicDelaunay2_vertex): 1st vertex of cell. 
            v2 (PeriodicDelaunay2_vertex): 2nd vertex of cell. 
            v3 (PeriodicDelaunay2_vertex): 3rd vertex of cell. 
            v4 (PeriodicDelaunay2_vertex): 4th vertex of cell. 

        """
        self.T.updated = <cbool>True
        self.x.set_vertices(v1.x, v2.x, v3.x, v4.x)

    def reset_vertices(self):
        r"""Reset all of this cell's vertices."""
        self.T.updated = <cbool>True
        self.x.set_vertices()

    def set_neighbor(self, int i, PeriodicDelaunay3_cell n):
        r"""Set the ith neighboring cell of this cell. 

        Args: 
            i (int): Index of this cell's neighbor that should be set. 
            n (PeriodicDelaunay3_cell): Cell to set ith neighbor of this cell to. 

        """
        self.T.updated = <cbool>True
        self.x.set_neighbor(i, n.x)

    def set_neighbors(self, PeriodicDelaunay3_cell c1, 
                      PeriodicDelaunay3_cell c2,
                      PeriodicDelaunay3_cell c3, PeriodicDelaunay3_cell c4):
        r"""Set this cell's neighboring cells. 

        Args: 
            c1 (PeriodicDelaunay3_cell): 1st neighboring cell. 
            c2 (PeriodicDelaunay3_cell): 2nd neighboring cell. 
            c3 (PeriodicDelaunay3_cell): 3rd neighboring cell. 
            c4 (PeriodicDelaunay3_cell): 4th neighboring cell. 

        """
        self.T.updated = <cbool>True
        self.x.set_neighbors(c1.x, c2.x, c3.x, c4.x)

    def reset_neighbors(self):
        r"""Reset all of this cell's neighboring cells."""
        self.T.updated = <cbool>True
        self.x.set_neighbors()

    property center:
        """:obj:`ndarray` of float64: x,y,z coordinates of cell center."""
        def __get__(self):
            if self.is_infinite():
                return np.float('inf')*np.ones(3, 'float64')
            else:
                return (self.point(0) + \
                        self.point(1) + \
                        self.point(2) + \
                        self.point(3))/4.0

    property circumcenter:
        """:obj:`ndarray` of float64: x,y,z coordinates of cell circumcenter."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
            self.T.circumcenter(self.x, &out[0])
            return out

    property periodic_circumcenter:
        """:obj:`ndarray` of float64: x,y,z coordinates of cell circumcenter."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
            self.T.periodic_circumcenter(self.x, &out[0])
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this cell.

        Returns:
            PeriodicDelaunay3_vertex_vector: Iterator over vertices incident to 
                this cell.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef PeriodicDelaunay3_vertex_vector out
        out = PeriodicDelaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this cell.

        Returns:
            PeriodicDelaunay3_edge_vector: Iterator over edges incident to this 
                cell. 

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef PeriodicDelaunay3_edge_vector out = PeriodicDelaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this cell.

        Returns:
            PeriodicDelaunay3_facet_vector: Iterator over facets incident to 
                this cell. 

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef PeriodicDelaunay3_facet_vector out
        out = PeriodicDelaunay3_facet_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this cell.

        Returns:
            PeriodicDelaunay3_cell_vector: Iterator over cells incident to this 
                cell.

        """
        cdef vector[PeriodicDelaunay_with_info_3[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef PeriodicDelaunay3_cell_vector out = PeriodicDelaunay3_cell_vector()
        out.assign(self.T, it)
        return out

    def side(self, np.ndarray[np.float64_t, ndim=1] p):
        r"""Determine if a point is inside, outside or on this cell.

        Args:
            p (np.ndarray of float64): x,y,z coordinates.

        Returns:
            int: -1 if p is inside this cell, 0 if p is on one of this cell's 
                vertices, edges, or facets, and 1 if p is outside this cell.

        """
        cdef int lt, li, lj
        lt = li = lj = 999
        return self.T.side_of_cell(&p[0], self.x, lt, li, lj)

    def side_of_sphere(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Determine where a point is with repect to this cell's 
            circumsphere. 

        Args: 
            pos (:obj:`ndarray` of np.float64): x,y,z coordinates. 

        Returns: 
            int: -1, 0, or 1 if `pos` is within, on, or inside this cell's 
                circumsphere respectively. 

        """
        return self.T.side_of_sphere(self.x, &pos[0])


cdef class PeriodicDelaunay3_cell_iter:
    r"""Wrapper class for a triangulation cell iteration.

    Args:
        T (PeriodicDelaunay3): Triangulation that this cell belongs to. 
        cell (:obj:`str`, optional): String specifying the cell that
            should be referenced. Valid options include: 
                'all_begin': The first cell in an iteration over all cells. 
                'all_end': The last cell in an iteration over all cells.
    
    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ Triangulation 
            object that this cell belongs to. 
        x (:obj:`PeriodicDelaunay_with_info_3[info_t].All_cells_iter`): C++ cell
            object. Direct interaction with this object is not recommended.

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef PeriodicDelaunay_with_info_3[info_t].All_cells_iter x

    def __cinit__(self, PeriodicDelaunay3 T, str cell = None):
        self.T = T.T
        if cell == 'all_begin':
            self.x = self.T.all_cells_begin()
        elif cell == 'all_end':
            self.x = self.T.all_cells_end()

    def __richcmp__(PeriodicDelaunay3_cell_iter self, 
                    PeriodicDelaunay3_cell_iter solf, int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the cell is incident to the infinite vertex.

        Returns:
            bool: True if the cell is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def increment(self):
        r"""Advance to the next cell in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous cell in the triangulation."""
        predecrement(self.x)

    property cell:
        r"""PeriodicDelaunay3_cell: Corresponding cell object."""
        def __get__(self):
            cdef PeriodicDelaunay3_cell out = PeriodicDelaunay3_cell()
            out.T = self.T
            out.x = PeriodicDelaunay_with_info_3[info_t].Cell(self.x)
            return out


cdef class PeriodicDelaunay3_cell_range:
    r"""Wrapper class for iterating over a range of triangulation cells.

    Args:
        xstart (PeriodicDelaunay3_cell_iter): The starting cell. 
        xstop (PeriodicDelaunay3_cell_iter): Final cell that will end the 
            iteration. 
        finite (:obj:`bool`, optional): If True, only finite cells are 
            iterated over. Otherwise, all cells are iterated over. Defaults
            to False.  

    Attributes:
        x (PeriodicDelaunay3_cell_iter): The current cell. 
        xstop (PeriodicDelaunay3_cell_iter): Final cell that will end the 
            iteration. 
        finite (bool): If True, only finite cells are iterated over. Otherwise, 
            all cells are iterated over.   

    """
    cdef PeriodicDelaunay3_cell_iter x
    cdef PeriodicDelaunay3_cell_iter xstop
    cdef pybool finite
    def __cinit__(self, PeriodicDelaunay3_cell_iter xstart, 
                  PeriodicDelaunay3_cell_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef PeriodicDelaunay3_cell out
        if self.x != self.xstop:
            out = self.x.cell
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class PeriodicDelaunay3_cell_vector:
    r"""Wrapper class for a vector of cells.

    Attributes:
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended.
        v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Cell]`): Vector of 
            C++ cells.
        n (int): The number of cells in the vector.
        i (int): The index of the currect cell.

    """
    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef vector[PeriodicDelaunay_with_info_3[info_t].Cell] v
    cdef int n
    cdef int i

    cdef void assign(self, PeriodicDelaunay_with_info_3[info_t] *T,
                     vector[PeriodicDelaunay_with_info_3[info_t].Cell] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
                object. Direct interaction with this object is not recommended.
            v (:obj:`vector[PeriodicDelaunay_with_info_3[info_t].Cell]`): Vector 
                of C++ cells.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef PeriodicDelaunay3_cell out
        if self.i < self.n:
            out = PeriodicDelaunay3_cell()
            out.T = self.T
            out.x = self.v[self.i]
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef PeriodicDelaunay3_cell out
        if isinstance(i, int):
            out = PeriodicDelaunay3_cell()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("PeriodicDelaunay3_cell_vector indices must be "+
                            "integers, not {}".format(type(i)))


cdef class PeriodicDelaunay3:
    r"""Wrapper class for a 3D PeriodicDelaunay triangulation.

    Attributes:
        n (int): The number of points inserted into the triangulation.
        T (:obj:`PeriodicDelaunay_with_info_3[info_t]`): C++ triangulation 
            object. Direct interaction with this object is not recommended. 
        n_per_insert (list of int): The number of points inserted at each
            insert.

    """

    cdef PeriodicDelaunay_with_info_3[info_t] *T
    cdef readonly int n
    cdef public object n_per_insert
    cdef readonly pybool _locked
    cdef public object _cache_to_clear_on_update

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray[np.float64_t] left_edge = None,
                  np.ndarray[np.float64_t] right_edge = None):
        cdef np.ndarray[np.float64_t] domain = np.empty(2*3, 'float64')
        if left_edge is None or right_edge is None:
            domain[:3] = [0,0,0]
            domain[3:] = [1,1,1]
        else:
            domain[:3] = left_edge
            domain[3:] = right_edge
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T = new PeriodicDelaunay_with_info_3[info_t](&domain[0])
        self.n = 0
        self.n_per_insert = []
        self._locked = False
        self._cache_to_clear_on_update = {}

    def is_equivalent(PeriodicDelaunay3 self, PeriodicDelaunay3 solf):
        r"""Determine if two triangulations are equivalent. Currently only 
        checks that the triangulations have the same numbers of vertices, cells,
        edges, and facets.

        Args:
            solf (:class:`cgal4py.delaunay.PeriodicDelaunay3`): Triangulation 
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
            :class:`cgal4py.delaunay.PeriodicDelaunay3`: Triangulation
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
                :meth:`cgal4py.delaunay.PeriodicDelaunay3.deserialize`. 

        Returns: 
            :class:`cgal4py.delaunay.PeriodicDelaunay3`: Triangulation 
                constructed from deserialized information. 

        """
        T = cls()
        T.deserialize(*args)
        return T

    @classmethod
    def from_serial_buffer(cls, *args, **kwargs):
        r"""Create a triangulation from serialized information in a buffer.

        Args:
            See :meth:`cgal4py.delaunay.PeriodicDelaunay3.deserialize_from_buffer`.

        Returns:
            :class:`cgal4py.delaunay.PeriodicDelaunay3`: Triangulation
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
        cdef int ndim = 3
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
                                   "'{}' ".format(attr)+
                                   "while triangulation is locked.")
            solf._update_tess()
            if attr not in solf._cache_to_clear_on_update:
                solf._cache_to_clear_on_update[attr] = fget(solf)
            return solf._cache_to_clear_on_update[attr]
        return property(wrapped_fget, None, None, fget.__doc__)

    def is_valid(self):
        r"""Determine if the triangulation is a valid PeriodicDelaunay 
        triangulation. 

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
                    dimension (e.g. [xmin, ymin, zmin, xmax, ymax, zmax]).
                cover (np.ndarray of int32): Number of times points are 
                    replicated in each dimension to allow wrapping.  
                cells (np.ndarray of info_t): (n,m) Indices for m vertices in 
                    each of the n cells. A value of np.iinfo(np_info).max 
                    indicates the infinite vertex. 
                neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of 
                    the m neighbors to each of the n cells. 
                offsets (np.ndarray of int32): (n,m) Offset of m vertices in 
                    each of the ncells.
                idx_inf (I): Value representing the infinite vertex and or 
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
        d = 3
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
        d = 3
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
        # TODO: sort offsets
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
        d = 3
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
        # TODO: sort offsets
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
        d = 3
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
        # TODO: sort offsets
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
        d = 3
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
        # TODO: sort offsets
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
                    dimension (e.g. [xmin, ymin, zmin, xmax, ymax, zmax]).
                cover (np.ndarray of int32): Number of times points are 
                    replicated in each dimension to allow wrapping.
                cells (np.ndarray of info_t): (n,m) Indices for m vertices in 
                    each of the n cells. A value of np.iinfo(np_info).max 
                    indicates the infinite vertex. 
                neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of 
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

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
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
                dimension (e.g. [xmin, ymin, zmin, xmax, ymax, zmax]).
            cover (np.ndarray of int32): Number of times points are replicated 
                in each dimension to allow wrapping. 
            cells (np.ndarray of info_t): (n,m) Indices for m vertices in each 
                of the n cells. A value of np.iinfo(np_info).max A value of 
                np.iinfo(np_info).max indicates the infinite vertex. 
            neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of the m 
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
                dimension (e.g. [xmin, ymin, zmin, xmax, ymax, zmax]).
            cover (np.ndarray of int32): Number of times points are replicated 
                in each dimension to allow wrapping. 
            cells (np.ndarray of info_t): (n,m) Indices for m vertices in each 
                of the n cells. A value of np.iinfo(np_info).max A value of 
                np.iinfo(np_info).max indicates the infinite vertex. 
            neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of the m 
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
            *args: All arguments are passed to :func:`plot.plot3D`.
            **kwargs: All keyword arguments are passed to :func:`plot.plot3D`.

        """
        plot.plot3D(self, *args, **kwargs)

    @_dependent_property
    def num_sheets(self):
        r"""np.ndarray of int32: The number of times the original domain is 
        replicated in each dimension to allow wrapping around periodic 
        boundaries."""
        cdef np.ndarray[np.int32_t] ns = np.empty(3,'int32')
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
    def num_finite_facets(self):
        r"""int: The number of finite facets in the triangulation."""
        return self.T.num_finite_facets()
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
    def num_infinite_facets(self):
        r"""int: The number of infinite facets in the triangulation."""
        return self.T.num_infinite_facets()
    @_dependent_property
    def num_infinite_cells(self):
        r"""int: The number of infinite cells in the triangulation."""
        return self.T.num_infinite_cells()
    @_dependent_property
    def num_verts(self): 
        r"""int: The total number of vertices (finite + infinite) in the 
        triangulation."""
        return self.T.num_verts()
    @_dependent_property
    def num_edges(self):
        r"""int: The total number of edges (finite + infinite) in the 
        triangulation."""
        return self.T.num_edges()
    @_dependent_property
    def num_facets(self):
        r"""int: The total number of facets (finite + infinite) in the 
        triangulation."""
        return self.T.num_facets()
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
    def num_stored_facets(self):
        r"""int: The total number of facets (Finite + infinite) in the 
        triangulation including duplicates made to allow periodic 
        wrapping."""
        return self.T.num_stored_facets()
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
        cdef np.ndarray[np.float64_t] domain = np.empty(2*3, 'float64')
        domain[:3] = left_edge
        domain[3:] = right_edge
        self.T.set_domain(&domain[0])

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def insert(self, np.ndarray[double, ndim=2, mode="c"] pts not None):
        r"""Insert points into the triangulation.

        Args:
            pts (:obj:`ndarray` of :obj:`float64`): Array of 3D cartesian 
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
        assert(m == 3)
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
        r"""ndarray: The x,y,z coordinates of the vertices"""
        cdef np.ndarray[np.float64_t, ndim=2] out
        out = np.zeros([self.n, 3], 'float64')
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
            self.T.dual_volumes(&out[0])
        return out

    @_update_to_tess
    def remove(self, PeriodicDelaunay3_vertex x):
        r"""Remove a vertex from the triangulation. 

        Args: 
            x (PeriodicDelaunay3_vertex): Vertex that should be removed. 

        """
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.remove(x.x)

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def move(self, PeriodicDelaunay3_vertex x, 
             np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location. If there is a vertex at the given 
        given coordinates, return that vertex and remove the one that was being 
        moved. 

        Args: 
            x (PeriodicDelaunay3_vertex): Vertex that should be moved. 
            pos (:obj:`ndarray` of float64): x,y,z coordinates that the vertex 
                be moved to. 

        Returns: 
            PeriodicDelaunay3_vertex: Vertex at the new position. 

        """
        assert(len(pos) == 3)
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex v
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            v = self.T.move(x.x, &pos[0])
        cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
        out.assign(self.T, v)
        return out

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def move_if_no_collision(self, PeriodicDelaunay3_vertex x,
                             np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location only if there is not already a 
        vertex at the given coordinates. If there is a vertex there, it is 
        returned and the vertex being moved remains at its original location. 

        Args: 
            x (PeriodicDelaunay3_vertex): Vertex that should be moved. 
            pos (:obj:`ndarray` of float64): x,y,z coordinates that the vertex 
                be moved to. 

        Returns: 
            PeriodicDelaunay2_vertex: Vertex at the new position. 

        """
        assert(len(pos) == 3)
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex v
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            v = self.T.move_if_no_collision(x.x, &pos[0])
        cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
        out.assign(self.T, v)
        return out

    def get_vertex(self, np_info_t index):
        r"""Get the vertex object corresponding to the given index. 

        Args: 
            index (np_info_t): Index of vertex that should be found. 

        Returns: 
            PeriodicDelaunay3_vertex: Vertex corresponding to the given index. 
                If the index is not found, the infinite vertex is returned. 

        """
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex v
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            v = self.T.get_vertex(index)
        cdef PeriodicDelaunay3_vertex out = PeriodicDelaunay3_vertex()
        out.assign(self.T, v)
        return out

    def locate(self, np.ndarray[np.float64_t, ndim=1] pos,
               PeriodicDelaunay3_cell start = None):
        r"""Get the vertex/cell/facet/edge that a given point is a part of.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.   
            start (PeriodicDelaunay3_cell, optional): Cell to start the search 
                at. 

        Returns:
            Object associated with the search point (vertex, edge, facet, or 
                cell).

        """
        assert(len(pos) == 3)
        cdef int lt, li, lj
        lt = li = lj = 999
        cdef PeriodicDelaunay3_cell c = PeriodicDelaunay3_cell()
        if start is not None:
            c.assign(self.T, self.T.locate(&pos[0], lt, li, lj, start.x))
        else:
            c.assign(self.T, self.T.locate(&pos[0], lt, li, lj))
        print(lt)
        assert(lt != 999)
        if lt < 2:
            assert(li != 999)
        if lt == 0: # vertex
            return c.vertex(li)
        elif lt == 1: # edge
            return PeriodicDelaunay3_edge.from_cell(c, li, lj)
        elif lt == 2: # facet
            return PeriodicDelaunay3_facet.from_cell(c, li)
        elif lt == 3: # cell
            return c
        elif lt == 4:
            print("Point {} is outside the convex hull.".format(pos))
            return c
        elif lt == 5:
            print("Point {} is outside the affine hull.".format(pos))
            return 0
        else:
            raise RuntimeError("Value of {} ".format(lt)+
                               "not expected from CGAL locate.")

    @property
    def all_verts_begin(self):
        r"""PeriodicDelaunay3_vertex_iter: Starting vertex for all vertices in 
        the triangulation."""
        return PeriodicDelaunay3_vertex_iter(self, 'all_begin')
    @property
    def all_verts_end(self):
        r"""PeriodicDelaunay3_vertex_iter: Final vertex for all vertices in the 
        triangulation."""
        return PeriodicDelaunay3_vertex_iter(self, 'all_end')
    @property
    def all_verts(self):
        r"""PeriodicDelaunay3_vertex_range: Iterable for all vertices in the 
        triangulation."""
        return PeriodicDelaunay3_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end)
    @property
    def finite_verts(self):
        r"""PeriodicDelaunay3_vertex_range: Iterable for finite vertices in the 
        triangulation."""
        return PeriodicDelaunay3_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end, finite = True)

    @property
    def all_edges_begin(self):
        r"""PeriodicDelaunay3_edge_iter: Starting edge for all edges in the 
        triangulation."""
        return PeriodicDelaunay3_edge_iter(self, 'all_begin')
    @property
    def all_edges_end(self):
        r"""PeriodicDelaunay3_edge_iter: Final edge for all edges in the 
        triangulation."""
        return PeriodicDelaunay3_edge_iter(self, 'all_end')
    @property
    def all_edges(self):
        r"""PeriodicDelaunay3_edge_range: Iterable for all edges in the 
        triangulation."""
        return PeriodicDelaunay3_edge_range(self.all_edges_begin,
                                    self.all_edges_end)
    @property
    def finite_edges(self):
        r"""PeriodicDelaunay3_edge_range: Iterable for finite edges in the 
        triangulation."""
        return PeriodicDelaunay3_edge_range(self.all_edges_begin,
                                    self.all_edges_end, finite = True)

    @property
    def all_facets_begin(self):
        r"""PeriodicDelaunay3_facet_iter: Starting facet for all facets in the 
        triangulation."""
        return PeriodicDelaunay3_facet_iter(self, 'all_begin')
    @property
    def all_facets_end(self):
        r"""PeriodicDelaunay3_facet_iter: Final facet for all facets in the 
        triangulation."""
        return PeriodicDelaunay3_facet_iter(self, 'all_end')
    @property
    def all_facets(self):
        r"""PeriodicDelaunay3_facet_range: Iterable for all facets in the 
        triangulation."""
        return PeriodicDelaunay3_facet_range(self.all_facets_begin,
                                     self.all_facets_end)
    @property
    def finite_facets(self):
        r"""PeriodicDelaunay3_facet_range: Iterable for finite facets in the 
        triangulation."""
        return PeriodicDelaunay3_facet_range(self.all_facets_begin,
                                     self.all_facets_end, finite = True)

    @property
    def all_cells_begin(self):
        r"""PeriodicDelaunay3_cell_iter: Starting cell for all cells in the 
        triangulation."""
        return PeriodicDelaunay3_cell_iter(self, 'all_begin')
    @property
    def all_cells_end(self):
        r"""PeriodicDelaunay3_cell_iter: Final cell for all cells in the 
        triangulation."""
        return PeriodicDelaunay3_cell_iter(self, 'all_end')
    @property
    def all_cells(self):
        r"""PeriodicDelaunay3_cell_range: Iterable for all cells in the
        triangulation."""
        return PeriodicDelaunay3_cell_range(self.all_cells_begin,
                                    self.all_cells_end)
    @property
    def finite_cells(self):
        r"""PeriodicDelaunay3_cell_range: Iterable for finite cells in the
        triangulation."""
        return PeriodicDelaunay3_cell_range(self.all_cells_begin,
                                    self.all_cells_end, finite = True)

    def is_edge(self, PeriodicDelaunay3_vertex v1, PeriodicDelaunay3_vertex v2,
                PeriodicDelaunay3_cell c = PeriodicDelaunay3_cell(), 
                int i1 = 0, int i2 = 0):
        r"""Determine if two vertices form an edge in the triangulation.  

        Args:
            v1 (PeriodicDelaunay3_vertex): First vertex. 
            v2 (PeriodicDelaunay3_vertex): Second vertex.
            c (PeriodicDelaunay3_cell, optional): If provided and the two 
                vertices form an edge, the cell incident to the edge is stored 
                here.
            i1 (int, optional): If provided and the two vertices form an edge,
                the index of v1 in cell c is stored here.
            i2 (int, optional): If provided and the two vertices form an edge,
                the index of v2 in cell c is stored here.

        Returns:
            bool: True if v1 and v2 form an edge, False otherwise. 

        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.is_edge(v1.x, v2.x, c.x, i1, i2)
        return <pybool>out

    def is_facet(self, PeriodicDelaunay3_vertex v1, PeriodicDelaunay3_vertex v2,
                 PeriodicDelaunay3_vertex v3, 
                 PeriodicDelaunay3_cell c = PeriodicDelaunay3_cell(),
                 int i1 = 0, int i2 = 0, int i3 = 0):
        r"""Determine if two vertices form a facet in the triangulation.  

        Args:
            v1 (PeriodicDelaunay3_vertex): First vertex. 
            v2 (PeriodicDelaunay3_vertex): Second vertex.
            v3 (PeriodicDelaunay3_vertex): Third vertex.
            c (PeriodicDelaunay3_cell, optional): If provided and the two 
                vertices form a facet, the cell incident to the facet is stored 
                here.
            i1 (int, optional): If provided and the two vertices form a facet,
                the index of v1 in cell c is stored here.
            i2 (int, optional): If provided and the two vertices form a facet,
                the index of v2 in cell c is stored here.
            i3 (int, optional): If provided and the two vertices form a facet,
                the index of v3 in cell c is stored here.

        Returns:
            bool: True if v1, v2, and v3 form a facet, False otherwise. 

        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.is_facet(v1.x, v2.x, v3.x, c.x, i1, i2, i3)
        return <pybool>out

    def is_cell(self, PeriodicDelaunay3_vertex v1, PeriodicDelaunay3_vertex v2,
                PeriodicDelaunay3_vertex v3, PeriodicDelaunay3_vertex v4, 
                PeriodicDelaunay3_cell c = PeriodicDelaunay3_cell(),
                int i1 = 0, int i2 = 0, int i3 = 0, int i4 = 0):
        r"""Determine if two vertices form a cell in the triangulation.  

        Args:
            v1 (PeriodicDelaunay3_vertex): First vertex. 
            v2 (PeriodicDelaunay3_vertex): Second vertex.
            v3 (PeriodicDelaunay3_vertex): Third vertex.
            v4 (PeriodicDelaunay3_vertex): Fourth vertex.
            c (PeriodicDelaunay3_cell, optional): If provided and the two 
                vertices form a cell, the cell they form is stored here.
            i1 (int, optional): If provided and the two vertices form a cell,
                the index of v1 in cell c is stored here.
            i2 (int, optional): If provided and the two vertices form a cell,
                the index of v2 in cell c is stored here.
            i3 (int, optional): If provided and the two vertices form a cell,
                the index of v3 in cell c is stored here.
            i3 (int, optional): If provided and the two vertices form a cell,
                the index of v4 in cell c is stored here.

        Returns:
            bool: True if v1, v2, v3, and v4 form a cell, False otherwise. 

        """
        cdef cbool out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            out = self.T.is_cell(v1.x, v2.x, v3.x, v4.x,
                                 c.x, i1, i2, i3, i4)
        return <pybool>out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def nearest_vertex(self, np.ndarray[np.float64_t, ndim=1] x):
        r"""Determine which vertex is closes to a given set of x,y coordinates.

        Args:
            x (:obj:`ndarray` of float64): x,y coordinates. 

        Returns: 
            PeriodicDelaunay3_vertex: Vertex closest to x. 

        """
        assert(len(x) == 3)
        cdef PeriodicDelaunay_with_info_3[info_t].Vertex vc
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            vc = self.T.nearest_vertex(&x[0])
        cdef PeriodicDelaunay3_vertex v = PeriodicDelaunay3_vertex()
        v.assign(self.T, vc)
        return v

    def mirror_facet(self, PeriodicDelaunay3_facet x):
        r"""Get the same facet as referenced from its other incident cell. 

        Args:
            x (PeriodicDelaunay3_facet): Facet to mirror.

        Returns:
            PeriodicDelaunay3_facet: The same facet as x, but referenced from 
                the other cell incident to x.

        """
        return x.mirror

    def mirror_index(self, PeriodicDelaunay3_cell x, int i):
        r"""Get the index of a cell with respect to its ith neighbor. 

        Args: 
            x (PeriodicDelaunay3_cell): Cell to get mirrored index for. 
            i (int): Index of neighbor that should be used to determine the 
                mirrored index. 

        Returns: 
            int: Index of cell x with respect to its ith neighbor. 

        """
        return x.mirror_index(i)

    def mirror_vertex(self, PeriodicDelaunay3_cell x, int i):
        r"""Get the vertex of a cell's ith neighbor opposite to the cell. 

        Args: 
            x (PeriodicDelaunay3_cell): Cell. 
            i (int): Index of neighbor that should be used to determine the 
                mirrored vertex. 

        Returns:
            PeriodicDelaunay3_vertex: Vertex in the ith neighboring cell of cell 
                x, that is opposite to cell x. 

        """
        return x.mirror_vertex(i)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_boundary_of_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                                  PeriodicDelaunay3_cell start):
        r"""Get the facets bounding the zone in conflict with a given point. 

        Args: 
            pos (:obj:`ndarray` of float64): x,y,z coordinates. 
            start (PeriodicDelaunay3_cell): Cell to start list of facets at. 

        Returns: 
            :obj:`list` of PeriodicDelaunay3_facet: Facets bounding the zone in 
                 conflict with pos. 

        """
        assert(len(pos) == 3)
        cdef pair[vector[PeriodicDelaunay_with_info_3[info_t].Cell],
                  vector[PeriodicDelaunay_with_info_3[info_t].Facet]] cv
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            cv = self.T.find_conflicts(&pos[0], start.x)
        cdef object out_facets = []
        cdef np.uint32_t i
        cdef PeriodicDelaunay3_facet f
        for i in range(cv.second.size()):
            f = PeriodicDelaunay3_facet()
            f.assign(self.T, cv.second[i])
            out_facets.append(f)
        return out_facets

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                      PeriodicDelaunay3_cell start):
        r"""Get the cells in conflict with a given point.

        Args: 
            pos (:obj:`ndarray` of float64): x,y,z coordinates. 
            start (PeriodicDelaunay3_cell): Cell to start list of facets at. 

        Returns: 
            :obj:`list` of PeriodicDelaunay3_cell: Cells in conflict with pos. 

        """
        assert(len(pos) == 3)
        cdef pair[vector[PeriodicDelaunay_with_info_3[info_t].Cell],
                  vector[PeriodicDelaunay_with_info_3[info_t].Facet]] cv
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            cv = self.T.find_conflicts(&pos[0], start.x)
        cdef object out_cells = []
        cdef np.uint32_t i
        cdef PeriodicDelaunay3_cell c
        for i in range(cv.first.size()):
            c = PeriodicDelaunay3_cell()
            c.assign(self.T, cv.first[i])
            out_cells.append(c)
        return out_cells

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_conflicts_and_boundary(self, np.ndarray[np.float64_t, ndim=1] pos,
                                   PeriodicDelaunay3_cell start):
        r"""Get the cells in conflict with a given point and the facets bounding 
            the zone in conflict.

        Args: 
            pos (:obj:`ndarray` of float64): x,y,z coordinates. 
            start (PeriodicDelaunay3_cell): Cell to start list of facets at. 

        Returns: 
            tuple: :obj:`list` of PeriodicDelaunay3_cells in conflict with pos 
                and :obj:`list` of PeriodicDelaunay3_facets bounding the zone 
                in conflict.

        """
        assert(len(pos) == 3)
        cdef pair[vector[PeriodicDelaunay_with_info_3[info_t].Cell],
                  vector[PeriodicDelaunay_with_info_3[info_t].Facet]] cv
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            cv = self.T.find_conflicts(&pos[0], start.x)
        cdef object out_facets = []
        cdef object out_cells = []
        cdef np.uint32_t i
        cdef PeriodicDelaunay3_cell c
        cdef PeriodicDelaunay3_facet f
        for i in range(cv.first.size()):
            c = PeriodicDelaunay3_cell()
            c.assign(self.T, cv.first[i])
            out_cells.append(c)
        for i in range(cv.second.size()):
            f = PeriodicDelaunay3_facet()
            f.assign(self.T, cv.second[i])
            out_facets.append(f)
        return out_cells, out_facets

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
        assert(left_edges.shape[1] == 3)
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
        assert(len(left_edge)==3)
        assert(len(right_edge)==3)
        global np_info
        cdef int i, j, k
        cdef vector[info_t] lr, lx, ly, lz, rx, ry, rz, alln
        cdef cbool cperiodic = <cbool>periodic
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.boundary_points(&left_edge[0], &right_edge[0], cperiodic,
                                   lx, ly, lz, rx, ry, lz, alln)
        # Get counts to preallocate 
        cdef object lind = [None, None, None]
        cdef object rind = [None, None, None]
        cdef info_t iN = 0
        for i in range(3):
            if   i == 0: lr = lx
            elif i == 1: lr = ly
            elif i == 2: lr = lz
            iN = <info_t>lr.size()
            lind[i] = np.zeros(iN, np_info)
        for i in range(3):
            if   i == 0: lr = rx
            elif i == 1: lr = ry
            elif i == 2: lr = rz
            iN = <info_t>lr.size()
            rind[i] = np.zeros(iN, np_info)
        # Fill in 
        cdef np.ndarray[info_t] iind
        cdef np.ndarray[info_t] lr_arr
        iN = alln.size()
        iind = np.array([alln[j] for j in range(<int>iN)], np_info)
        for i in range(3):
            if   i == 0: lr = lx
            elif i == 1: lr = ly
            elif i == 2: lr = lz
            iN = <info_t>lr.size()
            lr_arr = np.array([lr[j] for j in range(<int>iN)], np_info)
            lind[i] = lr_arr
        for i in range(3):
            if   i == 0: lr = rx
            elif i == 1: lr = ry
            elif i == 2: lr = rz
            iN = <info_t>lr.size()
            lr_arr = np.array([lr[j] for j in range(<int>iN)], np_info)
            rind[i] = lr_arr
        # Return
        return lind, rind, iind
