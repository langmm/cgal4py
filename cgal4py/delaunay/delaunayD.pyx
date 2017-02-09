"""
delaunayD.pyx

Wrapper for CGAL 3D Delaunay Triangulation
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

cdef class DelaunayD_vertex:
    r"""Wrapper class for a triangulation vertex.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
            that this vertex belongs to. 
        x (:obj:`Delaunay_with_info_D[info_t].Vertex`): C++ vertex object 
            Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef Delaunay_with_info_D[info_t].Vertex x

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     Delaunay_with_info_D[info_t].Vertex x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
                that this vertex belongs to. 
            x (:obj:`Delaunay_with_info_D[info_t].Vertex`): C++ vertex object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    @property
    def num_dims(self):
        r"""int: Number of dimensions."""
        return D

    def __repr__(self):
        cdef str out = "DelaunayD_vertex[{} at ".format(self.index)
        cdef np.ndarray[np.float64_t] p = self.point
        for i in range(self.num_dims-1):
            out += "{:+7.2e},".format(p[i])
        out += "{:+7.2e}]".format(p[self.num_dims-1])
        return out

    def __richcmp__(DelaunayD_vertex self, DelaunayD_vertex solf, int op):
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
        r"""Set this vertex's corrdinates.

        Args:
            pos (:obj:`ndarray` of float64): new coordinates for vertex.

        """
        self.T.updated = <cbool>True
        assert(len(pos) == self.num_dims)
        self.x.set_point(&pos[0])

    def set_cell(self, DelaunayD_cell c):
        r"""Set the designated cell for this vertex.

        Args:
            c (DelaunayD_cell): Cell that should be set as the designated cell.

        """
        self.T.updated = <cbool>True
        self.x.set_cell(c.x)

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian coordinates of the
        vertex."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out
            out = np.zeros(D, 'float64')
            if self.is_infinite():
                out[:] = np.float('inf')
            else:
                self.x.point(&out[0])
            return out

    property index:
        r"""info_t: The index of the vertex point in the input array."""
        def __get__(self):
            cdef info_t out
            if self.is_infinite():
                out = <info_t>np.iinfo(np_info).max
            else:
                out = self.x.info()
            return out

    property dual_volume:
        r"""float64: The volume of the dual Voronoi cell. If the volume is 
        infinite, -1.0 is returned."""
        def __get__(self):
            cdef np.float64_t out = self.T.dual_volume(self.x)
            return out

    property cell:
        r"""DelaunayD_cell: Designated cell for this vertex."""
        def __get__(self):
            cdef Delaunay_with_info_D[info_t].Cell c
            c = self.x.cell()
            cdef DelaunayD_cell out = DelaunayD_cell()
            out.assign(self.T, c)
            return out

    def incident_vertices(self):
        r"""Find vertices that are adjacent to this vertex.

        Returns:
            DelaunayD_vertex_vector: Iterator over vertices incident to this 
                vertex.

        """
        cdef vector[Delaunay_with_info_D[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef DelaunayD_vertex_vector out = DelaunayD_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_faces(self, int i):
        r"""Find faces that are incident to this vertex.

        Args:
            int i: The dimensionality of the faces that should be returned.

        Returns:
            DelaunayD_face_vector: Iterator over faces incident to this
                vertex.

        """
        cdef vector[Delaunay_with_info_D[info_t].Face] it
        it = self.T.incident_faces(self.x, i)
        cdef DelaunayD_face_vector out = DelaunayD_face_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this vertex.

        Returns:
            DelaunayD_cell_vector: Iterator over cells incident to this vertex.

        """
        cdef vector[Delaunay_with_info_D[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef DelaunayD_cell_vector out = DelaunayD_cell_vector()
        out.assign(self.T, it)
        return out


cdef class DelaunayD_vertex_iter:
    r"""Wrapper class for a triangulation vertex iterator.

    Args:
        T (DelaunayD): Triangulation that this vertex belongs to.
        vert (:obj:`str`, optional): String specifying the vertex that 
            should be referenced. Valid options include: 
                'all_begin': The first vertex in an iteration over all vertices.  
                'all_end': The last vertex in an iteration over all vertices. 
 
    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
            that this vertex belongs to. 
        x (:obj:`Delaunay_with_info_D[info_t].All_verts_iter`): C++ vertex 
            object. Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef Delaunay_with_info_D[info_t].All_verts_iter x

    def __cinit__(self, DelaunayD T, str vert = None):
        self.T = T.T
        if vert == 'all_begin':
            self.x = self.T.all_verts_begin()
        elif vert == 'all_end':
            self.x = self.T.all_verts_end()

    def __richcmp__(DelaunayD_vertex_iter self, DelaunayD_vertex_iter solf, 
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
        r"""DelaunayD_vertex: Corresponding vertex object."""
        def __get__(self):
            cdef DelaunayD_vertex out = DelaunayD_vertex()
            out.assign(self.T, Delaunay_with_info_D[info_t].Vertex(self.x)) 
            return out


cdef class DelaunayD_vertex_range:
    r"""Wrapper class for iterating over a range of triangulation vertices

    Args:
        xstart (DelaunayD_vertex_iter): The starting vertex.  
        xstop (DelaunayD_vertex_iter): Final vertex that will end the iteration. 
        finite (:obj:`bool`, optional): If True, only finite verts are 
            iterated over. Otherwise, all verts are iterated over. Defaults 
            False.

    Attributes:
        x (DelaunayD_vertex_iter): The current vertex. 
        xstop (DelaunayD_vertex_iter): Final vertex that will end the iteration. 
        finite (bool): If True, only finite verts are iterater over. Otherwise
            all verts are iterated over. 

    """
    cdef DelaunayD_vertex_iter x
    cdef DelaunayD_vertex_iter xstop
    cdef pybool finite
    def __cinit__(self, DelaunayD_vertex_iter xstart, 
                  DelaunayD_vertex_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        cdef DelaunayD_vertex out
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        if self.x != self.xstop:
            out = self.x.vertex
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class DelaunayD_vertex_vector:
    r"""Wrapper class for a vector of vertices.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_D[info_t].Vertex]`): Vector of C++ 
            vertices.
        n (int): The number of vertices in the vector.
        i (int): The index of the currect vertex.

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef vector[Delaunay_with_info_D[info_t].Vertex] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     vector[Delaunay_with_info_D[info_t].Vertex] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
                Direct interaction with this object is not recommended.
            v (:obj:`vector[Delaunay_with_info_D[info_t].Vertex]`): Vector of 
                C++ vertices.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef DelaunayD_vertex out
        if self.i < self.n:
            out = DelaunayD_vertex()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef DelaunayD_vertex out
        if isinstance(i, int):
            out = DelaunayD_vertex()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("DelaunayD_vertex_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class DelaunayD_face:
    r"""Wrapper class for a triangulation face.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
            that this face belongs to. 
        x (:obj:`Delaunay_with_info_D[info_t].Face`): C++ face object 
            Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef Delaunay_with_info_D[info_t].Face x

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     Delaunay_with_info_D[info_t].Face x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
                that this face belongs to. 
            x (:obj:`Delaunay_with_info_D[info_t].Face`): C++ face object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    @property
    def dim(self):
        r"""int: Number of dimensions this face occupies."""
        return self.x.dim()

    @property
    def nverts(self):
        r"""int: Number of vertices in this face (self.ndim+1)."""
        return self.dim+1

    def __repr__(self):
        cdef str out = "DelaunayD_face[{}d: ".format(self.dim)
        cdef int i
        for i in range(self.nverts-1):
            out += "{},".format(repr(self.vertex(i)))
        out += "{}]".format(repr(self.vertex(self.nverts-1)))
        return out

    # def __richcmp__(DelaunayD_face self, DelaunayD_face solf, int op):
    #     if (op == 2):
    #         return <pybool>(self.x == solf.x)
    #     elif (op == 3):
    #         return <pybool>(self.x != solf.x)
    #     else:
    #         raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the face is incident to the infinite vertex.
        
        Returns:
            bool: True if the face is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def is_equivalent(self, DelaunayD_face solf):
        r"""Determine if another face has the same vertices as this face.

        Args:
            solf (DelaunayD_face): Face for comparison.

        Returns:
            bool: True if the two faces share the same vertices, False
                otherwise.

        """
        return <pybool>self.T.are_equal(self.x, solf.x)

    def vertex(self, int i):
        r"""Get the ith vertex on this face.

        Args:
            i (int): Index of vertex to return.

        Returns:
            DelaunayD_vertex: ith vertex on this face.

        """
        cdef Delaunay_with_info_D[info_t].Vertex x
        x = self.x.vertex(i)
        cdef DelaunayD_vertex out = DelaunayD_vertex()
        out.assign(self.T, x)
        return out

    def index(self, int i):
        r"""Get the index in the cell of the ith vertex on this face.

        Args:
            i (int): Index of vertex on this face to get index for.
        
        Returns:
            int: Index of ith vertex on this face within the reference cell.

        """
        cdef int out
        out = self.x.ind(i)
        return out

    property cell:
        r"""DelaunayD_cell: The cell this face is assigned to."""
        def __get__(self):
            cdef Delaunay_with_info_D[info_t].Cell c
            c = self.x.cell()
            cdef DelaunayD_cell out = DelaunayD_cell()
            out.assign(self.T, c)
            return out

    property center:
        r""":obj:`ndarray` of float64: coordinates of face center."""
        def __get__(self):
            if self.is_infinite():
                return np.float('inf')*np.ones(3, 'float64')
            else:
                ptot = self.vertex(0).point
                for i in range(1,self.nverts):
                    ptot += self.vertex(i).point
                return ptot/self.nverts

    property area:
        r"""float64: The area of the face. If infinite, -1 is returned"""
        def __get__(self):
            # return self.T.n_simplex_volume(self.x)
            raise NotImplementedError

    def incident_vertices(self):
        r"""Find vertices that are incident to this face.

        Returns:
            DelaunayD_vertex_vector: Iterator over vertices incident to this 
                face.

        """
        cdef vector[Delaunay_with_info_D[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef DelaunayD_vertex_vector out = DelaunayD_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_faces(self, int i):
        r"""Find faces that are incident to this face.

        Args:
            int i: Dimensionality of faces that should be returned.

        Returns:
            DelaunayD_face_vector: Iterator over faces incident to this face. 

        """
        cdef vector[Delaunay_with_info_D[info_t].Face] it
        it = self.T.incident_faces(self.x, i)
        cdef DelaunayD_face_vector out = DelaunayD_face_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this face.

        Returns:
            DelaunayD_cell_vector: Iterator over cells incident to this face.

        """
        cdef vector[Delaunay_with_info_D[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef DelaunayD_cell_vector out = DelaunayD_cell_vector()
        out.assign(self.T, it)
        return out


cdef class DelaunayD_face_vector:
    r"""Wrapper class for a vector of faces.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_D[info_t].Face]`): Vector of C++ 
            faces.
        n (int): The number of faces in the vector.
        i (int): The index of the currect face.

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef vector[Delaunay_with_info_D[info_t].Face] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     vector[Delaunay_with_info_D[info_t].Face] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
                Direct interaction with this object is not recommended.
            v (:obj:`vector[Delaunay_with_info_D[info_t].Face]`): Vector of 
                C++ faces.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef DelaunayD_face out
        if self.i < self.n:
            out = DelaunayD_face()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef DelaunayD_face out
        if isinstance(i, int):
            out = DelaunayD_face()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("DelaunayD_face_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class DelaunayD_facet:
    r"""Wrapper class for a triangulation facet.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
            that this facet belongs to. 
        x (:obj:`Delaunay_with_info_D[info_t].Facet`): C++ facet object 
            Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef Delaunay_with_info_D[info_t].Facet x

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     Delaunay_with_info_D[info_t].Facet x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
                that this facet belongs to. 
            x (:obj:`Delaunay_with_info_D[info_t].Facet`): C++ facet object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    @property
    def dim(self):
        r"""int: Number of dimensions that the facet covers (one less than
        overall domain)."""
        return D - 1

    @property
    def nverts(self):
        r"""int: Number of vertices in the facet (same as number of dimensions
        in the domain."""
        return D

    @staticmethod
    def from_cell(DelaunayD_cell c, int i):
        r"""Construct a facet from a cell and index of the vertex in the cell 
        opposite the desired facet.

        Args:
            c (DelaunayD_cell): Cell 
            i (int): Index of vertex in c that is opposite the facet.

        Returns:
            DelaunayD_facet: Facet incident to c and opposite vertex i in c.

        """
        cdef DelaunayD_facet out = DelaunayD_facet()
        cdef Delaunay_with_info_D[info_t].Facet e
        e = Delaunay_with_info_D[info_t].Facet(c.x, i)
        out.assign(c.T, e)
        return out

    def __repr__(self):
        cdef str out = "DelaunayD_facet["
        cdef int i
        for i in range(self.nverts-1):
            out += "{},".format(repr(self.vertex(i)))
        out += "{}]".format(repr(self.vertex(self.nverts-1)))
        return out

    # def __richcmp__(DelaunayD_facet self, DelaunayD_facet solf, int op):
    #     if (op == 2):
    #         return <pybool>(self.x == solf.x)
    #     elif (op == 3):
    #         return <pybool>(self.x != solf.x)
    #     else:
    #         raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the facet is incident to the infinite vertex.
        
        Returns:
            bool: True if the facet is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def is_equivalent(self, DelaunayD_facet solf):
        r"""Determine if another facet has the same vertices as this facet.

        Args:
            solf (DelaunayD_facet): Facet for comparison.

        Returns:
            bool: True if the two facets share the same vertices, False 
                otherwise.

        """
        return <pybool>self.T.are_equal(self.x, solf.x)

    def vertex(self, int i):
        r"""Get the ith vertex incident to this facet.

        Args:
            i (int): Index of vertex that should be returned.

        Returns:
            Delaunay_vertex: ith vertex of this facet.

        """
        cdef Delaunay_with_info_D[info_t].Vertex v
        v = self.x.vertex(i)
        cdef DelaunayD_vertex out = DelaunayD_vertex()
        out.assign(self.T, v)
        return out

    property center:
        r""":obj:`ndarray` of float64: x,y,z coordinates of cell center."""
        def __get__(self):
            if self.is_infinite():
                return np.float('inf')*np.ones(self.dim, 'float64')
            else:
                ptot = self.vertex(0).point
                for i in range(1,self.nverts):
                    ptot += self.vertex(i).point
                return ptot/self.nverts
                        
    property area:
        r"""float64: The area of the facet. If infinite, -1 is returned"""
        def __get__(self):
            # return self.T.n_simplex_volume(self.x)
            raise NotImplementedError

    property cell:
        r"""DelaunayD_cell: The cell this facet is assigned to."""
        def __get__(self):
            cdef Delaunay_with_info_D[info_t].Cell c
            c = self.x.cell()
            cdef DelaunayD_cell out = DelaunayD_cell()
            out.assign(self.T, c)
            return out

    property ind:
        r"""int: The index of the vertex this facet is opposite on its cell."""
        def __get__(self):
            return self.x.ind()


cdef class DelaunayD_facet_vector:
    r"""Wrapper class for a vector of facets.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_D[info_t].Facet]`): Vector of C++ 
            facets.
        n (int): The number of facets in the vector.
        i (int): The index of the currect facet.

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef vector[Delaunay_with_info_D[info_t].Facet] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     vector[Delaunay_with_info_D[info_t].Facet] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
                Direct interaction with this object is not recommended.
            v (:obj:`vector[Delaunay_with_info_D[info_t].Facet]`): Vector of 
                C++ facets.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef DelaunayD_facet out
        if self.i < self.n:
            out = DelaunayD_facet()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef DelaunayD_facet out
        if isinstance(i, int):
            out = DelaunayD_facet()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("DelaunayD_facet_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class DelaunayD_cell:
    r"""Wrapper class for a triangulation cell.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
            that this cell belongs to. 
        x (:obj:`Delaunay_with_info_D[info_t].Cell`): C++ cell object. 
            Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef Delaunay_with_info_D[info_t].Cell x

    @property
    def dim(self):
        r"""int: Number of dimensions that the facet covers (one less than
        overall domain)."""
        return self.T.num_dims()

    @property
    def nverts(self):
        r"""int: Number of vertices in the facet (same as number of dimensions
        in the domain."""
        return self.dim + 1

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     Delaunay_with_info_D[info_t].Cell x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
                that this cell belongs to. 
            x (:obj:`Delaunay_with_info_D[info_t].Cell`): C++ cell object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    def __repr__(self):
        cdef str out = "Delaunay2_cell["
        cdef int i
        for i in range(self.nverts-1):
            out += "{},".format(repr(self.vertex(i)))
        out += "{}]".format(repr(self.vertex(self.nverts-1)))
        return out

    def __richcmp__(DelaunayD_cell self, DelaunayD_cell solf, int op):
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
            DelaunayD_vertex: Vertex in the ith neighboring cell of this cell, 
                that is opposite to this cell. 

        """
        cdef Delaunay_with_info_D[info_t].Vertex vc
        vc = self.T.mirror_vertex(self.x, i)
        cdef DelaunayD_vertex out = DelaunayD_vertex()
        out.assign(self.T, vc)
        return out

    def facet(self, int i):
        r"""Find the facet opposite the ith vertex incident to this cell.

        Args:
            i (int): Index of vertex opposite the desired facet.

        Returns:
            DelaunayD_facet: The facet opposite the ith vertex incident to this 
                cell.

        """
        cdef DelaunayD_facet out = DelaunayD_facet.from_cell(self, i)
        return out

    def vertex(self, int i):
        r"""Find the ith vertex that is incident to this cell. 

        Args:
            i (int): The index of the vertex that should be returned.

        Returns:
            DelaunayD_vertex: The ith vertex incident to this cell. 

        """
        cdef Delaunay_with_info_D[info_t].Vertex v
        v = self.x.vertex(i)
        cdef DelaunayD_vertex out = DelaunayD_vertex()
        out.assign(self.T, v)
        return out

    def has_vertex(self, DelaunayD_vertex v, pybool return_index = False):
        r"""Determine if a vertex belongs to this cell.

        Args:
            v (DelaunayD_vertex): Vertex to test ownership for. 
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
            
    def ind_vertex(self, DelaunayD_vertex v):
        r"""Determine the index of a vertex within a cell. 

        Args: 
            v (DelaunayD_vertex): Vertex to find index for. 

        Returns: 
            int: Index of vertex within the cell. 

        """
        return self.x.ind(v.x)

    def neighbor(self, int i):
        r"""Find the neighboring cell opposite the ith vertex of this cell. 

        Args: 
            i (int): The index of the neighboring cell that should be returned. 

        Returns: 
            Delaunay2_cell: The neighboring cell opposite the ith vertex. 

        """
        cdef Delaunay_with_info_D[info_t].Cell v
        v = self.x.neighbor(i)
        cdef DelaunayD_cell out = DelaunayD_cell()
        out.assign(self.T, v)
        return out

    def has_neighbor(self, DelaunayD_cell v, pybool return_index = False):
        r"""Determine if a cell is a neighbor to this cell. 

        Args: 
            v (DelaunayD_cell): Cell to test as a neighbor. 
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

    def ind_neighbor(self, DelaunayD_cell v):
        r"""Determine the index of a neighboring cell. 

        Args: 
            v (DelaunayD_cell): Neighboring cell to find index for. 

        Returns: 
            int: Index of vertex opposite to neighboring cell. 

        """
        return self.x.ind(v.x)

    def set_vertex(self, int i, DelaunayD_vertex v):
        r"""Set the ith vertex of this cell. 

        Args: 
            i (int): Index of this cell's vertex that should be set. 
            v (DelaunayD_vertex): Vertex to set ith vertex of this cell to. 

        """
        self.T.updated = <cbool>True
        self.x.set_vertex(i, v.x)

    def set_neighbor(self, int i, DelaunayD_cell n):
        r"""Set the ith neighboring cell of this cell. 

        Args: 
            i (int): Index of this cell's neighbor that should be set. 
            n (DelaunayD_cell): Cell to set ith neighbor of this cell to. 

        """
        self.T.updated = <cbool>True
        self.x.set_neighbor(i, n.x)

    property center:
        """:obj:`ndarray` of float64: coordinates of cell center."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] ptot
            cdef int i
            if self.is_infinite():
                return np.float('inf')*np.ones(self.dim, 'float64')
            else:
                ptot = self.vertex(0).point
                for i in range(1, self.nverts):
                    ptot += self.vertex(i).point
                return ptot/self.nverts

    property circumcenter:
        """:obj:`ndarray` of float64: coordinates of cell circumcenter."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(self.dim, 'float64')
            self.T.circumcenter(self.x, &out[0])
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this cell.

        Returns:
            DelaunayD_vertex_vector: Iterator over vertices incident to this 
                cell.

        """
        cdef vector[Delaunay_with_info_D[info_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef DelaunayD_vertex_vector out = DelaunayD_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_faces(self, int i):
        r"""Find faces that are incident to this cell.

        Args:
            i (int): Dimensionality of the faces that should be returned.

        Returns:
            DelaunayD_face_vector: Iterator over faces incident to this cell. 

        """
        cdef vector[Delaunay_with_info_D[info_t].Face] it
        it = self.T.incident_faces(self.x, i)
        cdef DelaunayD_face_vector out = DelaunayD_face_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this cell.

        Returns:
            DelaunayD_cell_vector: Iterator over cells incident to this cell.

        """
        cdef vector[Delaunay_with_info_D[info_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef DelaunayD_cell_vector out = DelaunayD_cell_vector()
        out.assign(self.T, it)
        return out


cdef class DelaunayD_cell_iter:
    r"""Wrapper class for a triangulation cell iteration.

    Args:
        T (DelaunayD): Triangulation that this cell belongs to. 
        cell (:obj:`str`, optional): String specifying the cell that
            should be referenced. Valid options include: 
                'all_begin': The first cell in an iteration over all cells. 
                'all_end': The last cell in an iteration over all cells.
    
    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ Triangulation object 
            that this cell belongs to. 
        x (:obj:`Delaunay_with_info_D[info_t].All_cells_iter`): C++ cell
            object. Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef Delaunay_with_info_D[info_t].All_cells_iter x

    def __cinit__(self, DelaunayD T, str cell = None):
        self.T = T.T
        if cell == 'all_begin':
            self.x = self.T.all_cells_begin()
        elif cell == 'all_end':
            self.x = self.T.all_cells_end()

    def __richcmp__(DelaunayD_cell_iter self, DelaunayD_cell_iter solf, int op):
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
        r"""DelaunayD_cell: Corresponding cell object."""
        def __get__(self):
            cdef DelaunayD_cell out = DelaunayD_cell()
            out.T = self.T
            out.x = Delaunay_with_info_D[info_t].Cell(self.x)
            return out


cdef class DelaunayD_cell_range:
    r"""Wrapper class for iterating over a range of triangulation cells.

    Args:
        xstart (DelaunayD_cell_iter): The starting cell. 
        xstop (DelaunayD_cell_iter): Final cell that will end the iteration. 
        finite (:obj:`bool`, optional): If True, only finite cells are 
            iterated over. Otherwise, all cells are iterated over. Defaults
            to False.  

    Attributes:
        x (DelaunayD_cell_iter): The current cell. 
        xstop (DelaunayD_cell_iter): Final cell that will end the iteration. 
        finite (bool): If True, only finite cells are iterated over. Otherwise, 
            all cells are iterated over.   

    """
    cdef DelaunayD_cell_iter x
    cdef DelaunayD_cell_iter xstop
    cdef pybool finite
    def __cinit__(self, DelaunayD_cell_iter xstart, DelaunayD_cell_iter xstop,
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
        cdef DelaunayD_cell out
        if self.x != self.xstop:
            out = self.x.cell
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class DelaunayD_cell_vector:
    r"""Wrapper class for a vector of cells.

    Attributes:
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_D[info_t].Cell]`): Vector of C++ 
            cells.
        n (int): The number of cells in the vector.
        i (int): The index of the currect cell.

    """
    cdef Delaunay_with_info_D[info_t] *T
    cdef vector[Delaunay_with_info_D[info_t].Cell] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_D[info_t] *T,
                     vector[Delaunay_with_info_D[info_t].Cell] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object.
                Direct interaction with this object is not recommended.
            v (:obj:`vector[Delaunay_with_info_D[info_t].Cell]`): Vector of 
                C++ cells.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef DelaunayD_cell out
        if self.i < self.n:
            out = DelaunayD_cell()
            out.T = self.T
            out.x = self.v[self.i]
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef DelaunayD_cell out
        if isinstance(i, int):
            out = DelaunayD_cell()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("DelaunayD_cell_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class DelaunayD:
    r"""Wrapper class for a 3D Delaunay triangulation.

    Attributes:
        n (int): The number of points inserted into the triangulation.
        T (:obj:`Delaunay_with_info_D[info_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 
        n_per_insert (list of int): The number of points inserted at each
            insert.

    """

    cdef Delaunay_with_info_D[info_t] *T
    cdef readonly int n
    cdef public object n_per_insert
    cdef readonly pybool _locked
    cdef public object _cache_to_clear_on_update

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T = new Delaunay_with_info_D[info_t]()
        self.n = 0
        self.n_per_insert = []
        self._locked = False
        self._cache_to_clear_on_update = {}

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
                raise RuntimeError("Cannot get dependent property '{}' while triangulation is locked.".format(attr))
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

    def is_equivalent(DelaunayD self, DelaunayD solf):
        r"""Determine if two triangulations are equivalent. Currently only 
        checks that the triangulations have the same numbers of vertices, cells,
        edges, and facets.

        Args:
            solf (:class:`cgal4py.delaunay.DelaunayD`): Triangulation this one 
                should be compared to.

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
            :class:`cgal4py.delaunay.DelaunayD`: Triangulation constructed from
                saved information.

        """
        T = cls()
        T.read_from_file(fname)
        return T

    @classmethod
    def from_serial(cls, *args):
        r"""Create a triangulation from serialized information. 

        Args: 
            *args: All arguments are passed to :meth:`cgal4py.delaunay.DelaunayD.deserialize`. 

        Returns: 
            :class:`cgal4py.delaunay.DelaunayD`: Triangulation constructed from 
                deserialized information. 

        """
        T = cls()
        T.deserialize(*args)
        return T

    @classmethod
    def from_serial_buffer(cls, *args, **kwargs):
        r"""Create a triangulation from serialized information in a buffer.

        Args:
            See :meth:`cgal4py.delaunay.Delaunay3.deserialize_from_buffer`.

        Returns:
            :class:`cgal4py.delaunay.Delaunay3`: Triangulation constructed from
                deserialized information.

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
        cdef int ndim = D
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

    # def write_to_file(self, fname):
    #     r"""Write the serialized tessellation information to a file. 

    #     Args:
    #         fname (str): The full path to the file that the tessellation should 
    #             be written to.

    #     """
    #     cdef char* cfname = fname
    #     cdef bytes pyfname
    #     if PY_MAJOR_VERSION < 3:
    #         cfname = fname
    #     else:
    #         pyfname = bytes(fname, encoding="ascii")
    #         cfname = pyfname
    #     with nogil, cython.boundscheck(False), cython.wraparound(False):
    #         self.T.write_to_file(cfname)

    # @_update_to_tess
    # def read_from_file(self, fname):
    #     r"""Read serialized tessellation information from a file.

    #     Args:
    #         fname (str): The full path to the file that the tessellation should 
    #             be read from.

    #     """
    #     cdef char* cfname = fname
    #     cdef bytes pyfname
    #     if PY_MAJOR_VERSION < 3:
    #         cfname = fname
    #     else:
    #         pyfname = bytes(fname, encoding="ascii")
    #         cfname = pyfname
    #     with nogil, cython.boundscheck(False), cython.wraparound(False):
    #         self.T.read_from_file(cfname)
    #     self.n = self.T.num_finite_verts()
    #     self.n_per_insert.append(self.n)

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
                cells (np.ndarray of info_t): (n,m) Indices for m vertices in 
                    each of the n cells. A value of np.iinfo(np_info).max 
                    indicates the infinite vertex. 
                neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of 
                    the m neighbors to each of the n cells. 
                idx_inf (I): Value representing the infinite vertex and or 
                    a missing neighbor.

        """
        cdef info_t n, m, i
        cdef int32_t d, j
        cdef np.ndarray[np_info_t, ndim=2] cells
        cdef np.ndarray[np_info_t, ndim=2] neighbors
        # Initialize arrays based on properties
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(n == self.num_finite_verts)
        assert(m == self.num_cells)
        d = self.num_dims
        cells = np.zeros((m, d+1), np_info)
        neighbors = np.zeros((m, d+1), np_info)
        # Serialize and convert to original vertex order
        cdef info_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_idxinfo[info_t](
                n, m, d, &cells[0,0], &neighbors[0,0])
        # Sort if desired
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[info_t](&cells[0,0], &neighbors[0,0],
                                           m, d+1)
        return cells, neighbors, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_int32(self, info_t max_info,
                                  np.ndarray[np.int32_t] idx,
                                  pybool sort=False):
        cdef int32_t n, m
        cdef int32_t d
        cdef np.ndarray[np.int32_t, ndim=2] cells
        cdef np.ndarray[np.int32_t, ndim=2] neighbors
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = self.num_dims
        cells = np.zeros((m, d+1), 'int32')
        neighbors = np.zeros((m, d+1), 'int32')
        cdef int32_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[int32_t](
                n, m, d, &cells[0,0], &neighbors[0,0], max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[int32_t](&cells[0,0], &neighbors[0,0],
                                            m, d+1)
        return cells, neighbors, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_uint32(self, info_t max_info,
                                   np.ndarray[np.uint32_t] idx,
                                   pybool sort=False):
        cdef uint32_t n, m
        cdef int32_t d
        cdef np.ndarray[np.uint32_t, ndim=2] cells
        cdef np.ndarray[np.uint32_t, ndim=2] neighbors
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = self.num_dims
        cells = np.zeros((m, d+1), 'uint32')
        neighbors = np.zeros((m, d+1), 'uint32')
        cdef uint32_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[uint32_t](
                n, m, d, &cells[0,0], &neighbors[0,0], max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[uint32_t](&cells[0,0], &neighbors[0,0],
                                             m, d+1)
        return cells, neighbors, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_int64(self, info_t max_info,
                                  np.ndarray[np.int64_t] idx,
                                  pybool sort=False):
        cdef int64_t n, m
        cdef int32_t d
        cdef np.ndarray[np.int64_t, ndim=2] cells
        cdef np.ndarray[np.int64_t, ndim=2] neighbors
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = self.num_dims
        cells = np.zeros((m, d+1), 'int64')
        neighbors = np.zeros((m, d+1), 'int64')
        cdef int64_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[int64_t](
                n, m, d, &cells[0,0], &neighbors[0,0], max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[int64_t](&cells[0,0], &neighbors[0,0],
                                            m, d+1)
        return cells, neighbors, idx_inf

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _serialize_info2idx_uint64(self, info_t max_info,
                                   np.ndarray[np.uint64_t] idx,
                                   pybool sort=False):
        cdef uint64_t n, m
        cdef int32_t d
        cdef np.ndarray[np.uint64_t, ndim=2] cells
        cdef np.ndarray[np.uint64_t, ndim=2] neighbors
        n = self.T.num_finite_verts()
        m = self.T.num_cells()
        assert(idx.size >= n)
        d = self.num_dims
        cells = np.zeros((m, d+1), 'uint64')
        neighbors = np.zeros((m, d+1), 'uint64')
        cdef uint64_t idx_inf
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            idx_inf = self.T.serialize_info2idx[uint64_t](
                n, m, d, &cells[0,0], &neighbors[0,0], max_info, &idx[0])
        cells.resize(m, d+1, refcheck=False)
        neighbors.resize(m, d+1, refcheck=False)
        if sort:
            with nogil, cython.boundscheck(False), cython.wraparound(False):
                sortSerializedTess[uint64_t](&cells[0,0], &neighbors[0,0],
                                             m, d+1)
        return cells, neighbors, idx_inf

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
                cells (np.ndarray of info_t): (n,m) Indices for m vertices in 
                    each of the n cells. A value of np.iinfo(np_info).max 
                    indicates the infinite vertex. 
                neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of 
                    the m neighbors to each of the n cells. 
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
            raise TypeError("idx.dtype = {} is not supported.".format(
                idx.dtype))

    @_update_to_tess
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def deserialize(self, np.ndarray[np.float64_t, ndim=2] pos,
                    np.ndarray[np_info_t, ndim=2] cells,
                    np.ndarray[np_info_t, ndim=2] neighbors,
                    info_t idx_inf):
        r"""Deserialize triangulation. 

        Args: 
            pos (np.ndarray of float64): Coordinates of points. 
            cells (np.ndarray of info_t): (n,m) Indices for m vertices in each 
                of the n cells. A value of np.iinfo(np_info).max A value of 
                np.iinfo(np_info).max indicates the infinite vertex. 
            neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of the m 
                neighbors to each of the n cells. 
            idx_inf (info_t): Index indicating a vertex is infinite. 

        """
        cdef info_t n = pos.shape[0]
        cdef info_t m = cells.shape[0]
        cdef int32_t d = neighbors.shape[1]-1
        if (n == 0) or (m == 0):
            return
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.deserialize_idxinfo[info_t](n, m, d, &pos[0,0], 
                                               &cells[0,0], &neighbors[0,0],
                                               idx_inf)
        self.n = n
        self.n_per_insert.append(n)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @_update_to_tess
    def deserialize_with_info(self, np.ndarray[np.float64_t, ndim=2] pos,
                              np.ndarray[np_info_t, ndim=1] info,
                              np.ndarray[np_info_t, ndim=2] cells,
                              np.ndarray[np_info_t, ndim=2] neighbors,
                              info_t idx_inf):
        r"""Deserialize triangulation. 

        Args: 
            pos (np.ndarray of float64): Coordinates of points. 
            info (np.ndarray of info_t): Info for points. 
            cells (np.ndarray of info_t): (n,m) Indices for m vertices in each 
                of the n cells. A value of np.iinfo(np_info).max A value of 
                np.iinfo(np_info).max indicates the infinite vertex. 
            neighbors (np.ndarray of info_t): (n,l) Indices in `cells` of the m 
                neighbors to each of the n cells. 
            idx_inf (info_t): Index indicating a vertex is infinite. 

        """
        cdef info_t n = pos.shape[0]
        cdef info_t m = cells.shape[0]
        cdef int32_t d = neighbors.shape[1]-1
        if (n == 0) or (m == 0):
            return
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.deserialize[info_t](n, m, d, &pos[0,0], &info[0],
                                       &cells[0,0], &neighbors[0,0], idx_inf)
        self.n = n
        self.n_per_insert.append(n)

    @_dependent_property
    def num_dims(self):
        r"""int: The number of dimensions in the triangulation."""
        return self.T.num_dims()
    @_dependent_property
    def num_finite_verts(self): 
        r"""int: The number of finite vertices in the triangulation."""
        return self.T.num_finite_verts()
    @_dependent_property
    def num_finite_edges(self):
        r"""int: The number of finite edges in the triangulation."""
        return self.T.num_finite_faces(1)
    @_dependent_property
    def num_finite_facets(self):
        r"""int: The number of finite facets in the triangulation."""
        return self.T.num_finite_faces(D-1)
    @_dependent_property
    def num_finite_cells(self):
        r"""int: The number of finite cells in the triangulation."""
        return self.T.num_finite_cells()
    @_dependent_property
    def num_infinite_verts(self):
        r"""int: The number of infinite vertices in the triangulation."""
        return self.T.num_infinite_verts()
    @_dependent_property
    def num_infinite_cells(self):
        r"""int: The number of infinite cells in the triangulation."""
        return self.T.num_infinite_cells()
    @_dependent_property
    def num_infinite_edges(self):
        r"""int: The number of infinite edges in the triangulation."""
        return self.T.num_infinite_faces(1)
    @_dependent_property
    def num_infinite_facets(self):
        r"""int: The number of infinite facets in the triangulation."""
        return self.T.num_infinite_faces(D-1)
    @_dependent_property
    def num_verts(self): 
        r"""int: The total number of vertices (finite + infinite) in the 
        triangulation."""
        return self.T.num_verts()
    @_dependent_property
    def num_cells(self):
        r"""int: The total number of cells (finite + infinite) in the 
        triangulation."""
        return self.T.num_cells()
    @_dependent_property
    def num_edges(self):
        r"""int: The total number of edges (finite + infinite) in the 
        triangulation."""
        return self.T.num_faces(1)
    @_dependent_property
    def num_facets(self):
        r"""int: The total number of facets (finite + infinite) in the 
        triangulation."""
        return self.T.num_faces(D - 1)

    def num_finite_faces(self, int d):
        r"""The number of finite faces of dimension `d` in the triangulation.

        Args:
            d (int): Number of dimensions.

        Returns:
            int: The number of finite faces with dimensionality `d` in the 
                triangulation.

        """
        return self.T.num_finite_faces(d)

    def num_infinite_faces(self, int d):
        r"""The number of infinite faces of dimension `d` in the triangulation.

        Args:
            d (int): Number of dimensions.

        Returns:
            int: The number of infinite faces with dimensionality `d` in the 
                triangulation.

        """
        return self.T.num_infinite_faces(d)

    def num_faces(self, int d):
        r"""The total number of faces of dimension `d` in the triangulation.

        Args:
            d (int): Number of dimensions.

        Returns:
            int: The number of faces with dimensionality `d` in the 
                triangulation.

        """
        return self.T.num_faces(d)

    @_dependent_property
    def infinite_vertex(self):
        r"""DelaunayD_vertex: The infinite vertex."""
        cdef Delaunay_with_info_D[info_t].Vertex x = self.T.infinite_vertex()
        cdef DelaunayD_vertex out = DelaunayD_vertex()
        out.assign(self.T, x)
        return out

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
        assert(m == D)
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
        out = np.zeros([self.n, self.num_dims], 'float64')
        if self.n == 0:
            return out
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.info_ordered_vertices(&out[0,0])
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
    def remove(self, DelaunayD_vertex x):
        r"""Remove a vertex from the triangulation. 

        Args: 
            x (DelaunayD_vertex): Vertex that should be removed. 

        """
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.remove(x.x)

    def get_vertex(self, np_info_t index):
        r"""Get the vertex object corresponding to the given index. 

        Args: 
            index (np_info_t): Index of vertex that should be found. 

        Returns: 
            DelaunayD_vertex: Vertex corresponding to the given index. If the 
                index is not found, the infinite vertex is returned. 

        """
        cdef Delaunay_with_info_D[info_t].Vertex v
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            v = self.T.get_vertex(index)
        cdef DelaunayD_vertex out = DelaunayD_vertex()
        out.assign(self.T, v)
        return out

    def locate(self, np.ndarray[np.float64_t, ndim=1] pos,
               DelaunayD_cell start = None):
        r"""Get the vertex/cell/facet/edge that a given point is a part of.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.   
            start (DelaunayD_cell, optional): Cell to start the search at. 

        Returns:

        """
        assert(len(pos) == self.num_dims)
        cdef int lt
        lt = 999
        cdef DelaunayD_face f = DelaunayD_face()
        cdef DelaunayD_facet ft = DelaunayD_facet()
        cdef DelaunayD_cell c = DelaunayD_cell()
        cdef int ind
        if start is not None:
            c.assign(self.T, self.T.locate(&pos[0], lt, f.x, ft.x, start.x))
        else:
            c.assign(self.T, self.T.locate(&pos[0], lt, f.x, ft.x))
        print(lt)
        assert(lt != 999)
        if lt == 0: # vertex
            return c.vertex(f.index(0))
        elif lt == 1: # face
            return f
        elif lt == 2: # facet
            ind = ft.ind
            ft = c.facet(ind)
            return ft
        elif lt == 3: # cell
            return c
        elif lt == 4:
            print("Point {} is outside the convex hull.".format(pos))
            return c
        elif lt == 5:
            print("Point {} is outside the affine hull.".format(pos))
            return 0
        else:
            raise RuntimeError("Value of {} not expected from CGAL locate.".format(lt))

    @property
    def all_verts_begin(self):
        r"""DelaunayD_vertex_iter: Starting vertex for all vertices in the 
        triangulation."""
        return DelaunayD_vertex_iter(self, 'all_begin')
    @property
    def all_verts_end(self):
        r"""DelaunayD_vertex_iter: Final vertex for all vertices in the 
        triangulation."""
        return DelaunayD_vertex_iter(self, 'all_end')
    @property
    def all_verts(self):
        r"""DelaunayD_vertex_range: Iterable for all vertices in the 
        triangulation."""
        return DelaunayD_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end)
    @property
    def finite_verts(self):
        r"""DelaunayD_vertex_range: Iterable for finite vertices in the 
        triangulation."""
        return DelaunayD_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end, finite = True)

    @property
    def all_cells_begin(self):
        r"""DelaunayD_cell_iter: Starting cell for all cells in the triangulation."""
        return DelaunayD_cell_iter(self, 'all_begin')
    @property
    def all_cells_end(self):
        r"""DelaunayD_cell_iter: Finall cell for all cells in the triangulation."""
        return DelaunayD_cell_iter(self, 'all_end')
    @property
    def all_cells(self):
        r"""DelaunayD_cell_range: Iterable for all cells in the
        triangulation."""
        return DelaunayD_cell_range(self.all_cells_begin,
                                    self.all_cells_end)
    @property
    def finite_cells(self):
        r"""DelaunayD_cell_range: Iterable for finite cells in the
        triangulation."""
        return DelaunayD_cell_range(self.all_cells_begin,
                                    self.all_cells_end, finite = True)

    def mirror_index(self, DelaunayD_cell x, int i):
        r"""Get the index of a cell with respect to its ith neighbor. 

        Args: 
            x (DelaunayD_cell): Cell to get mirrored index for. 
            i (int): Index of neighbor that should be used to determine the 
                mirrored index. 

        Returns: 
            int: Index of cell x with respect to its ith neighbor. 

        """
        return x.mirror_index(i)

    def mirror_vertex(self, DelaunayD_cell x, int i):
        r"""Get the vertex of a cell's ith neighbor opposite to the cell. 

        Args: 
            x (DelaunayD_cell): Cell. 
            i (int): Index of neighbor that should be used to determine the 
                mirrored vertex. 

        Returns:
            DelaunayD_vertex: Vertex in the ith neighboring cell of cell x, 
                that is opposite to cell x. 

        """
        return x.mirror_vertex(i)

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
        assert(left_edges.shape[1] == self.num_dims)
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
        cdef object out = [None for i in xrange(vout.size())]
        for i in xrange(vout.size()):
            out[i] = np.empty(vout[i].size(), np_info)
            for j in xrange(vout[i].size()):
                out[i][j] = vout[i][j]
        return out

