# distutils: language = c++
# distutils: libraries = CGAL
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1

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
        uint32_t num_finite_verts()
        uint32_t num_finite_cells()
        uint32_t num_infinite_verts()
        uint32_t num_infinite_cells()
        uint32_t num_verts()
        uint32_t num_cells()

        cppclass Vertex
        cppclass Cell

        void insert(double *, Info *val, uint32_t n)
        void remove(Vertex v)
        Vertex move(Vertex v, double *pos)
        Vertex move_if_no_collision(Vertex v, double *pos)

        void write_to_file(const char* filename) except +
        void read_from_file(const char* filename) except +

        Vertex get_vertex(Info index)

        void edge_info(vector[pair[Info,Info]]& edges)

        cppclass All_verts_iter:
            All_verts_iter()
            All_verts_iter& operator++()
            All_verts_iter& operator--()
            bool operator==(All_verts_iter other)
            bool operator!=(All_verts_iter other)
        All_verts_iter all_verts_begin()
        All_verts_iter all_verts_end()

        cppclass Vertex:
            Vertex()
            Vertex(All_verts_iter v)
            bool operator==(Vertex other)
            bool operator!=(Vertex other)
            void point(double* out)
            vector[double] point()
            Info info()

        cppclass All_cells_iter:
            All_cells_iter()
            All_cells_iter& operator++()
            All_cells_iter& operator--()
            bool operator==(All_cells_iter other)
            bool operator!=(All_cells_iter other)
        All_cells_iter all_cells_begin()
        All_cells_iter all_cells_end()

        cppclass Cell_circ:
            Cell_circ()
            Cell_circ& operator++()
            Cell_circ& operator--()
            bool operator==(Cell_circ other)
            bool operator!=(Cell_circ other)
            bool is_done()

        cppclass Cell:
            Cell()
            Cell(All_cells_iter c)
            Cell(Cell_circ c)
            bool operator==(Cell other)
            bool operator!=(Cell other)

        bool is_infinite(Vertex x)
        bool is_infinite(Cell x)
        bool is_infinite(All_verts_iter x)
        bool is_infinite(All_cells_iter x)

        vector[Cell] incident_cells(Vertex x)
        vector[Vertex] incident_vertices(Vertex x)

        Vertex nearest_vertex(double* pos)
        void circumcenter(Cell x, double* out)
        double dual_area(const Vertex v)


cdef class Delaunay2:
    cdef int n
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef void _insert(self, np.ndarray[double, ndim=2, mode="c"] pts)
    cdef object _edge_info(self, int max_incl, np.uint64_t[:])

