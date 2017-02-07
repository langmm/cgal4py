cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

cdef extern from "c_delaunay3.hpp":
    cdef int VALID

    cdef cppclass Delaunay_with_info_3[Info] nogil:
        Delaunay_with_info_3() except +
        Delaunay_with_info_3(double *pts, Info *val, uint32_t n) except +
        bool updated
        bool is_valid() const
        uint32_t num_finite_verts() const
        uint32_t num_finite_edges() const
        uint32_t num_finite_facets() const
        uint32_t num_finite_cells() const
        uint32_t num_infinite_verts() const
        uint32_t num_infinite_edges() const
        uint32_t num_infinite_facets() const
        uint32_t num_infinite_cells() const
        uint32_t num_verts() const
        uint32_t num_edges() const
        uint32_t num_facets() const
        uint32_t num_cells() const

        bool is_equal(const Delaunay_with_info_3[Info] other) const

        cppclass Vertex
        cppclass Edge
        cppclass Facet
        cppclass Cell

        Vertex infinite_vertex() const

        void insert(double *, Info *val, uint32_t n) except +
        void remove(Vertex) except +
        void clear() except + 
        Vertex move(Vertex v, double *pos) except + 
        Vertex move_if_no_collision(Vertex v, double *pos) except +

        void write_to_file(const char* filename) except +
        void read_from_file(const char* filename) except +
        I serialize[I](I &n, I &m, int32_t &d,
                       double* vert_pos, Info* vert_info,
                       I* cells, I* neighbors) const
        Info serialize_idxinfo[I](I &n, I &m, int32_t &d,
                                  Info* cells, I* neighbors) const
        I serialize_info2idx[I](I &n, I &m, int32_t &d,
                                I* faces, I* neighbors,
                                Info max_info, I* idx) const
        void deserialize[I](I n, I m, int32_t d,
                         double* vert_pos, Info* vert_info,
                         I* cells, I* neighbors, I idx_inf)
        void deserialize_idxinfo[I](I n, I m, int32_t d, double* vert_pos,
                                    I* cells, I* neighbors, I idx_inf)

        Vertex get_vertex(Info index) except +
        Cell locate(double* pos, int& lt, int& li, int& lj)
        Cell locate(double* pos, int& lt, int& li, int& lj, Cell c)

        void info_ordered_vertices(double* pos)
        void vertex_info(Info* verts)
        void edge_info(Info* edges)

        cppclass All_verts_iter:
            All_verts_iter()
            All_verts_iter& operator++()
            All_verts_iter& operator--()
            bool operator==(All_verts_iter other)
            bool operator!=(All_verts_iter other)
            Vertex vertex()
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
            Cell cell()
            void set_point(double* x)
            void set_cell(Cell c)

        cppclass All_edges_iter:
            All_edges_iter()
            All_edges_iter& operator++()
            All_edges_iter& operator--()
            bool operator==(All_edges_iter other)
            bool operator!=(All_edges_iter other)
        All_edges_iter all_edges_begin()
        All_edges_iter all_edges_end()

        cppclass Edge:
            Edge()
            Edge(All_edges_iter it)
            Edge(Cell x, int i1, int i2)
            bool operator==(Edge other)
            bool operator!=(Edge other)
            Cell cell()
            int ind1()
            int ind2()
            Vertex vertex(int i)
            Vertex v1()
            Vertex v2()

        cppclass All_facets_iter:
            All_facets_iter()
            All_facets_iter& operator++()
            All_facets_iter& operator--()
            bool operator==(All_facets_iter other)
            bool operator!=(All_facets_iter other)
        All_facets_iter all_facets_begin()
        All_facets_iter all_facets_end()

        cppclass Facet:
            Facet()
            Facet(All_facets_iter it)
            Facet(Cell x, int i1)
            bool operator==(Facet other)
            bool operator!=(Facet other)
            Cell cell()
            int ind()
            Vertex vertex(int i)
            Edge edge(int i)

        cppclass All_cells_iter:
            All_cells_iter()
            All_cells_iter& operator++()
            All_cells_iter& operator--()
            bool operator==(All_cells_iter other)
            bool operator!=(All_cells_iter other)
        All_cells_iter all_cells_begin()
        All_cells_iter all_cells_end()

        cppclass Cell:
            Cell()
            Cell(All_cells_iter c)
            Cell(Vertex v1, Vertex v2, Vertex v3, Vertex v4)
            Cell(Vertex v1, Vertex v2, Vertex v3, Vertex v4,
                 Cell n1, Cell n2, Cell n3, Cell n4)
            bool operator==(Cell other)
            bool operator!=(Cell other)

            Facet facet(int i)

            Vertex vertex(int i)
            bool has_vertex(Vertex v)
            bool has_vertex(Vertex v, int *i)
            int ind(Vertex v)

            Cell neighbor(int i)
            bool has_neighbor(Cell c)
            bool has_neighbor(Cell c, int *i)
            int ind(Cell c)

            void set_vertex(int i, Vertex v)
            void set_vertices()
            void set_vertices(Vertex v1, Vertex v2, Vertex v3, Vertex v4)
            void set_neighbor(int i, Cell c)
            void set_neighbors()
            void set_neighbors(Cell c1, Cell c2, Cell c3, Cell c4)

            double min_angle()

        bool are_equal(const Edge e1, const Edge e2) 
        bool are_equal(const Facet f1, const Facet f2) 

        bool is_infinite(Vertex x)
        bool is_infinite(Edge x)
        bool is_infinite(Facet x)
        bool is_infinite(Cell x)
        bool is_infinite(All_verts_iter x)
        bool is_infinite(All_edges_iter x)
        bool is_infinite(All_facets_iter x)
        bool is_infinite(All_cells_iter x)

        bool is_edge(Vertex x1, Vertex x2, Cell& c, int& i, int& j) const
        bool is_facet(Vertex x1, Vertex x2, Vertex x3, Cell& c, int& i, int& j, int& k) const
        bool is_cell(Vertex x1, Vertex x2, Vertex x3, Vertex x4,
                     Cell& c, int& i1, int& i2, int& i3, int& i4) const

        vector[Vertex] incident_vertices(Vertex x)
        vector[Edge] incident_edges(Vertex x)
        vector[Facet] incident_facets(Vertex x)
        vector[Cell] incident_cells(Vertex x)

        vector[Vertex] incident_vertices(Edge x)
        vector[Edge] incident_edges(Edge x)
        vector[Facet] incident_facets(Edge x)
        vector[Cell] incident_cells(Edge x)

        vector[Vertex] incident_vertices(Facet x)
        vector[Edge] incident_edges(Facet x)
        vector[Facet] incident_facets(Facet x)
        vector[Cell] incident_cells(Facet x)

        vector[Vertex] incident_vertices(Cell x)
        vector[Edge] incident_edges(Cell x)
        vector[Facet] incident_facets(Cell x)
        vector[Cell] incident_cells(Cell x)

        Vertex nearest_vertex(double* pos)

        Facet mirror_facet(Facet x) const
        int mirror_index(Cell x, int i) const
        Vertex mirror_vertex(Cell x, int i) const

        void circumcenter(Cell x, double* out) const
        double dual_volume(const Vertex v) const
        void dual_volumes(double* vols) const
        double length(const Edge e) const

        bool is_boundary_cell(const Cell c) const
        double minimum_angle(const Cell c) const
        int minimum_angles(double* angles) const

        bool flip(Cell x, int i, int j)
        bool flip(Edge x)
        bool flip(Cell x, int i)
        bool flip(Facet x)
        void flip_flippable(Cell x, int i, int j)
        void flip_flippable(Edge x)
        void flip_flippable(Cell x, int i)
        void flip_flippable(Facet x)

        pair[vector[Cell],vector[Facet]] find_conflicts(double* pos, Cell start)

        int side_of_cell(const double* pos, Cell c, int& lt, int& li, int& lj) const
        int side_of_edge(const double* pos, const Edge e, int& lt, int& li) const
        int side_of_facet(const double* pos, const Facet f, int& lt, int& li, int& lj) const 
        # int side_of_circle(const Facet f, const double* pos)
        int side_of_sphere(const Cell c, const double* pos)
        bool is_Gabriel(const Edge e)
        bool is_Gabriel(const Facet f)

        vector[vector[Info]] outgoing_points(uint64_t nbox,
                                             double *left_edges, double *right_edges)
        void boundary_points(double *left_edge, double *right_edge, bool periodic,
                             vector[Info]& lx, vector[Info]& ly, vector[Info]& lz,
                             vector[Info]& rx, vector[Info]& ry, vector[Info]& rz,
                             vector[Info]& alln)
