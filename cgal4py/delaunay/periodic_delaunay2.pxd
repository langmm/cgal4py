cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

cdef extern from "c_periodic_delaunay2.hpp":
    cdef int VALID

    cdef cppclass PeriodicDelaunay_with_info_2[Info] nogil:
        PeriodicDelaunay_with_info_2() except +
        PeriodicDelaunay_with_info_2(const double *domain) except +
        PeriodicDelaunay_with_info_2(double *pts, Info *val, uint32_t n) except +
        PeriodicDelaunay_with_info_2(double *pts, Info *val, uint32_t n,
                                     const double *domain) except +
        bool updated
        bool is_valid() const
        void num_sheets(int32_t *ns_out) const
        uint32_t num_sheets_total() const
        uint32_t num_finite_verts() const
        uint32_t num_finite_edges() const
        uint32_t num_finite_cells() const
        uint32_t num_infinite_verts() const
        uint32_t num_infinite_edges() const
        uint32_t num_infinite_cells() const
        uint32_t num_verts() const
        uint32_t num_edges() const
        uint32_t num_cells() const
        uint32_t num_stored_verts() const
        uint32_t num_stored_edges() const
        uint32_t num_stored_cells() const

        bool is_equal(const PeriodicDelaunay_with_info_2[Info] other) const

        cppclass Vertex
        cppclass Edge
        cppclass Cell

        bool has_offset(Vertex v) const
        bool has_offset(Edge e) const
        bool has_offset(Cell c) const
        void point(Vertex v, double* pos) const
        void periodic_point(Vertex v, double* pos) const
        void periodic_offset(Vertex v, int32_t* off) const
        void point(Edge e, int i, double* pos) const
        void periodic_point(Edge e, int i, double* pos) const
        void periodic_offset(Edge e, int i, int32_t* off) const
        void point(Cell c, int i, double* pos) const
        void periodic_point(Cell c, int i, double* pos) const
        void periodic_offset(Cell c, int i, int32_t* off) const

        void set_domain(const double *domain) except +
        void insert(double *, Info *val, uint32_t n) except +
        void remove(Vertex v) except +
        void clear() except +
        Vertex move(Vertex v, double *pos) except +
        Vertex move_if_no_collision(Vertex v, double *pos) except +

        void write_to_file(const char* filename) except +
        void read_from_file(const char* filename) except +
        I serialize[I](I &n, I &m, int32_t &d,
                       double* domain, int32_t* cover,
                       double* vert_pos, Info* vert_info,
                       I* faces, I* neighbors, int32_t *offsets) const
        Info serialize_idxinfo[I](I &n, I &m, int32_t &d, 
                                  double* domain, int32_t* cover,
                                  Info* faces, I* neighbors, int32_t *offsets) const
        I serialize_info2idx[I](I &n, I &m, int32_t &d,
                                double* domain, int32_t* cover,
                                I* faces, I* neighbors, int32_t *offsets,
                                Info max_info, I* idx) const
        void deserialize[I](I n, I m, int32_t d,
                            double* domain, int32_t *cover,
                            double* vert_pos, Info* vert_info,
                            I* faces, I* neighbors, int32_t* offsets, I idx_inf) 
        void deserialize_idxinfo[I](I n, I m, int32_t d, 
                                    double* domain, int32_t *cover,
                                    double* vert_pos,
                                    I* faces, I* neighbors, int32_t* offsets, I idx_inf)

        Vertex get_vertex(Info index) except +
        Cell locate(double* pos, int& lt, int& li)
        Cell locate(double* pos, int& lt, int& li, Cell c)

        void info_ordered_vertices(double* pos)
        void vertex_info(Info* verts)
        void edge_info(Info* edges)

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
            void offset(int32_t* out)
            Info info()
            Cell cell()
            void set_cell(Cell c)
            void set_point(double* x)

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
            Edge(Vertex x1, Vertex x2)
            Edge(Cell x, int i)
            Cell cell()
            int ind()
            Vertex v1()
            Vertex v2()
            Vertex vertex(int i)
            bool operator==(Edge other)
            bool operator!=(Edge other)

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
            Cell(Vertex v1, Vertex v2, Vertex v3)
            Cell(Vertex v1, Vertex v2, Vertex v3, Cell c1, Cell c2, Cell c3)
            bool operator==(Cell other)
            bool operator!=(Cell other)

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
            void set_vertices(Vertex v1, Vertex v2, Vertex v3)
            void set_neighbor(int i, Cell c)
            void set_neighbors()
            void set_neighbors(Cell c1, Cell c2, Cell c3)

            void reorient()
            void ccw_permute()
            void cw_permute()

            int dimension() except + 

        bool is_infinite(Vertex x)
        bool is_infinite(Edge x)
        bool is_infinite(Cell x)
        bool is_infinite(All_verts_iter x)
        bool is_infinite(All_edges_iter x)
        bool is_infinite(All_cells_iter x)

        bool is_edge(Vertex x1, Vertex x2)
        bool is_edge(Vertex x1, Vertex x2, Cell& c, int& i)
        bool is_cell(Vertex x1, Vertex x2, Vertex x3)
        bool is_cell(Vertex x1, Vertex x2, Vertex x3, Cell& c)

        vector[Vertex] incident_vertices(Vertex x)
        vector[Edge] incident_edges(Vertex x)
        vector[Cell] incident_cells(Vertex x)

        vector[Vertex] incident_vertices(Edge x)
        vector[Edge] incident_edges(Edge x)
        vector[Cell] incident_cells(Edge x)

        vector[Vertex] incident_vertices(Cell x)
        vector[Edge] incident_edges(Cell x)
        vector[Cell] incident_cells(Cell x)

        Vertex nearest_vertex(double* pos)

        int mirror_index(Cell x, int i) const 
        Vertex mirror_vertex(Cell x, int i) const 

        void circumcenter(Cell x, double* out)
        double dual_area(const Vertex v)
        void dual_areas(double* vols) const
        double length(const Edge e)

        bool flip(Cell x, int i)
        bool flip(Edge x)
        void flip_flippable(Cell x, int i)
        void flip_flippable(Edge x)

        vector[Edge] get_boundary_of_conflicts(double* pos, Cell start)
        vector[Cell] get_conflicts(double* pos, Cell start)
        pair[vector[Cell],vector[Edge]] get_conflicts_and_boundary(double* pos, Cell start)

        int oriented_side(Cell f, const double* pos) const
        int side_of_oriented_circle(Cell f, const double* pos) const

        vector[vector[Info]] outgoing_points(uint64_t nbox,
                                             double *left_edges, double *right_edges)
        void boundary_points(double *left_edge, double *right_edge, bool periodic,
                             vector[Info]& lx, vector[Info]& ly,
                             vector[Info]& rx, vector[Info]& ry,
                             vector[Info]& alln) const

