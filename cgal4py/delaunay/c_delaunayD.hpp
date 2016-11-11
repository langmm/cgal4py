#include <vector>
#include <set>
#include <array>
#include <utility>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdint.h>
#ifdef READTHEDOCS
#define VALID 1
#include "dummy_CGAL.hpp"
#else
#define VALID 1
#include <CGAL/config.h>
#if (CGAL_VERSION_NR >= 1040601000)
#include <CGAL/Epick_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/Unique_hash_map.h>
#include <CGAL/Linear_algebraHd.h>
#else
#include "dummy_CGAL.hpp"
#endif
#endif

typedef CGAL::Linear_algebraHd<double> LA;
typedef LA::Matrix Matrix;
typedef LA::Vector Vector;

int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

int D = 4; // REPLACE

template <typename Info_>
class Delaunay_with_info_D
{
public:
  typedef CGAL::Epick_d< CGAL::Dimension_tag<D> >      K;
  typedef CGAL::Delaunay_triangulation<K>              Delaunay;
  typedef typename Delaunay::Point                     Point;
  // typedef typename Delaunay::Vertex                    Vertex_T;
  typedef typename Delaunay::Facet                     Facet_handle;
  // typedef typename Delaunay::Full_cell                 Cell_T;
  typedef typename Delaunay::Face                      Face_handle;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Full_cell_handle          Cell_handle;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Facet_iterator            Facet_iterator;
  typedef typename Delaunay::Full_cell_iterator        Cell_iterator;
  typedef typename Delaunay::Finite_vertex_iterator    Finite_vertex_iterator;
  typedef typename Delaunay::Finite_facet_iterator     Finite_facet_iterator;
  typedef typename Delaunay::Finite_full_cell_iterator Finite_cell_iterator;
  typedef typename Delaunay::Locate_type               Locate_type;
  typedef typename K::Cartesian_const_iterator_d       Cartesian_const_iterator_d;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Cell_handle,int>    Cell_hash;
  typedef Info_ Info;
  Delaunay T = Delaunay(D);
  bool updated = false;
  Delaunay_with_info_D() {};
  Delaunay_with_info_D(double *pts, Info *val, uint32_t n) {
    insert(pts, val, n);
  }
  bool is_valid() const { return T.is_valid(); }
  uint32_t num_dims() const { return (uint32_t)(T.current_dimension()); }
  uint32_t num_finite_verts() const { return (uint32_t)(T.number_of_vertices()); }
  uint32_t num_finite_cells() const { return (uint32_t)(T.number_of_finite_full_cells()); }
  uint32_t num_infinite_verts() const { return 1; }
  uint32_t num_infinite_cells() const { return (num_cells() - num_finite_cells()); }
  uint32_t num_verts() const { return (num_finite_verts() + num_infinite_verts()); }
  uint32_t num_cells() const { return (uint32_t)(T.number_of_full_cells()); }
  bool is_equal(const Delaunay_with_info_D<D, Info> other) const {
    // Verts
    if (num_verts() != other.num_verts()) return false;
    if (num_finite_verts() != other.num_finite_verts()) return false;
    if (num_infinite_verts() != other.num_infinite_verts()) return false;
    // Cells
    if (num_cells() != other.num_cells()) return false;
    if (num_finite_cells() != other.num_finite_cells()) return false;
    if (num_infinite_cells() != other.num_infinite_cells()) return false;
    return true;
  }

  class Vertex;
  class Facet;
  class Cell;
  class Face;

  Point pos2point(double* pos) {
    std::vector<Info> vp;
    for (uint32_t i; i < num_dims(); i++)
      vp.push_back(pos[i]);
    return Point(vp.begin(), vp.end());
  }
  void insert(double *pts, Info *val, uint32_t n)
  {
    updated = true;
    uint32_t i, j;
    Vertex_handle v;
    std::vector<Info> vp;
    for (i = 0; i < n; i++) {
      v = T.insert(pos2point(pts+(num_dims()*i)));
      v->data() = val[i];
    }
  }
  void remove(Vertex v) { updated = true; T.remove(v._x); }
  void clear() { updated = true; T.clear(); }

  Vertex get_vertex(Info index) const {
    Finite_vertex_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      if (it->data() == index)
        return Vertex(static_cast<Vertex_handle>(it));
    }
    return Vertex(T.infinite_vertex());
  }

  Face locate(double* pos, int& lt, Face f, Facet ft) const {
    Point p = Point(pos[0], pos[1], pos[2]);
    Locate_type lt_out = Locate_type(0);
    Cell out = Cell(T.locate(p, lt_out, f._x, ft._x));
    lt = (int)lt_out;
    return out;
  }
  Face locate(double* pos, int& lt, Face f, Facet ft, Cell c) const {
    Point p = Point(pos[0], pos[1], pos[2]);
    Locate_type lt_out = Locate_type(0);
    Cell out = Cell(T.locate(p, lt_out, f._x, ft._x, c._x));
    lt = (int)lt_out;
    return out;
  }

  template <typename Wrap, typename Wrap_handle>
  class wrap_insert_iterator
  {
  protected:
    std::vector<Wrap>* container;

  public:
    explicit wrap_insert_iterator (std::vector<Wrap>& x) : container(&x) {}
    wrap_insert_iterator& operator= (Wrap_handle value) {
      Wrap value_wrap = Wrap();
      value_wrap._x = value;
      container->push_back(value_wrap);
      return *this;
    }
    wrap_insert_iterator& operator* ()
    { return *this; }
    wrap_insert_iterator& operator++ ()
    { return *this; }
    wrap_insert_iterator operator++ (int)
    { return *this; }
  };

  // Vertex construct
  class All_verts_iter {
  public:
    Vertex_iterator _x = Vertex_iterator();
    All_verts_iter() {
      _x = Vertex_iterator();
    }
    All_verts_iter(Vertex_iterator x) { _x = x; }
    All_verts_iter& operator*() { return *this; }
    All_verts_iter& operator++() {
      _x++;
      return *this;
    }
    All_verts_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_verts_iter other) { return (_x == other._x); }
    bool operator!=(All_verts_iter other) { return (_x != other._x); }
    Vertex vertex() { return Vertex((Vertex_handle)(_x)); }
  };
  All_verts_iter all_verts_begin() { return All_verts_iter(T.vertices_begin()); }
  All_verts_iter all_verts_end() { return All_verts_iter(T.vertices_end()); }

  class Vertex {
  public:
    Vertex_handle _x = Vertex_handle();
    Vertex() { _x = Vertex_handle(); }
    Vertex(Vertex_handle x) { _x = x; }
    Vertex(All_verts_iter x) { _x = static_cast<Vertex_handle>(x._x); }
    bool operator==(Vertex other) { return (_x == other._x); }
    bool operator!=(Vertex other) { return (_x != other._x); }
    void point(double* out) {
      Point p = _x->point();
      Cartesian_const_iterator_d it;
      int i = 0;
      for (it = _x->cartesian_begin(); it != _x->cartesian_end(); ++it) {
	out[i] = *it;
	i++;
      }
    }
    Info info() { return _x->data(); }
    Cell cell() { return Cell(_x->full_cell()); }
    void set_cell(Cell c) { _x->set_full_cell(c._x); }
    void set_point(double *x) {
      Point p = pos2point(x);
      _x->set_point(p);
    }
  };


  // Face construct
  class Face {
  public:
    Face_handle _x = Face_handle();
    Face() { _x = Face_handle(); }
    Face(Face_handle x) { _x = x; }
    bool operator==(Face other) { return (_x == other._x); }
    bool operator!=(Face other) { return (_x != other._x); }
    Vertex vertex(int i) { return Vertex(_x->vertex(i)); }
    int ind(int i) { return _x->index(i); }
    Cell cell() { return Cell(_x->full_cell()); }
    int dim() { return _x->face_dimension(); }
    void set_cell(Cell c) { _x->set_full_cell(c._x); }
    void set_index(int i, int j) { _x->set_index(i, j); }
    std::vector<Vertex> vertices() const {
      int i;
      std::vector<Vertex> out;
      for (i = 0; i <= dim(); i++)
	out.push_back(vertex(i));
      return out;
    }
  };


  // Facet construct
  class All_facets_iter {
  public:
    Facet_iterator _x = Facet_iterator();
    All_facets_iter() { _x = Facet_iterator(); }
    All_facets_iter(Facet_iterator x) { _x = x; }
    All_facets_iter& operator*() { return *this; }
    All_facets_iter& operator++() {
      _x++;
      return *this;
    }
    All_facets_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_facets_iter other) { return (_x == other._x); }
    bool operator!=(All_facets_iter other) { return (_x != other._x); }
  };
  All_facets_iter all_facets_begin() { return All_facets_iter(T.facets_begin()); }
  All_facets_iter all_facets_end() { return All_facets_iter(T.facets_end()); }

  class Facet {
  public:
    Facet_handle _x = Facet_handle();
    Facet() {}
    Facet(Facet_handle x) { _x = x; }
    Facet(Facet_iterator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(Finite_facet_iterator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(All_facets_iter x) { _x = static_cast<Facet_handle>(*(x._x)); }
    Facet(Cell x, int i1) { _x = Facet_handle(x._x, i1); }
    Cell cell() const { return Cell(_x.full_cell()); }
    int ind() const { return _x.index_of_covertex(); }
    Vertex vertex(int i) const {
      return Vertex(cell().vertex(ind() + 1 + i));
    }
    bool operator==(Facet other) const { return (_x == other._x); }
    bool operator!=(Facet other) const { return (_x != other._x); }
  };

  // Cell construct
  class All_cells_iter {
  public:
    Cell_iterator _x = Cell_iterator();
    All_cells_iter() {
      _x = Cell_iterator();
    }
    All_cells_iter(Cell_iterator x) { _x = x; }
    All_cells_iter& operator*() { return *this; }
    All_cells_iter& operator++() {
      _x++;
      return *this;
    }
    All_cells_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_cells_iter other) { return (_x == other._x); }
    bool operator!=(All_cells_iter other) { return (_x != other._x); }
  };
  All_cells_iter all_cells_begin() { return All_cells_iter(T.full_cells_begin()); }
  All_cells_iter all_cells_end() { return All_cells_iter(T.full_cells_end()); }

  class Cell {
  public:
    Cell_handle _x = Cell_handle();
    Cell() { _x = Cell_handle(); }
    Cell(Cell_handle x) { _x = x; }
    Cell(All_cells_iter x) { _x = static_cast<Cell_handle>(x._x); }
    bool operator==(Cell other) const { return (_x == other._x); }
    bool operator!=(Cell other) const { return (_x != other._x); }
  
    Vertex vertex(int i) const { return Vertex(_x->vertex(i)); }
    bool has_vertex(Vertex v) const { return _x->has_vertex(v._x); }
    bool has_vertex(Vertex v, int *i) const { return _x->has_vertex(v._x, *i); }
    int ind(Vertex v) const { return _x->index(v._x); }
    std::vector<Vertex> vertices() const {
      std::vector<Vertex> out;
      Vertex_iterator it;
      for (it = _x->vertices_begin(); it != _x->vertices_end(); ++it) {
	out.push_back(Vertex((Vertex_handle)(it)));
      }
      return out;
    }

    Cell neighbor(int i) const { return Cell(_x->neighbor(i)); }
    bool has_neighbor(Cell c) const { return _x->has_neighbor(c._x); }
    bool has_neighbor(Cell c, int *i) const { return _x->has_neighbor(c._x, *i); }
    int ind(Cell c) const { return _x->index(c._x); }
    std::vector<Cell> neighbors() const {
      std::vector<Cell> out;
      Vertex_iterator it;
      for (it = _x->verticess_begin(); it != _x->vertices_end(); ++it) {
	out.push_back(Cell((Cell_handle)(_x->neighbor(_x->index(it)))));
      }
      return out;
    }

    void set_vertex(int i, Vertex v) { _x->set_vertex(i, v._x); }
    void set_vertices() { _x->set_vertices(); }
    void set_neighbor(int i, Cell c) { _x->set_neighbor(i, c._x); }

  };

  bool are_equal(Face f1, Face f2) {
    if (f1.dim() != f2.dim())
      return false;
    int i1, i2;
    Vertex v1;
    for (i1 = 0; i1 < (f1.dim()+1); i1++) {
      bool match = false;
      v1 = f1.vertex(i1);
      for (i2 = 0; i2 < (f2.dim()); i2++) {
	if (v1 == f2.vertex(i2)) {
	  match = true;
	  break;
	}
      }
      if (!match)
	return false;
    }
    return true;
  }
  bool are_equal(Facet f1, Facet f2) {
    int i1, i2;
    Vertex v1;
    for (i1 = 0; i1 < (f1.dim()+1); i1++) {
      bool match = false;
      v1 = f1.vertex(i1);
      for (i2 = 0; i2 < (f2.dim()); i2++) {
	if (v1 == f2.vertex(i2)) {
	  match = true;
	  break;
	}
      }
      if (!match)
	return false;
    }
    return true;
  }

  // Testing incidence to the infinite vertex
  bool is_infinite(Vertex x) const { return T.is_infinite(x._x); }
  bool is_infinite(Face x) const { return T.is_infinite(x._x); }
  bool is_infinite(Facet x) const { return T.is_infinite(x._x); }
  bool is_infinite(Cell x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_verts_iter x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_facets_iter x) const {
    const Facet_iterator f = x._x;
    return T.is_infinite(*f);
  }
  bool is_infinite(All_cells_iter x) const { return T.is_infinite(x._x); }

  // Constructs incident
  std::vector<Vertex> incident_vertices(Vertex x) const {
    std::vector<Vertex> out;
    std::vector<Face> faces = incident_faces(x, 1);
    Vertex v;
    std::size_t i, j;
    for (i = 0; i < faces.size(); i++) {
      for (j = 0; j < 2; j++) {
	v = faces[i].vertex(j);
	if (v != x)
	  out.push_back(v);
      }
    }
    return out;
  }
  std::vector<Vertex> incident_vertices(Face x) const {
    std::vector<Vertex> out = x.vertices();
    return out;
  }
  std::vector<Vertex> incident_vertices(Cell x) const {
    std::vector<Vertex> out = x.vertices();
    return out;
  }

  std::vector<Face> incident_faces(Vertex x, int i) const {
    std::vector<Face> out;
    T.tds().incident_faces(x._x, i, wrap_insert_iterator<Face,Face_handle>(out));
    return out;
  }
  std::vector<Face> incident_faces(Face x, int i) const {
    std::set<Face> sout;
    std::vector<Vertex> verts = incident_vertices(x);
    std::vector<Face> faces;
    std::size_t j, k;
    for (j = 0; j < verts.size(); j++) {
      faces = incident_faces(verts[j], i);
      for (k = 0; k < faces.size(); k++) {
	if (faces[k] != x)
	  sout.insert(faces[k]);
      }
    }
    std::vector<Face> out;
    std::copy(sout.begin(),sout.end(),std::back_inserter(out));
    return out;
  }
  std::vector<Face> incident_faces(Cell x, int i) const {
    std::set<Face> sout;
    std::vector<Vertex> verts = incident_vertices(x);
    std::vector<Face> faces;
    std::size_t j, k;
    for (j = 0; j < verts.size(); j++) {
      faces = incident_faces(verts[j], i);
      for (k = 0; k < faces.size(); k++) {
	if (faces[k] != x)
	  sout.insert(faces[k]);
      }
    }
    std::vector<Face> out;
    std::copy(sout.begin(),sout.end(),std::back_inserter(out));
    return out;
  }

  std::vector<Cell> incident_cells(Vertex x) const {
    std::vector<Cell> out;
    T.incident_full_cells(x._x, wrap_insert_iterator<Cell,Cell_handle>(out));
    return out;
  }
  std::vector<Cell> incident_cells(Face x) const {
    std::vector<Cell> out;
    T.incident_full_cells(x._x, wrap_insert_iterator<Cell,Cell_handle>(out));
    return out;
  }
  std::vector<Cell> incident_cells(Cell x) const {
    std::vector<Cell> out = x.neighbors();
    return out;
  }

  int mirror_index(Cell x, int i) const { return T.mirror_index(x._x, i); }
  Vertex mirror_vertex(Cell x, int i) const { return Vertex(T.mirror_vertex(x._x, i)); }

  void circumcenter(Cell x, double* out) const {
    int i;
    if (T.is_infinite(x._x)) {
      for (i = 0; i < num_dims(); i++)
	out[i] = std::numeric_limits<double>::infinity();
    } else {
      Point p = x._x->circumcenter();
      for (i = 0; i < num_dims(); i++)
	out[i] = p[i];
    }
  }

  // TODO: fix these. need to transform face to remove 
  // extra dimension and create a true simplex
  double n_simplex_volume(Face f) const {
    int mat_dim = f.dim();
    int i,j;
    Matrix A(mat_dim,mat_dim);
    Point p0 = f.vertex(0)._x;
    Point p1;
    for (i = 0; i < mat_dim; i++) { // column
      for (j = 0; j < mat_dim; j++) { // row
	p1 = f.vertex(i+1)._x;
	A(i,j) = p1[j] - p0[j];
      }
    }
    double det = (double)(LA::determinant(A));
    return std::abs(det/((double)(factorial(mat_dim))));
  }
  double n_simplex_volume(Facet f) const {
    int mat_dim = f.dim();
    int i,j;
    Matrix A(mat_dim,mat_dim);
    Point p0 = f.vertex(0)._x;
    Point p1;
    for (i = 0; i < mat_dim; i++) { // column
      for (j = 0; j < mat_dim; j++) { // row
	p1 = f.vertex(i+1)._x;
	A(i,j) = p1[j] - p0[j];
      }
    }
    double det = (double)(LA::determinant(A));
    return std::abs(det/((double)(factorial(mat_dim))));
  }
  double n_simplex_volume(std::vector<Point> pts) const {
    int mat_dim = T.current_dimension();
    int i,j;
    Matrix A(mat_dim,mat_dim);
    for (i = 0; i < mat_dim; i++) { // column
      for (j = 0; j < mat_dim; j++) { // row
	A(i,j) = pts[i+1][j] - pts[0][j];
      }
    }
    double det = (double)(LA::determinant(A));
    return std::abs(det/((double)(factorial(mat_dim))));
  }
  
  double dual_volume(const Vertex v) const {
    std::vector<Face> edges = incident_faces(v, 1);
    std::vector<Cell> cells;
    Point orig = v._x->point();
    Point midp;
    double vol = 0.0;
    std::size_t i, j, k;
    std::vector<Point> centers;
    std::vector<Point> pts;
    for (i = 0; i < edges.size(); i++) {
      midp = K::midpoint(edges[i].vertex(0), edges[i].vertex(1));

      cells = incident_cells(edges[i]);

      centers.clear();
      for (j = 0; j < cells.size(); j++)
	centers.push_back(cells[j]._x->circumcenter());
      centers.push_back(centers[0]);
      
      for (k = 0; k < (centers.size() - T.tds().current_dimension() + 1); k++) {
	pts.clear();
	pts.push_back(orig);
	pts.push_back(midp);
	for (j = 0; j < (T.tds().current_dimension() - 1); j++) {
	  pts.push_back(centers[k+j]);
	}
	vol += n_simplex_volume(pts);
      }
    }
    return vol;
  }
  void dual_volumes(double *vols) const {
    for (Finite_vertex_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      vols[it->data()] = dual_volume(Vertex(it));
    }
  }

  void write_to_file(const char* filename) const
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      os << T;
    }
  }

  void read_from_file(const char* filename)
  {
    std::ifstream is(filename, std::ios::binary);
    if (!is) std::cerr << "Error cannot open file: " << filename << std::endl;
    else {
      is >> T;
    }
  }

  template <typename I>
  I serialize(I &n, I &m, int32_t &d,
              double* vert_pos, Info* vert_info,
              I* cells, I* neighbors) const
  {
    I idx_inf = std::numeric_limits<I>::max();

    // Header
    n = static_cast<int>(T.number_of_vertices());
    m = static_cast<int>(T.number_of_full_cells());
    d = static_cast<int>(T.current_dimension());
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Vertex_hash V;
    Cell_hash C;

    // first (infinite) vertex
    Vertex_handle vit;
    int inum = 0, i;
    Vertex_handle v = T.infinite_vertex();
    V[v] = -1;

    // other vertices
    for( Vertex_iterator vit = T.tds().vertices_begin(); vit != T.tds().vertices_end() ; ++vit) {
      if ( v != vit ) {
	for (i = 0; i < d; i++) {
	  vert_pos[d*inum + i] = (double)(vit->point()[i]);
	}
	vert_info[inum] = vit->data();
        V[vit] = inum++;
      }
    }

    // vertices of the cells
    inum = 0;
    for( Cell_iterator ib = T.full_cells_begin();
	 ib != T.full_cells_end(); ++ib) {
      for (int j = 0; j < dim ; ++j) {
	vit = ib->vertex(j);
	if ( v == vit )
	  cells[dim*inum + j] = idx_inf;
	else
	  cells[dim*inum + j] = V[vit];
      }
      C[ib] = inum++;
    }

    // neighbor pointers of the cells
    inum = 0;
    for( Cell_iterator it = T.full_cells_begin();
	 it != T.full_cells_end(); ++it) {
      for (int j = 0; j < d+1; ++j){
	neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
      }
      inum++;
    }
    return idx_inf;
  }


  template <typename I>
  Info serialize_idxinfo(I &n, I &m, int32_t &d,
                         Info* cells, I* neighbors) const
  {
    Info idx_inf = std::numeric_limits<Info>::max();

    // Header
    n = static_cast<int>(T.number_of_vertices());
    m = static_cast<int>(T.number_of_full_cells());
    d = static_cast<int>(T.current_dimension());
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Cell_hash C;

    // first (infinite) vertex
    Vertex_handle vit;
    int inum = 0;
    Vertex_handle v = T.infinite_vertex();

    // vertices of the cells
    inum = 0;
    for( Cell_iterator ib = T.full_cells_begin();
         ib != T.full_cells_end(); ++ib) {
      for (int j = 0; j < dim ; ++j) {
        vit = ib->vertex(j);
        if ( v == vit )
          cells[dim*inum + j] = idx_inf;
        else
          cells[dim*inum + j] = vit->data();
      }
      C[ib] = inum++;
    }

    // neighbor pointers of the cells
    inum = 0;
    for( Cell_iterator it = T.full_cells_begin();
         it != T.full_cells_end(); ++it) {
      for (int j = 0; j < d+1; ++j){
        neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
      }
      inum++;
    }
    return idx_inf;
  }

  template <typename I>
  I serialize_info2idx(I &n, I &m, int32_t &d,
                       I* cells, I* neighbors,
                       Info max_info, I* idx) const
  {
    I idx_inf = std::numeric_limits<I>::max();

    // Header
    n = static_cast<I>(max_info);
    d = static_cast<int>(T.current_dimension());
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Cell_hash C;

    // first (infinite) vertex
    Vertex_handle vit;
    Vertex_handle v = T.infinite_vertex();

    // vertices of the cells
    int j;
    bool *include_cell = (bool*)malloc(m*sizeof(bool));
    I inum = 0, inum_tot = 0;
    for( Cell_iterator ib = T.full_cells_begin();
         ib != T.full_cells_end(); ++ib) {
      include_cell[inum_tot] = false;
      for (j = 0; j < dim ; ++j) {
        vit = ib->vertex(j);
        // if ((v != vit) and (vit->data() < max_info)) {
        //   include_cell[inum_tot] = true;
        //   break;
        // }
        if ( v == vit) {
          include_cell[inum_tot] = false;
          break;
        } else if (vit->data() < max_info) {
          include_cell[inum_tot] = true;
        }
      }
      if (include_cell[inum_tot]) {
        for (j = 0; j < dim ; ++j) {
          vit = ib->vertex(j);
          if ( v == vit )
            cells[dim*inum + j] = idx_inf;
          else
            cells[dim*inum + j] = idx[vit->data()];
        }
        C[ib] = inum++;
      } else {
        C[ib] = idx_inf;
      }
      inum_tot++;
    }
    m = inum;

    // neighbor pointers of the cells
    inum = 0, inum_tot = 0;
    for( Cell_iterator it = T.full_cells_begin();
         it != T.full_cells_end(); ++it) {
      if (include_cell[inum_tot]) {
        for (int j = 0; j < d+1; ++j){
          neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
        }
        inum++;
      }
      inum_tot++;
    }

    free(include_cell);
    return idx_inf;
  }

  template <typename I>
  void deserialize(I n, I m, int32_t d,
                   double* vert_pos, Info* vert_info,
                   I* cells, I* neighbors, I idx_inf)
  {
    updated = true;

    if (T.number_of_vertices() != 0)
      T.clear();

    if (n==0) {
      return;
    }

    T.tds().set_current_dimension(d);

    Cell_iterator to_delete = T.full_cells_begin();

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);

    // infinite vertex
    V[n] = T.infinite_vertex();

    // read vertices
    I i;
    std::vector<double> vp;
    for(i = 0; i < n; ++i) {
      vp.clear();
      for (int j; j < d; j++)
	vp.push_back(vert_pos[d*i + j]);
      V[i] = T.tds().new_vertex();
      V[i]->point() = Point(vp.begin(), vp.end());
      V[i]->data() = vert_info[i];
    }

    // Creation of the cells
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    for(i = 0; i < m; ++i) {
      C[i] = T.tds().new_full_cell() ;
      for(int j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
        if (index == idx_inf)
          v = V[n];
        else
          v = V[index];
        C[i]->set_vertex(j, v);
        v->set_full_cell(C[i]);
      }
    }

    // Setting the neighbor pointers
    for(i = 0; i < m; ++i) {
      for(int j = 0; j < d+1; ++j){
        index = neighbors[(d+1)*i + j];
        C[i]->set_neighbor(j, C[index]);
      }
    }

    // delete flat cell
    T.tds().delete_full_cell(to_delete);

  }

  template <typename I>
  void deserialize_idxinfo(I n, I m, int32_t d, double* vert_pos,
                           I* cells, I* neighbors, I idx_inf)
  {
    updated = true;

    if (T.number_of_vertices() != 0)
      T.clear();

    if (n==0) {
      return;
    }

    T.tds().set_current_dimension(d);

    Cell_iterator to_delete = T.tds().full_cells_begin();

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);

    // infinite vertex
    V[n] = T.infinite_vertex();

    // read vertices
    I i;
    std::vector<double> vp(d);
    for(i = 0; i < n; ++i) {
      for (int j = 0; j < d; j++) 
	vp[j] = vert_pos[d*i + j];
      V[i] = T.tds().new_vertex();
      V[i]->point() = Point(vp.begin(), vp.end());
      V[i]->data() = (Info)(i);
    }

    // Creation of the cells
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    for(i = 0; i < m; ++i) {
      C[i] = T.tds().new_cell() ;
      for(int j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
        if (index == idx_inf)
          v = V[n];
        else
          v = V[index];
        C[i]->set_vertex(j, v);
        v->set_full_cell(C[i]);
      }
    }

    // Setting the neighbor pointers
    for(i = 0; i < m; ++i) {
      for(int j = 0; j < d+1; ++j){
        index = neighbors[(d+1)*i + j];
        C[i]->set_neighbor(j, C[index]);
      }
    }

    // delete flat cell
    T.tds().delete_full_cell(to_delete);

  }

  void info_ordered_vertices(double* pos) const {
    Info i;
    Point p;
    int j;
    int d = T.current_dimension();
    for (Finite_vertex_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      i = it->data();
      p = it->point();
      for (j = 0; j < d; j++)
	pos[d*i + j] = p[j];
    }
  }

  void vertex_info(Info* verts) const {
    int i = 0;
    for (Finite_vertex_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      verts[i] = it->data();
      i++;
    }
  }

  bool intersect_sph_box(Point *c, double r, double *le, double *re) const {
    for (int i = 0; i < T.current_dimension(); i++) {
      if (c[i] < le[i]) {
	if ((c[i] + r) < le[i])
	  return false;
      } else if (c[i] > re[i]) {
	if ((c[i] - r) > re[i])
	  return false;
      }
    }
    return true;
  }

  std::vector<std::vector<Info>> outgoing_points(uint64_t nbox,
                                                 double *left_edges,
                                                 double *right_edges) const {
    std::vector<std::vector<Info>> out;
    uint64_t b;
    for (b = 0; b < nbox; b++)
      out.push_back(std::vector<Info>());

    Vertex_handle v;
    Point cc, p1;
    double cr;
    int i, iinf = 0;
    int d = T.current_dimension();

    for (Cell_iterator it = T.full_cells_begin(); it != T.full_cells_end(); it++) {
      if (T.is_infinite(it) == true) {
        // Find index of infinite vertex
        for (i = 0; i < (d+1); i++) {
          v = it->vertex(i);
          if (T.is_infinite(v)) {
            iinf = i;
            break;
          }
        }
        for (b = 0; b < nbox; b++)
          for (i = 1; i < (d+1); i++) 
	    out[b].push_back((it->vertex((iinf+i) % (d+1)))->data());
      } else {
        p1 = it->vertex(0)->point();
        cc = it->circumcenter();
        cr = std::sqrt(static_cast<double>(K::squared_distance(p1, cc)));
        for (b = 0; b < nbox; b++) {
          if (intersect_sph_box(&cc, cr, left_edges + d*b, right_edges + d*b))
            for (i = 0; i < (d+1); i++) 
	      out[b].push_back((it->vertex(i))->data());
        }
      }
    }
    for (b = 0; b < nbox; b++) {
      std::sort( out[b].begin(), out[b].end() );
      out[b].erase( std::unique( out[b].begin(), out[b].end() ), out[b].end() );
    }

    return out;
  }
  
};


