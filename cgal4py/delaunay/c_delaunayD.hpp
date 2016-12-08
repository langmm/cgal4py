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
#include <CGAL/Triangulation_data_structure.h>
#include <CGAL/Triangulation_full_cell.h>
#include <CGAL/Triangulation_vertex.h>
#include <CGAL/Unique_hash_map.h>
#include <CGAL/Linear_algebraHd.h>
#else
#include "dummy_CGAL.hpp"
#endif
#endif

typedef CGAL::Linear_algebraHd<double> LA;
typedef LA::Matrix Matrix;
typedef LA::Vector Vector;

const int D = 4; // REPLACE

int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

template <typename Info_>
class Delaunay_with_info_D
{
public:
  typedef CGAL::Epick_d< CGAL::Dimension_tag<D> >      K;
  typedef CGAL::Triangulation_vertex<K,Info_>          Vb;
  typedef CGAL::Triangulation_full_cell<K>             Cb;
  typedef CGAL::Triangulation_data_structure<CGAL::Dimension_tag<D>,Vb,Cb> Tds;
  typedef CGAL::Delaunay_triangulation<K, Tds>         Delaunay;
  typedef typename Delaunay::Geom_traits               Geom_traits;
  typedef typename Delaunay::Point                     Point;
  typedef typename Delaunay::Vertex                    DVertex;
  typedef typename Delaunay::Facet                     Facet_handle;
  // typedef typename Delaunay::Full_cell                 Cell_T;
  typedef typename Delaunay::Face                      Face_handle;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Full_cell_handle          Cell_handle;
  typedef typename Delaunay::Vertex_const_handle       Vertex_const_handle;
  typedef typename Delaunay::Full_cell_const_handle    Cell_const_handle;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Vertex_const_iterator     Vertex_const_iterator;
  typedef typename Delaunay::Facet_iterator            Facet_iterator;
  typedef typename Delaunay::Full_cell_iterator        Cell_iterator;
  typedef typename Delaunay::Full_cell_const_iterator  Cell_const_iterator;
  typedef typename Delaunay::Finite_vertex_iterator    Finite_vertex_iterator;
  typedef typename Delaunay::Finite_vertex_const_iterator    Finite_vertex_const_iterator;
  typedef typename Delaunay::Finite_facet_iterator     Finite_facet_iterator;
  typedef typename Delaunay::Finite_full_cell_iterator Finite_cell_iterator;
  typedef typename Delaunay::Locate_type               Locate_type;
  typedef typename K::Cartesian_const_iterator_d       Cartesian_const_iterator_d;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Cell_const_iterator,int>    Cell_hash;
  typedef typename CGAL::Unique_hash_map<Vertex_const_handle,int>  Vertex_const_hash;
  typedef typename CGAL::Unique_hash_map<Cell_const_iterator,int>    Cell_const_hash;
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
  uint32_t num_finite_faces(int d) {
    uint32_t out = 0;
    typedef std::vector<Face_handle> Faces;
    Faces faces;
    std::back_insert_iterator< Faces > face_out(faces);
    typename Faces::iterator fit;
    Finite_vertex_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); ++it) {
      faces.clear();
      T.tds().incident_faces(it.base(), d, face_out);
      fit = faces.begin();
      for ( ; fit != faces.end(); ++fit) {
	if (!(T.is_infinite(*fit)))
	  out++;
      }
    }
    return out/(d + 1);
  }
  uint32_t num_infinite_verts() const { return 1; }
  uint32_t num_infinite_cells() const { return (num_cells() - num_finite_cells()); }
  uint32_t num_infinite_faces(int d) {
    Vertex_handle vh = T.infinite_vertex();
    typedef std::vector<Face_handle> Faces;
    Faces faces;
    std::back_insert_iterator< Faces > face_out(faces);
    T.tds().incident_faces(vh, d, face_out);
    return (uint32_t)(faces.size());
  }
  uint32_t num_verts() const { return (num_finite_verts() + num_infinite_verts()); }
  uint32_t num_cells() const { return (uint32_t)(T.number_of_full_cells()); }
  uint32_t num_faces(int d) { return (num_finite_faces(d) + num_infinite_faces(d)); }
  bool is_equal(const Delaunay_with_info_D<Info> other) const {
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

  Vertex infinite_vertex() const {
    return Vertex(T.infinite_vertex());
  }

  Vertex_handle non_const_handle(Vertex_const_handle vh) const {
    DVertex* p_v = (DVertex*)&(*vh);
    return (Vertex_handle(p_v));
  }

  Point pos2point(double* pos) const {
    std::vector<double> vp;
    for (uint32_t i = 0; i < D; i++)
      vp.push_back(pos[i]);
    return Point(vp.begin(), vp.end());
  }

  void insert(double *pts, Info *val, uint32_t n)
  {
    updated = true;
    uint32_t i;
    Vertex_handle v;
    for (i = 0; i < n; i++) {
      v = T.insert(pos2point(pts+(D*i)));
      v->data() = val[i];
    }
    v = T.infinite_vertex();
    v->data() = std::numeric_limits<Info>::max();
  }
  void remove(Vertex v) { updated = true; T.remove(v._x); }
  void clear() { updated = true; T.clear(); }

  Vertex get_vertex(Info index) {
    // Finite_vertex_const_iterator it = T.finite_vertices_begin();
    Finite_vertex_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      if (it->data() == index)
	return Vertex(it.base());
	// return Vertex(non_const_handle(it.base()));
    }
    return Vertex(T.infinite_vertex());
  }

  Cell locate(double* pos, int& lt, Face &f, Facet &ft) const {
    Point p = pos2point(pos);
    Locate_type lt_out = Locate_type(0);
    Cell out = Cell(T.locate(p, lt_out, f._x, ft._x));
    lt = (int)lt_out;
    return out;
  }
  Cell locate(double* pos, int& lt, Face &f, Facet &ft, Cell c) const {
    Point p = pos2point(pos);
    Locate_type lt_out = Locate_type(0);
    Facet_handle fth;
    Cell out = Cell(T.locate(p, lt_out, f._x, fth, c._x));
    ft = Facet(Facet_handle(fth.full_cell(), fth.index_of_covertex()));
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
    bool operator< (const Vertex other) const {
      return (_x->data() < other._x->data());
    }
    void point(double* out) {
      Point p = _x->point();
      Cartesian_const_iterator_d it;
      int i = 0;
      for (it = p.cartesian_begin(); it != p.cartesian_end(); ++it) {
      	out[i] = (double)(*it);
      	i++;
      }
    }
    Info info() { return _x->data(); }
    Cell cell() { return Cell(_x->full_cell()); }
    void set_cell(Cell c) { _x->set_full_cell(c._x); }
    void set_point(double *x) {
      std::vector<double> vp;
      Cartesian_const_iterator_d it;
      Point p = _x->point();
      int i = 0;
      for (it = p.cartesian_begin(); it != p.cartesian_end(); ++it) {
  	vp.push_back(x[i]);
  	i++;
      }
      p = Point(vp.begin(), vp.end());
      _x->set_point(p);
    }
  };


  // Face construct
  class Face {
  public:
    Face_handle _x = Face_handle(D);
    Face() { _x = Face_handle(D); }
    Face(Face_handle x) { _x = x; }
    Face(Cell x) { _x = Face_handle(x._x); }
    Vertex vertex(int i) const { return Vertex(_x.vertex(i % (dim()+1))); }
    int ind(int i) const { return _x.index(i); }
    Cell cell() const { return Cell(_x.full_cell()); }
    int dim() const { return _x.face_dimension(); }
    void set_cell(Cell c) { _x.set_full_cell(c._x); }
    void set_index(int i, int j) { _x.set_index(i, j); }
    std::vector<Vertex> vertices() const {
      int i;
      std::vector<Vertex> out;
      for (i = 0; i <= dim(); i++)
  	out.push_back(vertex(i));
      return out;
    }
    std::vector<Info> unique_vect() const {
      std::vector<Info> out;
      Vertex v;
      for (int i = 0; i <= dim(); i ++) {
	v = vertex(i);
	out.push_back(v.info());
      }
      std::sort(out.begin(), out.end());
      return out;
    }
    bool operator==(const Face& other) const {
      return (unique_vect() == other.unique_vect());
    }
    bool operator!=(const Face& other) const { return !(*this == other); }
    bool operator< (const Face& other) const {
      std::vector<Info> v1 = unique_vect();
      std::vector<Info> v2 = other.unique_vect();
      return std::lexicographical_compare(v1.begin(), v1.end(),
					  v2.begin(), v2.end());
    }
    bool operator> (const Face& other) const { return other < *this; }
    bool operator<=(const Face& other) const { return !(*this > other); }
    bool operator>=(const Face& other) const { return !(*this < other); }
  };
  // bool operator< (const Face& f1, const Face& f2) {
  //   std::vector<Info> v1 = f1.unique_vect();
  //   std::vector<Info> v2 = f2.unique_vect();
  //   return std::lexicographical_compare(v1.begin(), v1.end(),
  // 					v2.begin(), v2.end());
  // }
  // bool operator> (const Face& f1, const Face& f2) { return f2 < f1; }
  // bool operator<=(const Face& f1, const Face& f2) { return !(f1 > f2); }
  // bool operator>=(const Face& f1, const Face& f2) { return !(f1 < f2); }


  // Facet construct
  class Facet {
  public:
    Facet_handle _x = Facet_handle();
    Facet() {}
    Facet(Facet_handle x) { _x = x; }
    Facet(Facet_iterator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(Cell x, int i1) { _x = Facet_handle(x._x, i1); }
    // bool operator==(Facet other) const { return (_x == other._x); }
    // bool operator!=(Facet other) const { return (_x != other._x); }
    Cell cell() const { return Cell(_x.full_cell()); }
    int ind() const { return _x.index_of_covertex(); }
    Vertex vertex(int i) const {
      return Vertex(_x.full_cell()->vertex(i));//(ind() + 1 + i)%(D+1)));
    }
  };

  // Cell construct
  class All_cells_iter {
  public:
    Cell_iterator _x = Cell_iterator();
    All_cells_iter() {
      _x = Cell_iterator();
    }
    All_cells_iter(Cell_iterator x) { _x = x; }
    All_cells_iter(Cell_const_iterator x) { _x = x; }
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
  
    Vertex vertex(int i) const { return Vertex(_x->vertex(i%(D+1))); }
    bool has_vertex(Vertex v) const { return _x->has_vertex(v._x); }
    bool has_vertex(Vertex v, int *i) const { return _x->has_vertex(v._x, *i); }
    int ind(Vertex v) const { return _x->index(v._x); }
    std::vector<Vertex> vertices() const {
      std::vector<Vertex> out;
      for (int i = 0; i < (D+1); i++) {
	out.push_back(vertex(i));
      }
      // Vertex_iterator it;
      // for (it = _x->vertices_begin(); it != _x->vertices_end(); ++it) {
      // 	out.push_back(Vertex((Vertex_handle)(it)));
      // }
      return out;
    }

    Cell neighbor(int i) const { return Cell(_x->neighbor(i)); }
    bool has_neighbor(Cell c) const { return _x->has_neighbor(c._x); }
    bool has_neighbor(Cell c, int *i) const { return _x->has_neighbor(c._x, *i); }
    int ind(Cell c) const { return _x->index(c._x); }
    std::vector<Cell> neighbors() const {
      std::vector<Cell> out;
      for (int i = 0; i < (D+1); i++) {
	out.push_back(neighbor(i));
      }
      // Vertex_iterator it;
      // for (it = _x->vertices_begin(); it != _x->vertices_end(); ++it) {
      // 	out.push_back(Cell((Cell_handle)(_x->neighbor(_x->index(it)))));
      // }
      return out;
    }

    void set_vertex(int i, Vertex v) { _x->set_vertex(i, v._x); }
    void set_neighbor(int i, Cell c) { _x->set_neighbor(i, c._x); }

  };

  bool are_equal(Face f1, Face f2) const {
    if (f1.dim() != f2.dim())
      return false;
    int i1, i2;
    Vertex v1;
    for (i1 = 0; i1 < (f1.dim()+1); i1++) {
      bool match = false;
      v1 = f1.vertex(i1);
      for (i2 = 0; i2 < (f2.dim()+1); i2++) {
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
  bool are_equal(Facet f1, Facet f2) const {
    int i1, i2;
    Vertex v1;
    int ndim = num_dims();
    for (i1 = 0; i1 < (ndim); i1++) {
      bool match = false;
      v1 = f1.vertex(i1);
      for (i2 = 0; i2 < (ndim); i2++) {
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

  // // Testing incidence to the infinite vertex
  bool is_infinite(Vertex x) const { return T.is_infinite(x._x); }
  bool is_infinite(Face x) const { return T.is_infinite(x._x); }
  bool is_infinite(Facet x) const { return T.is_infinite(x._x); }
  bool is_infinite(Cell x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_verts_iter x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_cells_iter x) const { return T.is_infinite(x._x); }

  // Constructs incident
  std::vector<Vertex> incident_vertices(Vertex x) {
    std::set<Vertex> sout;
    std::vector<Vertex> out;
    std::vector<Face> faces = incident_faces(x, 1);
    Vertex v;
    std::size_t i, j;
    for (i = 0; i < faces.size(); i++) {
      for (j = 0; j < 2; j++) {
  	v = faces[i].vertex(j);
  	if (v.info() != x.info())
	  sout.insert(v);
  	  // out.push_back(v);
      }
    }
    std::copy(sout.begin(),sout.end(),std::back_inserter(out));
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

  std::vector<Face> incident_faces(Vertex x, int i) {
    std::vector<Face> out;
    T.tds().incident_faces(x._x, i, wrap_insert_iterator<Face,Face_handle>(out));
    return out;
  }
  std::vector<Face> incident_faces(Face x, int face_dim) {
    // Using combinatorics for seleting K from a set of N
    // http://stackoverflow.com/questions/12991758/creating-all-possible-k-combinations-of-n-items-in-c
    std::vector<Face> out;
    int i, j, k, l;
    int K, N;
    Face f;
    Cell c;
    int idx = 0;
    if (face_dim < x.dim()) {
      K = face_dim+1;
      N = x.dim()+1;
      c = x.cell();
      std::string bitmask(K, 1); // K leading 1's
      bitmask.resize(N, 0); // N-K trailing 0's
      do {
	f = Face(c);
	for (i = 0, j = 0; i < N; ++i) { // [0..N-1] integers
	  if (bitmask[i]) {
	    f.set_index(j, x.ind(i));
	    j++;
	  }
	}
	out.push_back(f);
      } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    } else {
      std::string cellmask(D+1, 0);
      std::set<Face> sout;
      std::vector<Cell> cells = incident_cells(x);
      // std::vector<Vertex> verts = incident_vertices(x);
      std::vector<Face> faces;
      K = std::max(1, face_dim-x.dim());
      N = D - x.dim();
      for (i = 0; i < (int)(cells.size()); i++) {
	c = cells[i];
	std::string facemask(x.dim()+1, 1);
	if (face_dim == x.dim())
	  facemask[0] = 0;
	do {
	  for (j = 0; j < (D+1); j++)
	    cellmask[j] = 1;
	  for (j = 0; j < (x.dim()+1); j++) {
	    if (c.has_vertex(x.vertex(j), &idx))
	      cellmask[idx] = 0;
	  }
	  std::string bitmask(K, 1); // K leading 1's
	  bitmask.resize(N, 0); // N-K trailing 0's
	  do {
	    f = Face(c);
	    l = 0;
	    for (j = 0; j < (x.dim()+1); j++) {
	      if (facemask[j])
		if (c.has_vertex(x.vertex(j), &idx))
		  f.set_index(l, idx);
	    }
	    for (j = 0, k = 0; j < (D+1); j++) {
	      if (cellmask[j]) {
		if (bitmask[k]) {
		  f.set_index(l, j);
		  l++;
		}
		k++;
	      }
	    }
	    sout.insert(f);
	  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
	} while (std::prev_permutation(facemask.begin(), facemask.end()));
      }
      std::copy(sout.begin(),sout.end(),std::back_inserter(out));
    }
    return out;
  }
  // std::vector<Face> incident_faces(Face x, int i) {
  //   std::set<Face> sout;
  //   std::vector<Vertex> verts = incident_vertices(x);
  //   std::vector<Face> faces;
  //   std::size_t j, k;
  //   for (j = 0; j < verts.size(); j++) {
  //     faces = incident_faces(verts[j], i);
  //     for (k = 0; k < faces.size(); k++) {
  // 	if (faces[i] != x)
  // 	// if (!are_equal(faces[k], x))
  // 	  sout.insert(faces[k]);
  //     }
  //   }
  //   std::vector<Face> out;
  //   std::copy(sout.begin(),sout.end(),std::back_inserter(out));
  //   return out;
  // }
  std::vector<Face> incident_faces(Cell x, int face_dim) {
    // Using combinatorics for seleting K from a set of N
    // http://stackoverflow.com/questions/12991758/creating-all-possible-k-combinations-of-n-items-in-c
    int K = face_dim+1; // Number of vertices in face of dim face_dim
    int N = D+1; // Number of vertices in cell of dim D
    std::vector<Face> out;
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    int i, j;
    Face f;
    do {
      f = Face(x);
      for (i = 0, j = 0; i < N; ++i) { // [0..N-1] integers
	if (bitmask[i]) {
	  f.set_index(j, i);
	  j++;
	}
      }
      out.push_back(f);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
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

  int mirror_index(Cell x, int i) const { return x._x->mirror_index(i); }
  Vertex mirror_vertex(Cell x, int i) const {
    return x._x->mirror_vertex(i, num_dims());
  }

  void circumcenter(Cell x, double* out) const {
    uint32_t i;
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
  
  double dual_volume(const Vertex v) {
    std::vector<Face> edges = incident_faces(v, 1);
    std::vector<Cell> cells;
    Point orig = v._x->point();
    Point midp;
    double vol = 0.0;
    std::size_t i, j, k;
    std::vector<Point> centers;
    std::vector<Point> pts;
    for (i = 0; i < edges.size(); i++) {
      if (T.is_infinite(edges[i]._x))
	return -1.0;

      midp = T.geom_traits().midpoint_d_object()(edges[i].vertex(0)._x->point(),
                                                 edges[i].vertex(1)._x->point());

      cells = incident_cells(edges[i]);

      centers.clear();
      for (j = 0; j < cells.size(); j++)
  	centers.push_back(cells[j]._x->circumcenter());
      centers.push_back(centers[0]);
      
      for (k = 0; k < (centers.size() - T.tds().current_dimension() + 1); k++) {
  	pts.clear();
  	pts.push_back(orig);
  	pts.push_back(midp);
  	for (j = 0; j < (std::size_t)(T.tds().current_dimension() - 1); j++) {
  	  pts.push_back(centers[k+j]);
  	}
  	vol += n_simplex_volume(pts);
      }
    }
    return vol;
  }
  void dual_volumes(double *vols) {
    Finite_vertex_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      vols[(uint64_t)(it->data())] = dual_volume(Vertex(it.base()));
    }
  }

  // // Write works, read dosn't
  // void write_to_file(const char* filename) const
  // {
  //   std::ofstream os(filename, std::ios::binary);
  //   if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
  //   else {
  //     os << T;
  //   }
  // }
  // void read_from_file(const char* filename)
  // {
  //   std::ifstream is(filename, std::ios::binary);
  //   if (!is) std::cerr << "Error cannot open file: " << filename << std::endl;
  //   else {
  //     is >> T;
  //   }
  // }

  void write_to_file(const char* filename) const
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      write_to_buffer(os);
      os.close();
    }
  }
  
  void write_to_buffer(std::ofstream &os) const {

    // Header
    int n = static_cast<int>(T.number_of_vertices());
    int m = static_cast<int>(T.number_of_full_cells());
    int d = static_cast<int>(T.current_dimension());
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return;
    }

    Vertex_const_hash V;
    Cell_hash C;

    // first (infinite) vertex
    Vertex_handle vit;
    int inum = 0, i;
    Vertex_const_handle v = T.infinite_vertex();
    V[v] = -1;

    // other vertices
    Info info;
    double x;
    for ( Vertex_const_iterator vit = T.vertices_begin();
	  vit != T.vertices_end(); ++vit) {
      if ( v != vit ) {
        V[vit] = inum++;
	info = vit->data();
	os.write((char*)&info, sizeof(Info));
  	for (i = 0; i < d; i++) {
	  x = (double)(vit->point()[i]);
	  os.write((char*)&x, sizeof(double));
  	}
      }
    }

    // vertices of the cells
    inum = 0;
    int index;
    for( Cell_const_iterator ib = T.full_cells_begin();
  	 ib != T.full_cells_end(); ++ib) {
      C[ib] = inum++;
      for (int j = 0; j < dim ; ++j) {
        index = V[ib->vertex(j)];
        os.write((char*)&index, sizeof(int));
      }
    }

    // neighbor pointers of the cells
    for( Cell_const_iterator it = T.full_cells_begin();
  	 it != T.full_cells_end(); ++it) {
      for (int j = 0; j < d+1; ++j) {
        index = C[it->neighbor(j)];
        os.write((char*)&index, sizeof(int));
      }
    }

  }

  void read_from_file(const char* filename)
  {
    std::ifstream is(filename, std::ios::binary);
    if (!is) std::cerr << "Error cannot open file: " << filename << std::endl;
    else {
      read_from_buffer(is);
      is.close();
    }
  }
  
  void read_from_buffer(std::ifstream &is) {
    updated = true;

    if (T.number_of_vertices() != 0)
      T.clear();

    // header
    int n, m, d;
    is.read((char*)&n, sizeof(int));
    is.read((char*)&m, sizeof(int));
    is.read((char*)&d, sizeof(int));

    if (n==0) {
      return;
    }

    T.tds().set_current_dimension(d);

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);

    // infinite vertex
    V[n] = T.infinite_vertex();

    // read vertices
    int i;
    Info info;
    double x;
    std::vector<double> vp;
    for(i = 0; i < n; ++i) {
      is.read((char*)&info, sizeof(Info));
      vp.clear();
      for (int j = 0; j < d; j++)
	is.read((char*)&x, sizeof(double));
      vp.push_back(x);
      V[i] = T.tds().new_vertex();
      V[i]->set_point(Point(vp.begin(), vp.end()));
      V[i]->data() = info;
    }

    // First cell
    int index;
    int dim = (d == -1 ? 1 : d + 1);
    i = 0;
    if (T.full_cells_begin() != T.full_cells_end()) {
      C[i] = T.full_cells_begin();
      for(int j = 0; j < dim ; ++j){
	is.read((char*)&index, sizeof(int));
        C[i]->set_vertex(j, V[index]);
        V[index]->set_full_cell(C[i]);
      }
      i++;
    }

    // Creation of the cells
    for( ; i < m; ++i) {
      C[i] = T.tds().new_full_cell() ;
      for(int j = 0; j < dim ; ++j){
	is.read((char*)&index, sizeof(int));
        C[i]->set_vertex(j, V[index]);
        V[index]->set_full_cell(C[i]);
      }
    }

    // Setting the neighbor pointers
    for(i = 0; i < m; ++i) {
      for(int j = 0; j < d+1; ++j) {
	is.read((char*)&index, sizeof(int));
        C[i]->set_neighbor(j, C[index]);
      }
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
    for( Cell_const_iterator ib = T.full_cells_begin();
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
    for( Cell_const_iterator it = T.full_cells_begin();
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
    for( Cell_const_iterator ib = T.full_cells_begin();
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
    for( Cell_const_iterator it = T.full_cells_begin();
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
    for( Cell_const_iterator ib = T.full_cells_begin();
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
    for( Cell_const_iterator it = T.full_cells_begin();
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

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);

    // infinite vertex
    V[n] = T.infinite_vertex();

    // read vertices
    I i;
    std::vector<double> vp;
    for(i = 0; i < n; ++i) {
      vp.clear();
      for (int j = 0; j < d; j++)
  	vp.push_back(vert_pos[d*i + j]);
      V[i] = T.tds().new_vertex();
      V[i]->set_point(Point(vp.begin(), vp.end()));
      V[i]->data() = vert_info[i];
    }

    // First cell
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    i = 0;
    if (T.full_cells_begin() != T.full_cells_end()) {
      C[i] = T.full_cells_begin();
      for(int j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
        if (index == idx_inf)
          v = V[n];
        else
          v = V[index];
        C[i]->set_vertex(j, v);
        v->set_full_cell(C[i]);
      }
      i++;
    }

    // Creation of the cells
    for( ; i < m; ++i) {
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
      V[i]->set_point(Point(vp.begin(), vp.end()));
      V[i]->data() = (Info)(i);
    }

    // First cell
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    i = 0;
    if (T.tds().full_cells_begin() != T.tds().full_cells_end()) {
      C[i] = T.tds().full_cells_begin();
      for(int j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
        if (index == idx_inf)
          v = V[n];
        else
          v = V[index];
        C[i]->set_vertex(j, v);
        v->set_full_cell(C[i]);
      }
      i++;
    }

    // Creation of the cells
    for( ; i < m; ++i) {
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

  }

  void info_ordered_vertices(double* pos) const {
    Info i;
    Point p;
    int j;
    int d = T.current_dimension();
    Finite_vertex_const_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      i = it->data();
      p = it->point();
      for (j = 0; j < d; j++)
  	pos[d*i + j] = p[j];
    }
  }

  void vertex_info(Info* verts) const {
    int i = 0;
    Finite_vertex_const_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      verts[i] = it->data();
      i++;
    }
  }

  bool intersect_sph_box(Point c, double r, double *le, double *re) const {
    for (int i = 0; i < T.current_dimension(); i++) {
      if ((double)(c[i]) < le[i]) {
  	if (((double)(c[i]) + r) < le[i])
  	  return false;
      } else if ((double)(c[i]) > re[i]) {
  	if (((double)(c[i]) - r) > re[i])
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

    for (Cell_const_iterator it = T.full_cells_begin(); it != T.full_cells_end(); it++) {
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
        cr = std::sqrt(static_cast<double>(T.geom_traits().squared_distance_d_object()(p1, cc)));
        for (b = 0; b < nbox; b++) {
          if (intersect_sph_box(cc, cr, left_edges + d*b, right_edges + d*b))
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


