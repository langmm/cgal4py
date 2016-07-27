// TODO: 
// - Add support for argbitrary return objects so that dual can be added
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/squared_distance_2.h>
#include <vector>
#include <set>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <CGAL/Unique_hash_map.h>
#include <stdint.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;

template <typename Info_>
class Delaunay_with_info_2
// class Delaunay_with_info_2 : public CGAL::Delaunay_triangulation_2<K, CGAL::Triangulation_data_structure_2<CGAL::Triangulation_vertex_base_with_info_2<Info_, K>>>
{
 public:
  typedef CGAL::Delaunay_triangulation_2<K, CGAL::Triangulation_data_structure_2<CGAL::Triangulation_vertex_base_with_info_2<Info_, K>>> Delaunay;
  typedef Info_ Info;
  typedef typename Delaunay::Point                     Point;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Edge                      Edge_handle;
  typedef typename Delaunay::Face_handle               Face_handle;
  typedef typename Delaunay::Vertex_circulator         Vertex_circulator;
  typedef typename Delaunay::Edge_circulator           Edge_circulator;
  typedef typename Delaunay::Face_circulator           Face_circulator;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Face_iterator             Face_iterator;
  typedef typename Delaunay::Edge_iterator             Edge_iterator;
  typedef typename Delaunay::All_vertices_iterator     All_vertices_iterator;
  typedef typename Delaunay::All_faces_iterator        All_faces_iterator;
  typedef typename Delaunay::All_edges_iterator        All_edges_iterator;
  typedef typename Delaunay::Finite_vertices_iterator  Finite_vertices_iterator;
  typedef typename Delaunay::Finite_edges_iterator     Finite_edges_iterator;
  typedef typename Delaunay::Triangle                  Triangle;
  typedef typename Delaunay::Locate_type               Locate_type;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Face_handle,int>    Face_hash;
  Delaunay T;
  bool updated = false;
  Delaunay_with_info_2() {};
  Delaunay_with_info_2(double *pts, Info *val, uint32_t n) { insert(pts, val, n); }
  bool is_valid() const { return T.is_valid(); }
  uint32_t num_finite_verts() const { return static_cast<uint32_t>(T.number_of_vertices()); }
  uint32_t num_finite_edges() const {
    Finite_edges_iterator it = T.finite_edges_begin();
    uint32_t count = 0;
    for ( ; it != T.finite_edges_end(); it++)
      count++;
    return count;
  }
  uint32_t num_finite_cells() const { return static_cast<uint32_t>(T.number_of_faces()); }
  uint32_t num_infinite_verts() const { return static_cast<uint32_t>(1); }
  uint32_t num_infinite_edges() const {
    Edge_circulator ec = T.incident_edges(T.infinite_vertex()), done(ec);
    if (ec == 0)
      return 0;
    int count = 0;
    do {
      count++;
    } while (++ec != done);
    return count;
  }
  uint32_t num_infinite_cells() const {
    Face_circulator fc = T.incident_faces(T.infinite_vertex()), done(fc);
    if (fc == 0)
      return 0;
    int count = 0;
    do {
      count++;
    } while (++fc != done);
    return count;
  }
  uint32_t num_verts() const { return (num_finite_verts() + num_infinite_verts()); }
  uint32_t num_edges() const { return (num_finite_edges() + num_infinite_edges()); }
  uint32_t num_cells() const { return (num_finite_cells() + num_infinite_cells()); }

  class Vertex;
  class Cell;

  void insert(double *pts, Info *val, uint32_t n)
  {
    updated = true;
    uint32_t i, j;
    std::vector< std::pair<Point,Info> > points;
    for (i = 0; i < n; i++) {
      j = 2*i;
      points.push_back( std::make_pair( Point(pts[j],pts[j+1]), val[i]) );
    }
    T.insert( points.begin(),points.end() );
  }
  void remove(Vertex v) { updated = true; T.remove(v._x); }
  void clear() { updated = true; T.clear(); }

  Vertex move(Vertex v, double *pos) {
    updated = true;
    Point p = Point(pos[0], pos[1]);
    return Vertex(T.move(v._x, p));
  }
  Vertex move_if_no_collision(Vertex v, double *pos) {
    updated = true;
    Point p = Point(pos[0], pos[1]);
    return Vertex(T.move_if_no_collision(v._x, p));
  }

  Vertex get_vertex(Info index) const {
    Finite_vertices_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      if (it->info() == index)
    	return Vertex(static_cast<Vertex_handle>(it));
    }
    return Vertex(T.infinite_vertex());
  }

  Cell locate(double* pos, int& lt, int& li) const {
    Point p = Point(pos[0], pos[1]);
    Locate_type lt_out;
    Cell out = Cell(T.locate(p, lt_out, li));
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

  class All_verts_iter {
  public:
    All_vertices_iterator _x = All_vertices_iterator();
    All_verts_iter() { _x = All_vertices_iterator(); }
    All_verts_iter(All_vertices_iterator x) { _x = x; }
    All_verts_iter& operator*() { return *this; }
    All_verts_iter& operator++() {
      _x++;
      return *this;
    }
    All_verts_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_verts_iter other) const { return (_x == other._x); }
    bool operator!=(All_verts_iter other) const { return (_x != other._x); }
  };
  All_verts_iter all_verts_begin() const { return All_verts_iter(T.all_vertices_begin()); }
  All_verts_iter all_verts_end() const { return All_verts_iter(T.all_vertices_end()); }

  class Vertex {
  public:
    Vertex_handle _x = Vertex_handle();
    Vertex() { _x = Vertex_handle(); }
    Vertex(Vertex_handle x) { _x = x; }
    Vertex(All_verts_iter x) { _x = static_cast<Vertex_handle>(x._x); }
    bool operator==(Vertex other) const { return (_x == other._x); }
    bool operator!=(Vertex other) const { return (_x != other._x); }
    void point(double* out) const {
      Point p = _x->point();
      out[0] = p.x();
      out[1] = p.y();
    }
    Info info() const { return _x->info(); }
    Cell cell() const { return Cell(_x->face()); }
    void set_cell(Cell c) { _x->set_face(c._x); }
    void set_point(double* x) {
      Point p = Point(x[0], x[1]);
      _x->set_point(p);
    }
  };

  class All_edges_iter {
  public:
    All_edges_iterator _x = All_edges_iterator();
    All_edges_iter() { _x = All_edges_iterator(); }
    All_edges_iter(All_edges_iterator x) { _x = x; }
    All_edges_iter& operator*() { return *this; }
    All_edges_iter& operator++() {
      _x++;
      return *this;
    }
    All_edges_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_edges_iter other) const { return (_x == other._x); }
    bool operator!=(All_edges_iter other) const { return (_x != other._x); }
  };
  All_edges_iter all_edges_begin() const { return All_edges_iter(T.all_edges_begin()); }
  All_edges_iter all_edges_end() const { return All_edges_iter(T.all_edges_end()); }

  class Edge {
  public:
    Edge_handle _x = Edge_handle();
    Edge() {}
    Edge(Edge_handle x) { _x = x; }
    Edge(All_edges_iterator x) { _x = Edge_handle(x->first, x->second); } 
    Edge(Finite_edges_iterator x) { _x = Edge_handle(x->first, x->second); } 
    Edge(Edge_circulator x) { _x = Edge_handle(x->first, x->second); }
    Edge(All_edges_iter x) { _x = Edge_handle(x._x->first, x._x->second); }
    Edge(Cell x, int i) { _x = Edge_handle(x._x, i); }
    Cell cell() const { return Cell(_x.first); }
    int ind() const { return _x.second; }
    Vertex_handle _v1() const { return _x.first->vertex((_x.second+2)%3); }
    Vertex_handle _v2() const { return _x.first->vertex((_x.second+1)%3); }
    Vertex v1() const { return Vertex(_v1()); }
    Vertex v2() const { return Vertex(_v2()); }
    bool operator==(Edge other) { 
      Vertex_handle x1 = _v1(), x2 = _v2();
      Vertex_handle o1 = other._v1(), o2 = other._v2();
      if ((x1 == o1) && (x2 == o2))
    	return true;
      else if ((x1 == o2) && (x2 == o1))
    	return true;
      else
    	return false;
    }
    bool operator!=(Edge other) { 
      Vertex_handle x1 = _v1(), x2 = _v2();
      Vertex_handle o1 = other._v1(), o2 = other._v2();
      if (x1 == o1) {
    	if (x2 != o2)
    	  return true;
    	else
    	  return false;
      } else if (x1 == o2) {
    	if (x2 != o1)
    	  return true;
    	else
    	  return false;
      } else
    	return true;
    }
  };


  // Cell constructs
  class All_cells_iter {
  public:
    All_faces_iterator _x = All_faces_iterator();
    All_cells_iter() { _x = All_faces_iterator(); }
    All_cells_iter(All_faces_iterator x) { _x = x; }
    All_cells_iter& operator*() { return *this; }
    All_cells_iter& operator++() {
      _x++;
      return *this;
    }
    All_cells_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_cells_iter other) const { return (_x == other._x); }
    bool operator!=(All_cells_iter other) const { return (_x != other._x); }
  };
  All_cells_iter all_cells_begin() const { return All_cells_iter(T.all_faces_begin()); }
  All_cells_iter all_cells_end() const { return All_cells_iter(T.all_faces_end()); }

  class Cell {
  public:
    Face_handle _x = Face_handle();
    Cell() { _x = Face_handle(); }
    Cell(Face_handle x) { _x = x; }
    Cell(All_cells_iter x) { _x = static_cast<Face_handle>(x._x); }
    Cell(Vertex v1, Vertex v2, Vertex v3) { _x = Face_handle(v1._x, v2._x, v3._x); }
    Cell(Vertex v1, Vertex v2, Vertex v3, Cell c1, Cell c2, Cell c3) {
      _x = Face_handle(v1._x, v2._x, v3._x, c1._x, c2._x, c3._x); }
    bool operator==(Cell other) const { return (_x == other._x); }
    bool operator!=(Cell other) const { return (_x != other._x); }

    Vertex vertex(int i) const { return Vertex(_x->vertex(i)); }
    bool has_vertex(Vertex v) const { return _x->has_vertex(v._x); }
    bool has_vertex(Vertex v, int *i) const { return _x->has_vertex(v._x, *i); }
    int ind(Vertex v) const { return _x->index(v._x); }

    Cell neighbor(int i) const { return Cell(_x->neighbor(i)); }
    bool has_neighbor(Cell c) const { return _x->has_neighbor(c._x); }
    bool has_neighbor(Cell c, int *i) const { return _x->has_neighbor(c._x, *i); }
    int ind(Cell c) const { return _x->index(c._x); }

    void set_vertex(int i, Vertex v) { _x->set_vertex(i, v._x); }
    void set_vertices() { _x->set_vertices(); }
    void set_vertices(Vertex v1, Vertex v2, Vertex v3) {
      _x->set_vertices(v1._x, v2._x, v3._x); }
    void set_neighbor(int i, Cell c) { _x->set_neighbor(i, c._x); }
    void set_neighbors() { _x->set_neighbors(); }
    void set_neighbors(Cell c1, Cell c2, Cell c3) {
      _x->set_neighbors(c1._x, c2._x, c3._x); }

    void reorient() { _x->reorient(); }
    void ccw_permute() { _x->ccw_permute(); }
    void cw_permute() { _x->cw_permute(); }

    int dimension() const { return _x->dimension(); }
  };


  // Check if construct is incident to the infinite vertex
  bool is_infinite(Vertex x) const { return T.is_infinite(x._x); }
  bool is_infinite(Edge x) const { return T.is_infinite(x._x); }
  bool is_infinite(Cell x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_verts_iter x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_edges_iter x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_cells_iter x) const { return T.is_infinite(x._x); }


  // Constructs incident to a vertex
  std::vector<Vertex> incident_vertices(Vertex x) const {
    std::vector<Vertex> out;
    Vertex_circulator vc = T.incident_vertices(x._x), done(vc);
    if (vc == 0)
      return out;
    do {
      out.push_back(Vertex(static_cast<Vertex_handle>(vc)));
    } while (++vc != done);
    return out;
  }
  std::vector<Edge> incident_edges(Vertex x) const {
    std::vector<Edge> out;
    Edge_circulator ec = T.incident_edges(x._x), done(ec);
    if (ec == 0)
      return out;
    do {
      out.push_back(Edge(ec));
    } while (++ec != done);
    return out;
  }
  std::vector<Cell> incident_cells(Vertex x) const {
    std::vector<Cell> out;
    Face_circulator fc = T.incident_faces(x._x), done(fc);
    if (fc == 0)
      return out;
    do {
      out.push_back(Cell(static_cast<Face_handle>(fc)));
    } while (++fc != done);
    return out;
  }

  // Constructs incident to an edge
  std::vector<Vertex> incident_vertices(Edge x) const {
    std::vector<Vertex> out;
    out.push_back(x.v1());
    out.push_back(x.v2());
    return out;
  }
  std::vector<Edge> incident_edges(Edge x) const {
    uint32_t i;
    std::vector<Edge> out1, out2, out;
    out1 = incident_edges(x.v1());
    out2 = incident_edges(x.v2());
    for (i = 0; i < out1.size(); i++) {
      if (out1[i] != x)
	out.push_back(out1[i]);
    }
    for (i = 0; i < out2.size(); i++) {
      if (out2[i] != x)
	out.push_back(out2[i]);
    }
    return out;
  }
  std::vector<Cell> incident_cells(Edge x) const {
    uint32_t i;
    std::vector<Cell> out1, out2;
    Vertex v1 = x.v1(), v2 = x.v2();
    out1 = incident_cells(v1);
    out2 = incident_cells(v2);
    for (i = 0; i < out2.size(); i++) {
      if ((!out2[i].has_vertex(v1)) && (!out2[i].has_vertex(v2)))
	out1.push_back(out2[i]);
    }
    return out1;
  }

  // Constructs incident to a cell
  std::vector<Vertex> incident_vertices(Cell x) const {
    uint32_t i;
    std::vector<Vertex> out;
    for (i = 0; i < 3; i++) 
      out.push_back(x.vertex(i));
    return out;
  }
  std::vector<Edge> incident_edges(Cell x) const {
    // uint32_t i1, i2;
    std::vector<Edge> out;
    for (int i = 0; i < 3; i++)
      out.push_back(Edge(x, i));
    return out;
  }
  std::vector<Cell> incident_cells(Cell x) const {
    uint32_t i;
    std::vector<Cell> out;
    for (i = 0; i < 3; i++)
      out.push_back(x.neighbor(i));
    return out;
  }

  Vertex nearest_vertex(double* pos) const {
    Point p = Point(pos[0], pos[1]);
    Vertex out = Vertex(T.nearest_vertex(p));
    return out;
  }

  void circumcenter(Cell x, double* out) const {
    if (T.is_infinite(x._x)) {
      out[0] = std::numeric_limits<double>::infinity();
      out[1] = std::numeric_limits<double>::infinity();
    } else {
      Point p = T.circumcenter(x._x);
      out[0] = p.x();
      out[1] = p.y();
    }
  }

  double dual_area(const Vertex v) const {

    Face_circulator fstart = T.incident_faces(v._x);
    Face_circulator fcit = fstart;
    std::vector<Point> pts;
    pts.push_back(T.circumcenter(fstart));
    fcit++;
    for ( ; fcit != fstart; fcit++) {
      if (T.is_infinite(fcit))
	return -1.0;
      Point dual = T.circumcenter(fcit);
      pts.push_back(dual);
    }
    pts.push_back(T.circumcenter(fstart));
    double vol = 0.0;
    Point orig = v._x->point();
    for (uint32_t i=0 ; i<pts.size()-1 ; i++) {
      vol = vol + Triangle(orig,pts[i],pts[i+1]).area();
    }

    return vol;
  }

  double length(const Edge e) const {
    if (is_infinite(e))
      return -1.0;
    Vertex_handle v1 = e._v1(), v2 = e._v2();
    Point p1 = v1->point();
    Point p2 = v2->point();
    double out = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, p2)));
    return out;
  }

  bool flip(Cell x, int i) { 
    updated = true;
    T.flip(x._x, i); 
    return true;
  }
  bool flip(Edge x) {
    updated = true;
    T.flip(x.cell()._x, x.ind());
    return true;
  }
  // for completeness with 3D case
  void flip_flippable(Cell x, int i) { 
    updated = true;
    T.flip(x._x, i); 
  }
  void flip_flippable(Edge x) { 
    updated = true;
    T.flip(x.cell()._x, x.ind()); 
  }

  std::vector<Edge> get_boundary_of_conflicts(double* pos, Cell start) const {
    std::vector<Edge> out;
    Point p = Point(pos[0], pos[1]);
    T.get_boundary_of_conflicts(p, wrap_insert_iterator<Edge,Edge_handle>(out), 
				start._x);
    return out;
  }
  std::vector<Cell> get_conflicts(double* pos, Cell start) const {
    std::vector<Cell> out;
    Point p = Point(pos[0], pos[1]);
    T.get_conflicts(p, wrap_insert_iterator<Cell,Face_handle>(out), start._x);
    return out;
  }
  std::pair<std::vector<Cell>,std::vector<Edge>>
    get_conflicts_and_boundary(double* pos, Cell start) const {
    std::pair<std::vector<Cell>,std::vector<Edge>> out;
    std::vector<Cell> fit;
    std::vector<Edge> eit;
    Point p = Point(pos[0], pos[1]);
    T.get_conflicts_and_boundary(p, 
				 wrap_insert_iterator<Cell,Face_handle>(fit), 
				 wrap_insert_iterator<Edge,Edge_handle>(eit), 
				 start._x);
    out = std::make_pair(fit, eit);
    return out;
  }

  int side_of_oriented_circle(Cell f, const double* pos) const {
    if (T.is_infinite(f._x))
      return -1;
    else if (std::isinf(pos[0]) || std::isinf(pos[1])) {
      return 1;
    } else {
      Point p = Point(pos[0], pos[1]);
      return (int)T.side_of_oriented_circle(f._x, p);
    }
  }

  void write_to_file(const char* filename) const
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      // Header
      int n = static_cast<int>(T.number_of_vertices());
      int m = static_cast<int>(T.tds().number_of_full_dim_faces());
      int d = static_cast<int>(T.dimension());
      os.write((char*)&n, sizeof(int));
      os.write((char*)&m, sizeof(int));
      os.write((char*)&d, sizeof(int));
      if (n==0) {
	os.close();
	return;
      }

      Vertex_hash V;
      Face_hash F;
      
      // first (infinite) vertex 
      Info info;
      double x, y;
      int inum = 0;
      Vertex_handle v = T.infinite_vertex();
      if ( v != Vertex_handle()) {
	V[v] = inum++;
      }
      
      // other vertices
      for( All_vertices_iterator vit = T.tds().vertices_begin(); vit != T.tds().vertices_end() ; ++vit) {
	if ( v != vit ) {
	  V[vit] = inum++;
	  info = static_cast<Info>(vit->info());
	  x = static_cast<double>(vit->point().x());
	  y = static_cast<double>(vit->point().y());
	  os.write((char*)&info, sizeof(Info));
	  os.write((char*)&x, sizeof(double));
	  os.write((char*)&y, sizeof(double));
	}
      }
      
      // vertices of the faces
      inum = 0;
      int dim = (d == -1 ? 1 :  d + 1);
      int index;
      int nvert = 0;
      for (All_faces_iterator ib = T.tds().face_iterator_base_begin();
	   ib != T.tds().face_iterator_base_end(); ++ib) {
	F[ib] = inum++;
	for (int j = 0; j < dim ; ++j) {
	  index = V[ib->vertex(j)];
	  os.write((char*)&index, sizeof(int));
	  nvert++;
	}
      }
  
      // neighbor pointers of the faces
      for (All_faces_iterator it = T.tds().face_iterator_base_begin();
	   it != T.tds().face_iterator_base_end(); ++it) {
	for (int j = 0; j < d+1; ++j){
	  index = F[it->neighbor(j)];
	  os.write((char*)&index, sizeof(int));
	}
      }

      os.close();
    }
  }

  void read_from_file(const char* filename)
  {
    std::ifstream is(filename, std::ios::binary);
    if (!is) std::cerr << "Error cannot open file: " << filename << std::endl;
    else {

      if (T.number_of_vertices() != 0) 
      	T.clear();
  
      // header
      int n, m, d;
      is.read((char*)&n, sizeof(int));
      is.read((char*)&m, sizeof(int));
      is.read((char*)&d, sizeof(int));

      if (n==0) {
	T.set_infinite_vertex(Vertex_handle());
	return;
      }

      T.tds().set_dimension(d);

      All_faces_iterator to_delete = T.tds().face_iterator_base_begin();

      std::vector<Vertex_handle> V(n+1);
      std::vector<Face_handle> F(m);

      // infinite vertex
      int i = 0;
      V[0] = T.infinite_vertex();
      ++i;

      // read vertices
      Info info;
      double x, y;
      for( ; i < (n+1); ++i) {
	V[i] = T.tds().create_vertex();
	is.read((char*)&info, sizeof(Info));
	is.read((char*)&x, sizeof(double));
	is.read((char*)&y, sizeof(double));
	(*(V[i])).point() = Point(x,y);
	(*(V[i])).info() = info;
      }

      // Creation of the faces
      int index;
      int dim = (d == -1 ? 1 : d + 1);
      {
	int nvert = 0;
	for(i = 0; i < m; ++i) {
	  F[i] = T.tds().create_face() ;
	  for(int j = 0; j < dim ; ++j){
	    is.read((char*)&index, sizeof(int));
	    F[i]->set_vertex(j, V[index]);
	    // The face pointer of vertices is set too often,
	    // but otherwise we had to use a further map 
	    V[index]->set_face(F[i]);
	    nvert++;
	  }
	}
      }

      // Setting the neighbor pointers
      {
	for(i = 0; i < m; ++i) {
	  for(int j = 0; j < d+1; ++j){
	    is.read((char*)&index, sizeof(int));
	    F[i]->set_neighbor(j, F[index]);
	  }
	}
      }

      // Remove flat face
      T.tds().delete_face(to_delete);

      T.set_infinite_vertex(V[0]);
      is.close();
    }
  }

  void info_ordered_vertices(double* pos) const {
    Info i;
    Point p;
    for (Finite_vertices_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      i = it->info();
      p = it->point();
      pos[2*i + 0] = p.x();
      pos[2*i + 1] = p.y();
    }
  }

  void vertex_info(Info* verts) const {
    int i = 0;
    for (Finite_vertices_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      verts[i] = it->info();
      i++;
    }
  }

  void edge_info(Info* edges) const {
    int i = 0;
    Info i1, i2;
    for (Finite_edges_iterator it = T.finite_edges_begin(); it != T.finite_edges_end(); it++) {
      i1 = it->first->vertex(T.cw(it->second))->info();
      i2 = it->first->vertex(T.ccw(it->second))->info();
      edges[2*i + 0] = i1;
      edges[2*i + 1] = i2;
      i++;
    }
  }

  void outgoing_points(double *left_edge, double *right_edge, bool periodic,
                       std::vector<Info>& lx, std::vector<Info>& ly,
                       std::vector<Info>& rx, std::vector<Info>& ry,
                       std::vector<Info>& alln) const
  {
    Vertex_handle v;
    Info hv;
    Point cc, p1;
    double cr;
    int i;

    for (All_faces_iterator it = T.all_faces_begin(); it != T.all_faces_end(); it++) {
      if (T.is_infinite(it) == true) {
        if (periodic == false) {
          for (i = 0; i < 3; i++) {
            v = it->vertex(i);
            if (T.is_infinite(v) == false) {
              hv = v->info();
              alln.push_back(hv);
            }
          }
        }
      } else {
        p1 = it->vertex(0)->point();
        cc = T.circumcenter(it);
        cr = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, cc)));
        if ((cc.x() + cr) > right_edge[0])
          for (i = 0; i < 3; i++) rx.push_back((it->vertex(i))->info());
        if ((cc.y() + cr) > right_edge[1])
          for (i = 0; i < 3; i++) ry.push_back((it->vertex(i))->info());
        if ((cc.x() - cr) < left_edge[0])
          for (i = 0; i < 3; i++) lx.push_back((it->vertex(i))->info());
        if ((cc.y() - cr) < left_edge[1])
          for (i = 0; i < 3; i++) ly.push_back((it->vertex(i))->info());
      }
    }
    std::sort( alln.begin(), alln.end() );
    std::sort( lx.begin(), lx.end() );
    std::sort( ly.begin(), ly.end() );
    std::sort( rx.begin(), rx.end() );
    std::sort( ry.begin(), ry.end() );
    alln.erase( std::unique( alln.begin(), alln.end() ), alln.end() );
    lx.erase( std::unique( lx.begin(), lx.end() ), lx.end() );
    ly.erase( std::unique( ly.begin(), ly.end() ), ly.end() );
    rx.erase( std::unique( rx.begin(), rx.end() ), rx.end() );
    ry.erase( std::unique( ry.begin(), ry.end() ), ry.end() );
    
  }
};

