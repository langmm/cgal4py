#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_circumcenter_3.h>
#include <CGAL/squared_distance_3.h>
#include <vector>
#include <set>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
// #include <CGAL/IO/io.h>
#include <CGAL/Unique_hash_map.h>
#include <stdint.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_cell_base_with_circumcenter_3<K>      Cb3;


template <typename Info_>
class Delaunay_with_info_3
// class Delaunay_with_info_3 : public CGAL::Delaunay_triangulation_3<K, CGAL::Triangulation_data_structure_3<CGAL::Triangulation_vertex_base_with_info_3<Info_, K>>>
{
 public:
  typedef CGAL::Delaunay_triangulation_3<K, CGAL::Triangulation_data_structure_3<CGAL::Triangulation_vertex_base_with_info_3<Info_, K>, Cb3>> Delaunay;
  //CGAL::Triangulation_cell_base_with_circumcenter_3<K>>> Delaunay;
  typedef typename Delaunay::Point                     Point;
  typedef typename Delaunay::Edge                      Edge;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Cell_handle               Cell_handle;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Cell_iterator             Cell_iterator;
  typedef typename Delaunay::Edge_iterator             Edge_iterator;
  typedef typename Delaunay::All_vertices_iterator     All_vertices_iterator;
  typedef typename Delaunay::All_cells_iterator        All_cells_iterator;
  typedef typename Delaunay::All_edges_iterator        All_edges_iterator;
  typedef typename Delaunay::Finite_vertices_iterator  Finite_vertices_iterator;
  typedef typename Delaunay::Finite_edges_iterator     Finite_edges_iterator;
  typedef typename Delaunay::Tetrahedron               Tetrahedron;
  typedef typename Delaunay::Facet_circulator          Facet_circulator;
  typedef typename Delaunay::Cell_circulator           Cell_circulator;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Cell_handle,int>    Cell_hash;
  typedef Info_ Info;
  Delaunay T;
  Delaunay_with_info_3() {};
  Delaunay_with_info_3(double *pts, Info *val, uint32_t n) { insert(pts, val, n); }
  uint32_t num_finite_verts() { return static_cast<uint32_t>(T.number_of_vertices()); }
  uint32_t num_finite_edges() { return static_cast<uint32_t>(T.number_of_finite_edges()); }
  uint32_t num_finite_cells() { return static_cast<uint32_t>(T.number_of_finite_cells()); }
  uint32_t num_infinite_verts() { return 1; }
  uint32_t num_infinite_edges() { return (T.number_of_edges() - T.number_of_finite_edges()); }
  uint32_t num_infinite_cells() { return (T.number_of_cells() - T.number_of_finite_cells()); }
  uint32_t num_verts() { return (T.number_of_vertices() + num_infinite_verts()); }
  uint32_t num_edges() { return T.number_of_edges(); }
  uint32_t num_cells() { return T.number_of_cells(); }

  class All_verts_iter {
  public:
    All_vertices_iterator _v = All_vertices_iterator();
    All_verts_iter() {
      _v = All_vertices_iterator();
    }
    All_verts_iter(All_vertices_iterator v) { _v = v; }
    All_verts_iter& operator*() { return *this; }
    All_verts_iter& operator++() {
      _v++;
      return *this;
    }
    All_verts_iter& operator--() {
      _v--;
      return *this;
    }
    bool operator==(All_verts_iter other) { return (_v == other._v); }
    bool operator!=(All_verts_iter other) { return (_v != other._v); }
    void point(double* out) {
      Point p = _v->point();
      out[0] = p.x();
      out[1] = p.y();
      out[2] = p.z();
    }
    std::vector<double> point() {
      std::vector<double> out;
      Point p = _v->point();
      out.push_back(p.x());
      out.push_back(p.y());
      out.push_back(p.z());
      return out;
    }
    Info info() {
      return _v->info();
    }
  };
  All_verts_iter all_verts_begin() { return All_verts_iter(T.all_vertices_begin()); }
  All_verts_iter all_verts_end() { return All_verts_iter(T.all_vertices_end()); }

  class All_cells_iter {
  public:
    All_cells_iterator _c = All_cells_iterator();
    All_cells_iter() {
      _c = All_cells_iterator();
    }
    All_cells_iter(All_cells_iterator c) { _c = c; }
    All_cells_iter& operator*() { return *this; }
    All_cells_iter& operator++() {
      _c++;
      return *this;
    }
    All_cells_iter& operator--() {
      _c--;
      return *this;
    }
    bool operator==(All_cells_iter other) { return (_c == other._c); }
    bool operator!=(All_cells_iter other) { return (_c != other._c); }
  };
  All_cells_iter all_cells_begin() { return All_cells_iter(T.all_cells_begin()); }
  All_cells_iter all_cells_end() { return All_cells_iter(T.all_cells_end()); }

  bool is_infinite(All_verts_iter x) { return T.is_infinite(x._v); }
  bool is_infinite(All_cells_iter x) { return T.is_infinite(x._c); }
  void circumcenter(All_cells_iter x, double* out) {
    Point p = (x._c)->circumcenter();
    out[0] = p.x();
    out[1] = p.y();
    out[2] = p.z();
  }

  void write_to_file(const char* filename)
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      // Header
      int n = static_cast<int>(T.number_of_vertices());
      int m = static_cast<int>(T.tds().number_of_cells());
      int d = static_cast<int>(T.dimension());
      os.write((char*)&n, sizeof(int));
      os.write((char*)&m, sizeof(int));
      os.write((char*)&d, sizeof(int));
      if (n==0) {
	os.close();
	return;
      }
      // printf("Wrote %d vertices, %d cells, for %d dimensions\n",n,m,d);

      Vertex_hash V;
      Cell_hash C;
      
      // first (infinite) vertex 
      int inum = 0;
      Vertex_handle v = T.infinite_vertex();
      if ( v != Vertex_handle()) {
	V[v] = inum++;
      }
      
      // other vertices
      Info info;
      double x, y, z;
      for( All_vertices_iterator vit = T.tds().vertices_begin(); vit != T.tds().vertices_end() ; ++vit) {
	if ( v != vit ) {
	  V[vit] = inum++;
	  info = static_cast<Info>(vit->info());
	  x = static_cast<double>(vit->point().x());
	  y = static_cast<double>(vit->point().y());
	  z = static_cast<double>(vit->point().z());
	  os.write((char*)&info, sizeof(Info));
	  os.write((char*)&x, sizeof(double));
	  os.write((char*)&y, sizeof(double));
	  os.write((char*)&z, sizeof(double));
	}
      }
      // printf("%d nverts, %d inum\n",T.number_of_vertices(),inum);
      
      // vertices of the cells
      inum = 0;
      int dim = (d == -1 ? 1 :  d + 1);
      int index;
      for( Cell_iterator ib = T.tds().cells_begin(); 
	   ib != T.tds().cells_end(); ++ib) {
	C[ib] = inum++;
	for(int j = 0; j < dim ; ++j) {
	  index = V[ib->vertex(j)];
	  os.write((char*)&index, sizeof(int));
	}
      }
      // printf("%d ncells, %d inum\n",T.tds().number_of_cells(),inum);
  
      // neighbor pointers of the cells
      for( Cell_iterator it = T.tds().cells_begin();
	   it != T.tds().cells_end(); ++it) {
	for(int j = 0; j < d+1; ++j){
	  index = C[it->neighbor(j)];
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
	return;
      }

      T.tds().set_dimension(d);
      All_cells_iterator to_delete = T.tds().cells_begin();

      std::vector<Vertex_handle> V(n+1);
      std::vector<Cell_handle> C(m);

      // infinite vertex
      int i = 0;
      V[0] = T.infinite_vertex();
      ++i;

      // read vertices
      Info info;
      double x, y, z;
      for( ; i <= n; ++i) {
	V[i] = T.tds().create_vertex();
	is.read((char*)&info, sizeof(Info));
	is.read((char*)&x, sizeof(double));
	is.read((char*)&y, sizeof(double));
	is.read((char*)&z, sizeof(double));
	(*(V[i])).point() = Point(x,y,z);
	(*(V[i])).info() = info;
      }

      // Creation of the cells
      int index;
      int dim = (d == -1 ? 1 : d + 1);
      {
	for(i = 0; i < m; ++i) {
	  C[i] = T.tds().create_cell() ;
	  for(int j = 0; j < dim ; ++j){
	    is.read((char*)&index, sizeof(int));
	    C[i]->set_vertex(j, V[index]);
	    V[index]->set_cell(C[i]);
	  }
	}
      }

      // Setting the neighbor pointers
      {
	for(i = 0; i < m; ++i) {
	  for(int j = 0; j < d+1; ++j){
	    is.read((char*)&index, sizeof(int));
	    C[i]->set_neighbor(j, C[index]);
	  }
	}
      }

      // delete flat cell
      T.tds().delete_cell(to_delete);

      is.close();
    }
  }
  void insert(double *pts, Info *val, uint32_t n)
  {
    uint32_t i, j;
    std::vector< std::pair<Point,Info> > points;
    for (i = 0; i < n; i++) {
      j = 3*i;
      points.push_back( std::make_pair( Point(pts[j],pts[j+1],pts[j+2]), val[i]) );
    }
    T.insert( points.begin(),points.end() );
  }
  void edge_info(std::vector< std::pair<Info,Info> >& edges)
  {
    Info i1, i2;
    for (Finite_edges_iterator it = T.finite_edges_begin(); it != T.finite_edges_end(); it++) {
      i1 = it->first->vertex(it->second)->info();
      i2 = it->first->vertex(it->third)->info();
      edges.push_back( std::make_pair( i1, i2 ) );
    }
  }

  double dual_volume(const Vertex_handle &v) {
    std::list<Edge> edges;
    T.incident_edges(v, std::back_inserter(edges));

    Point orig = v->point();
    double vol = 0.0;
    for (typename std::list<Edge>::iterator eit = edges.begin() ;
         eit != edges.end() ; ++eit) {

      Facet_circulator fstart = T.incident_facets(*eit);
      Facet_circulator fcit = fstart;
      std::vector<Point> pts;
      do {
	if (T.is_infinite(fcit->first))
	  return -1.0;
        Point dual_orig = fcit->first->circumcenter();
        pts.push_back(dual_orig);
        ++fcit;
      } while (fcit != fstart);

      for (uint32_t i=1 ; i<pts.size()-1 ; i++)
        vol += Tetrahedron(orig,pts[0],pts[i],pts[i+1]).volume();
    }
    return vol;
  }
  void outgoing_points(double *left_edge, double *right_edge, bool periodic,
                       std::vector<Info>& lx, std::vector<Info>& ly, std::vector<Info>& lz,
                       std::vector<Info>& rx, std::vector<Info>& ry, std::vector<Info>& rz,
                       std::vector<Info>& alln)
  {
    Vertex_handle v;
    Info hv;
    Point cc, p1;
    double cr;
    int i;

    for (All_cells_iterator it = T.all_cells_begin(); it != T.all_cells_end(); it++) {
      if (T.is_infinite(it) == true) {
        if (periodic == false) {
          for (i = 0; i < 4; i++) {
            v = it->vertex(i);
            if (T.is_infinite(v) == false) {
              hv = v->info();
              alln.push_back(hv);
            }
          }
        }
      } else {
        p1 = it->vertex(0)->point();
	cc = it->circumcenter();
        cr = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, cc)));
        if ((cc.x() + cr) > right_edge[0])
          for (i = 0; i < 4; i++) rx.push_back((it->vertex(i))->info());
        if ((cc.y() + cr) > right_edge[1])
          for (i = 0; i < 4; i++) ry.push_back((it->vertex(i))->info());
        if ((cc.z() + cr) > right_edge[2])
          for (i = 0; i < 4; i++) rz.push_back((it->vertex(i))->info());
        if ((cc.x() - cr) < left_edge[0])
          for (i = 0; i < 4; i++) lx.push_back((it->vertex(i))->info());
        if ((cc.y() - cr) < left_edge[1])
          for (i = 0; i < 4; i++) ly.push_back((it->vertex(i))->info());
        if ((cc.z() - cr) < left_edge[2])
          for (i = 0; i < 4; i++) lz.push_back((it->vertex(i))->info());
      }
    }

    std::sort( alln.begin(), alln.end() );
    std::sort( lx.begin(), lx.end() );
    std::sort( ly.begin(), ly.end() );
    std::sort( lz.begin(), lz.end() );
    std::sort( rx.begin(), rx.end() );
    std::sort( ry.begin(), ry.end() );
    std::sort( rz.begin(), rz.end() );
    alln.erase( std::unique( alln.begin(), alln.end() ), alln.end() );
    lx.erase( std::unique( lx.begin(), lx.end() ), lx.end() );
    ly.erase( std::unique( ly.begin(), ly.end() ), ly.end() );
    lz.erase( std::unique( lz.begin(), lz.end() ), lz.end() );
    rx.erase( std::unique( rx.begin(), rx.end() ), rx.end() );
    ry.erase( std::unique( ry.begin(), ry.end() ), ry.end() );
    rz.erase( std::unique( rz.begin(), rz.end() ), rz.end() );
  }
};

