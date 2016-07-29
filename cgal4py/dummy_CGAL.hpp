#include <vector>
#include <utility>
#include <memory>
#include <iterator>
#include <list>

namespace CGAL 
{
enum  Sign
  {   
    NEGATIVE = -1, ZERO = 0, POSITIVE = 1,
    RIGHT_TURN = -1, LEFT_TURN = 1,
    CLOCKWISE = -1, COUNTERCLOCKWISE = 1,
    COLLINEAR = 0, COPLANAR = 0, DEGENERATE = 0,
    ON_NEGATIVE_SIDE = -1, ON_ORIENTED_BOUNDARY = 0, ON_POSITIVE_SIDE = 1,
    SMALLER = -1, EQUAL = 0, LARGER = 1
  };
  typedef Sign Orientation;
  typedef Sign Oriented_side;
  typedef Sign Comparison_result;
  enum  Bounded_side
    {
      ON_UNBOUNDED_SIDE = -1,
      ON_BOUNDARY,
      ON_BOUNDED_SIDE
    };

  class Exact_predicates_inexact_constructions_kernel {
  public:
    class Point {
    public:
      Point() {};
      Point(double x, double y) {};
      double x() { return 0.0; }
      double y() { return 0.0; }
    };

    class Segment {
    public:
      Segment() {};
      Segment(Point p1, Point p2) {};
    };

    class Triangle {
    public:
      Triangle() {};
      Triangle(Point p1, Point p2, Point p3) {};
      double area() { return 0.0; }
    };
  };

  template < class H, bool Const >
  class Iterator;

  template < class T >
  class Container {
  public:
    typedef Container<T>         Self;
    typedef T                    value_type;
    typedef Iterator<Self,false> iterator;
    typedef Iterator<Self,true>  const_iterator;
    
    friend class Iterator<Self,false>;
    friend class Iterator<Self,true>;

    Container() {};
    Container & operator=(const Container &c) {};
    
    iterator begin() { return iterator(); }
    iterator end() { return iterator(); }

    const_iterator begin() const { return const_iterator(); }
    const_iterator end() const { return const_iterator(); }
    
  };
  
  template < class T >
  bool operator==(const Container<T> &lhs, const Container<T> &rhs) { return false; }
  template < class T >
  bool operator!=(const Container<T> &lhs, const Container<T> &rhs) { return false; }
  
  
  template < class H, bool Const >
  class Iterator {
  public:
    typedef Iterator<H, Const> Self;
    typedef typename H::iterator iterator;
    typedef typename H::value_type value_type;
    typedef typename std::conditional<Const, 
				      const value_type*, 
				      value_type*>::type pointer;
    typedef typename std::conditional<Const, 
				      const value_type&, 
				      value_type&>::type reference;
  private:
    pointer p;
  public:    
    Iterator() { p = NULL; }
    // Iterator(const iterator &it) { p = &(*it); }
    Self & operator= (const iterator &it) { return *this; }
    Self & operator++() { return *this; }
    Self & operator--() { return *this; }
    Self operator++(int) { Self tmp(*this); ++(*this); return tmp; }
    Self operator--(int) { Self tmp(*this); --(*this); return tmp; }
    reference operator*() const { return *(p); }
    pointer   operator->() const { return p; }
  };
  
  template < class H, bool Const >
  bool operator==(const Iterator<H,Const> &lhs, const Iterator<H,Const> &rhs) { return false; }
  template < class H, bool Const >
  bool operator!=(const Iterator<H,Const> &lhs, const Iterator<H,Const> &rhs) { return false; }

  template < class I >
  class Filter_iterator {
  public:
    typedef Filter_iterator<I> Self;
    typedef I                                Iterator;
    typedef typename I::reference            reference;
    typedef typename I::pointer              pointer;
    typedef typename I::value_type           value_type;
    // typedef std::iterator_traits<I>          ITI;
    // typedef typename ITI::reference          reference;
    // typedef typename ITI::pointer            pointer;
    // typedef typename ITI::value_type         value_type;
    // typedef typename I::iterator iterator;
    // typedef typename I::value_type value_type;
    // typedef value_type* pointer;
    // typedef value_type* reference;
  private:
    Iterator e_;
    Iterator c_;
    // pointer p;
  public:    
    Filter_iterator()
      : e_(0), c_(0) 
    {}
    Filter_iterator(Iterator e) 
      : e_(e), c_(e)
    {}
    Self & operator++() { return *this; }
    Self & operator--() { return *this; }
    Self operator++(int) { Self tmp(*this); ++(*this); return tmp; }
    Self operator--(int) { Self tmp(*this); --(*this); return tmp; }
    reference operator*() const { return *c_;  }
    pointer operator->() const  { return &*c_; }
    Iterator base() const { return c_; }
    operator Iterator() { return e_; }
  };

  template < class I >
  bool operator==(const Filter_iterator<I> &lhs, const Filter_iterator<I> &rhs) { return false; }
  template < class I >
  bool operator!=(const Filter_iterator<I> &lhs, const Filter_iterator<I> &rhs) { return false; }

  template < class Tds, typename Info_ >
  class Vertex_base {
    typedef Info_                                      Info;
    typedef typename Tds::Face_handle                  Face_handle;
    typedef typename Tds::Vertex_handle                Vertex_handle;
  public:
    typedef typename Exact_predicates_inexact_constructions_kernel::Point Point;
  private:
    Point _p;
    Info_ _info;
  public:
    Face_handle face() const { return Face_handle(); }
    void set_face(Face_handle f) {};
    bool is_valid(bool /*verbose*/=false, int /*level*/= 0) const { return false; }
    void set_point(const Point & p) {};
    const Point&  point() const { return _p; }
    Point&        point() { return _p; }
    const Info& info() const { return _info; }
    Info&       info() { return _info; }
  };

  template < class Tds >
  class Face_base {
    typedef typename Tds::Face_handle                  Face_handle;
    typedef typename Tds::Vertex_handle                Vertex_handle;
  public:
    Face_base() {};
    Face_base(Vertex_handle v0,
	      Vertex_handle v1,
	      Vertex_handle v2) {};
    Face_base(Vertex_handle v0,
	      Vertex_handle v1,
	      Vertex_handle v2,
	      Face_handle n0,
	      Face_handle n1,
	      Face_handle n2) {};
    Vertex_handle vertex(int i) const { return Vertex_handle(); }
    bool has_vertex(Vertex_handle v) const { return false; }
    bool has_vertex(Vertex_handle v, int& i) const { return false; }
    int index(Vertex_handle v) const { return 0; }
    Face_handle neighbor(int i) const { return Face_handle(); }
    bool has_neighbor(Face_handle n) const { return false; }
    bool has_neighbor(Face_handle n, int& i) const { return false; }
    int index(Face_handle n) const { return 0; }
    void set_vertex(int i, Vertex_handle v) {};
    void set_vertices() {};
    void set_vertices(Vertex_handle v0, Vertex_handle v1, Vertex_handle v2) {};
    void set_neighbor(int i, Face_handle n) {};
    void set_neighbors() {};
    void set_neighbors(Face_handle n0, Face_handle n1, Face_handle n2) {};
    void reorient() {};
    void ccw_permute() {};
    void cw_permute() {};
    int dimension() const { return 0; }
    bool is_valid(bool /*verbose*/=false, int /*level*/= 0) const { return false; }
  };

  template <class Tds>
  class Triangulation_ds_edge_iterator_2
  {
  public:
    typedef typename Tds::Edge           Edge;
    typedef typename Tds::Face_iterator  Face_iterator;
    typedef typename Tds::Face_handle    Face_handle;
    typedef Triangulation_ds_edge_iterator_2<Tds> Edge_iterator;
  private:
    const Tds* _tds;
    Face_iterator pos;
    mutable Edge edge;
  public:
    typedef Edge            value_type;
    typedef Edge*           pointer;
    typedef Edge&           reference;
    Triangulation_ds_edge_iterator_2() {};
    bool operator==(const Edge_iterator& fi) const { return false; }
    bool operator!=(const Edge_iterator& fi) const { return false; }
    Edge_iterator& operator++() { return *this; }
    Edge_iterator& operator--() { return *this; }
    Edge_iterator operator++(int) {  
      Edge_iterator tmp(*this);
      ++(*this);
      return tmp;
    }
    Edge_iterator operator--(int) {
      Edge_iterator tmp(*this);
      --(*this);
      return tmp;
    }
    Edge* operator->() const {
      edge.first = pos;
      return &edge;
    }
    Edge& operator*() const {
      edge.first = pos;
      return edge;
    }
  };

  template < class Tds>
  class Triangulation_ds_face_circulator_2 
  {
  public:
    typedef Triangulation_ds_face_circulator_2<Tds> Face_circulator;
    typedef typename Tds::Face                      Face;
    typedef typename Tds::Vertex                    Vertex;
    typedef typename Tds::Face_handle               Face_handle;
    typedef typename Tds::Vertex_handle             Vertex_handle;
  private:
    Vertex_handle _v;
    Face_handle    pos;
  public:
    Triangulation_ds_face_circulator_2() 
      : _v(), pos() 
    {}
    Triangulation_ds_face_circulator_2(Vertex_handle v,
				       Face_handle f = Face_handle()) {
      _v = v;
      pos = f;
    }
    Face_circulator& operator=(const Face_circulator& other) { return *this; }
    Face_circulator& operator++() { return *this; }
    Face_circulator operator++(int) {
      Face_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    Face_circulator& operator--() { return *this; }
    Face_circulator operator--(int) {
      Face_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    operator Face_handle() const { return pos; }
    bool operator==(const Face_circulator &fc) const { return false; }
    bool operator!=(const Face_circulator &fc) const { return false; }
    bool operator==(const int i) const { return false; }
    bool operator==(const Face_handle &fh) const { return pos == fh; }
    bool operator!=(const Face_handle &fh) const { return pos != fh; }
    Face& operator*() const { return *pos; }
    Face* operator->() const { return &*pos; }
  };
  template < class Tds_ >
  bool operator==(typename Tds_::Face_handle fh,
		  Triangulation_ds_face_circulator_2<Tds_> fc) { return (fc==fh); }
  template < class Tds_ >
  bool operator!=(typename Tds_::Face_handle fh,
		  Triangulation_ds_face_circulator_2<Tds_> fc) { return (fc!=fh); }

  template < class Tds >
  class Triangulation_ds_vertex_circulator_2 {
  public:
    typedef Triangulation_ds_vertex_circulator_2<Tds> Vertex_circulator;
    typedef typename Tds::Face                      Face;
    typedef typename Tds::Vertex                    Vertex;
    typedef typename Tds::Face_handle               Face_handle;
    typedef typename Tds::Vertex_handle             Vertex_handle;
  private:
    Vertex_handle _v;
    Face_handle   pos;
    int _ri;
  public:
    Triangulation_ds_vertex_circulator_2()
      :  _v(), pos()
    {}
    Triangulation_ds_vertex_circulator_2(Vertex_handle v,
					 Face_handle f = Face_handle()) 
      : _v(v), pos(pos)
    {}

    Vertex_circulator& operator++() { return *this; }
    Vertex_circulator  operator++(int) {
      Vertex_circulator tmp(*this);
      ++(*this);
      return tmp;
    }
    Vertex_circulator& operator--() { return *this; }
    Vertex_circulator  operator--(int) {
      Vertex_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    bool operator==(const int i) const { return false; }
    bool operator==(const Vertex_circulator &vc) const { return false; }
    bool operator!=(const Vertex_circulator &vc) const { return false; }
    bool operator==(const Vertex_handle &vh) const { return false; }
    bool operator!=(const Vertex_handle &vh) const{ return false; }
    Vertex& operator*() const { return *(pos->vertex(_ri)); }
    Vertex* operator->() const { return &*(pos->vertex(_ri)); }
    operator Vertex_handle() const { return _v; }
  };

  template < class Tds_ >
  bool operator==(typename Tds_::Vertex_handle vh,
		  Triangulation_ds_vertex_circulator_2<Tds_> vc) { return (vc==vh); }

  template < class Tds_ >
  bool operator!=(typename Tds_::Vertex_handle vh,
		  Triangulation_ds_vertex_circulator_2<Tds_> vc) { return !(vc==vh); }

  template < class Tds >
  class Triangulation_ds_edge_circulator_2 {
  public:
    typedef Triangulation_ds_edge_circulator_2<Tds>  Edge_circulator;
    typedef typename Tds::Face                       Face;
    typedef typename Tds::Vertex                     Vertex;
    typedef typename Tds::Edge                       Edge;
    typedef typename Tds::Face_handle                Face_handle;
    typedef typename Tds::Vertex_handle              Vertex_handle;

  private:
    int _ri;
    Vertex_handle _v;
    Face_handle  pos;
    mutable Edge edge;

  public:
    Triangulation_ds_edge_circulator_2()
      : _ri(0), _v(), pos()
    {}
    Triangulation_ds_edge_circulator_2( Vertex_handle v,
					Face_handle f = Face_handle()) 
      : _ri(0), _v(v), pos(f)
    {}

    Edge_circulator& operator++() { return *this; }
    Edge_circulator operator++(int) {
      Edge_circulator tmp(*this);
      ++(*this);
      return tmp;
    }
    Edge_circulator& operator--() { return *this; }
    Edge_circulator operator--(int) {
      Edge_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    bool operator==(const int i) const { return false; }
    bool operator==(const Edge_circulator &vc) const { return false; }
    bool operator!=(const Edge_circulator &vc) const { return false; }
    bool is_empty() const { return false; }
    Edge*  operator->() const {
      edge.first=pos;
      edge.second= _ri;
      return &edge;
    }
    Edge& operator*() const {
      edge.first=pos;
      edge.second= _ri;
      return edge;
    }
  };

  template < typename Info_, typename GT>
  class Triangulation_vertex_base_with_info_2 {
    Info_ _info;
  public:
    typedef Info_ Info;
  };

  template < class Vb >
  class Triangulation_data_structure_2 {
  public:
    typedef Triangulation_data_structure_2<Vb>         Tds;
    typedef typename Vb::Info Info;

    typedef Vertex_base<Tds,Info>                 Vertex;
    typedef Face_base<Tds>                        Face;

    friend class Triangulation_ds_edge_iterator_2<Tds>;
    friend class Triangulation_ds_face_circulator_2<Tds>;
    friend class Triangulation_ds_edge_circulator_2<Tds>;
    friend class Triangulation_ds_vertex_circulator_2<Tds>;

    typedef Triangulation_ds_edge_iterator_2<Tds>      Edge_iterator;
    typedef Triangulation_ds_face_circulator_2<Tds>    Face_circulator;
    typedef Triangulation_ds_vertex_circulator_2<Tds>  Vertex_circulator;
    typedef Triangulation_ds_edge_circulator_2<Tds>    Edge_circulator;
    
    typedef Container<Vertex>                          Vertex_range;
    typedef Container<Face>                            Face_range;
    typedef typename Face_range::iterator              Face_iterator;
    typedef typename Vertex_range::iterator            Vertex_iterator;
    typedef Vertex_iterator                            Vertex_handle;
    typedef Face_iterator                              Face_handle;
    typedef std::pair<Face_handle, int>                Edge;
    typedef std::list<Edge>                            List_edges;

  protected:
    Face_range   _faces;
    Vertex_range _vertices;

  public:
    Triangulation_data_structure_2() {};
    Face_range& faces() { return _faces; }
    Face_range& faces() const { return const_cast<Tds*>(this)->_faces; }
    Vertex_range& vertices() { return _vertices; }
    Vertex_range& vertices() const { return const_cast<Tds*>(this)->_vertices; }

    int dimension() const { return 0; }
    int number_of_vertices() const { return 0; }
    int number_of_faces() const { return 0; }
    int number_of_edges() const { return 0; }
    int number_of_full_dim_faces() const { return 0; }
    
    bool is_vertex(Vertex_handle v) const { return false; }
    bool is_edge(Face_handle fh, int i) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb,
		 Face_handle& fr,  int& i) const { return false; }
    bool is_face(Face_handle fh) const { return false; }
    bool is_face(Vertex_handle v1,
		 Vertex_handle v2,
		 Vertex_handle v3) const { return false; }
    bool is_face(Vertex_handle v1,
		 Vertex_handle v2,
		 Vertex_handle v3,
		 Face_handle& fr) const { return false; }

    Face_iterator faces_begin() const { return faces_end(); }
    Face_iterator faces_end() const { return faces().end(); }
    Vertex_iterator vertices_begin() const { return vertices_end(); }
    Vertex_iterator vertices_end() const { return vertices().end(); }
    Edge_iterator edges_begin() const { return Edge_iterator(); }
    Edge_iterator edges_end() const { return Edge_iterator(); }
    Face_iterator face_iterator_base_begin() const { return faces().end(); }
    Face_iterator face_iterator_base_end() const { return faces().end(); }

    Face_circulator incident_faces(Vertex_handle v,
				   Face_handle f =  Face_handle()) const { return Face_circulator(); }
    Vertex_circulator incident_vertices(Vertex_handle v,
					Face_handle f = Face_handle()) const { return Vertex_circulator(); }
    Edge_circulator incident_edges(Vertex_handle v,
				   Face_handle f = Face_handle()) const { return Edge_circulator(); }

    Vertex_handle mirror_vertex(Face_handle f, int i) const { return Vertex_handle(); }
    int mirror_index(Face_handle f, int i) const { return 0; }
    Edge mirror_edge(const Edge e) const { return Edge(); }

    void flip(Face_handle f, int i) {};
    Vertex_handle insert_first() { return Vertex_handle(); }
    Vertex_handle insert_second() { return Vertex_handle(); }
    Vertex_handle insert_in_face(Face_handle f) { return Vertex_handle(); }
    Vertex_handle insert_in_edge(Face_handle f, int i) { return Vertex_handle(); }
    Vertex_handle insert_dim_up(Vertex_handle w = Vertex_handle(),
				bool orient=true) { return Vertex_handle(); }
    void remove_degree_3(Vertex_handle v, Face_handle f = Face_handle()) {};
    void remove_1D(Vertex_handle v) {};
    void remove_second(Vertex_handle v) {};
    void remove_first(Vertex_handle v) {};
    void remove_dim_down(Vertex_handle v) {};
    void dim_down(Face_handle f, int i) {};
    Vertex_handle star_hole(List_edges& hole) { return Vertex_handle(); }
    void star_hole(Vertex_handle v, List_edges& hole) {};
    void make_hole(Vertex_handle v, List_edges& hole) {};
    
    Vertex_handle create_vertex(const Vertex &v = Vertex()) { return Vertex_handle(); }
    Vertex_handle create_vertex(Vertex_handle v) { return Vertex_handle(); }
    Face_handle create_face(const Face& f = Face()) { return Face_handle(); }
    Face_handle create_face(Face_handle f) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2,
			    Face_handle f3, int i3) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1, Vertex_handle v) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3,
			    Face_handle f1,
			    Face_handle f2,
			    Face_handle f3) { return Face_handle(); }
    void set_adjacency(Face_handle f0, int i0, Face_handle f1, int i1) const {};
    void delete_face(Face_handle) {};
    void delete_vertex(Vertex_handle) {};

    void clear() {};
    void set_dimension (int n) {};

    template < class EdgeIt >
    Vertex_handle star_hole(EdgeIt edge_begin, EdgeIt edge_end) { return Vertex_handle(); }
    template < class EdgeIt >
    void star_hole(Vertex_handle v, EdgeIt edge_begin, EdgeIt edge_end) {};
    template < class EdgeIt, class FaceIt >
    Vertex_handle star_hole(EdgeIt edge_begin,
			    EdgeIt edge_end,
			    FaceIt face_begin,
			    FaceIt face_end) { return Vertex_handle(); }
    template < class EdgeIt, class FaceIt >
    void  star_hole(Vertex_handle newv,
		    EdgeIt edge_begin,
		    EdgeIt edge_end,
		    FaceIt face_begin,
		    FaceIt face_end) {};
    

  };

  template < class Gt, class Tds >
  class Delaunay_triangulation_2 {
  public:
    typedef typename Tds::Vertex                 Vertex;
    typedef typename Tds::Face                   Face;
    typedef typename Tds::Edge                   Edge;
    typedef typename Tds::Vertex_handle          Vertex_handle;
    typedef typename Tds::Face_handle            Face_handle;

    typedef typename Tds::Face_circulator        Face_circulator;
    typedef typename Tds::Vertex_circulator      Vertex_circulator;
    typedef typename Tds::Edge_circulator        Edge_circulator;

    typedef typename Tds::Face_iterator          All_faces_iterator;
    typedef typename Tds::Edge_iterator          All_edges_iterator;
    typedef typename Tds::Vertex_iterator        All_vertices_iterator;

    typedef Filter_iterator<All_faces_iterator>  Finite_faces_iterator;
    typedef Filter_iterator<All_edges_iterator>  Finite_edges_iterator;
    typedef Filter_iterator<All_vertices_iterator>  Finite_vertices_iterator;

    typedef Finite_faces_iterator                Face_iterator;
    typedef Finite_edges_iterator                Edge_iterator;
    typedef Finite_vertices_iterator             Vertex_iterator;

    enum Locate_type {VERTEX=0,
		      EDGE,
		      FACE,
		      OUTSIDE_CONVEX_HULL,
		      OUTSIDE_AFFINE_HULL};

    typedef typename Tds::Info Info;
    typedef typename Gt::Point Point;
    typedef typename Gt::Triangle Triangle;
    typedef typename Gt::Segment Segment;

  protected:
    Gt  _gt;
    Tds _tds;
    Vertex_handle _infinite_vertex;

  public:
    Delaunay_triangulation_2() {};
    void clear() {};
    const Tds & tds() const { return _tds;}
    Tds & tds() { return _tds;}
    int dimension() const { return 0; }
    int number_of_vertices() const { return 0; }
    int number_of_faces() const { return 0; }
    Vertex_handle infinite_vertex() const { return _infinite_vertex; }
    Vertex_handle finite_vertex() const { return Vertex_handle(); }
    Face_handle infinite_face() const { return Face_handle(); }
    void set_infinite_vertex(const Vertex_handle& v) {};

    bool is_valid() const { return false; }

    bool is_infinite(Face_handle f) const { return false; }
    bool is_infinite(Vertex_handle v) const { return false; }
    bool is_infinite(Face_handle f, int i) const { return false; }
    bool is_infinite(const Edge& e) const { return false; }
    bool is_infinite(const Edge_circulator& ec) const { return false; }
    bool is_infinite(const All_edges_iterator& ei) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb, Face_handle& fr,
		 int & i) const { return false; }
    bool includes_edge(Vertex_handle va, Vertex_handle vb,
		       Vertex_handle& vbb,
		       Face_handle& fr, int & i) const { return false; }
    bool is_face(Vertex_handle v1, Vertex_handle v2, Vertex_handle v3) const { return false; }
    bool is_face(Vertex_handle v1, Vertex_handle v2, Vertex_handle v3,
		 Face_handle &fr) const { return false; }

    Triangle triangle(Face_handle f) const { return Triangle(); }
    Segment segment(Face_handle f, int i) const { return Segment(); }
    Segment segment(const Edge& e) const { return Segment(); }
    Segment segment(const Edge_circulator& ec) const { return Segment(); }
    // Segment segment(const All_edges_iterator& ei) const { return Segment(); }
    Segment segment(const Finite_edges_iterator& ei) const { return Segment(); }
    Point circumcenter(Face_handle f) const { return Point(); }
    Point circumcenter(const Point& p0,
		       const Point& p1,
		       const Point& p2) const { return Point(); }

    void flip(Face_handle f, int i) {};
    Vertex_handle insert_first(const Point& p) { return Vertex_handle(); }
    Vertex_handle insert_second(const Point& p){ return Vertex_handle(); }
    Vertex_handle insert_in_edge(const Point& p, Face_handle f,int i){ return Vertex_handle(); }
    Vertex_handle insert_in_face(const Point& p, Face_handle f){ return Vertex_handle(); }
    Vertex_handle insert_outside_convex_hull(const Point& p, Face_handle f){ return Vertex_handle(); }
    Vertex_handle insert_outside_affine_hull(const Point& p){ return Vertex_handle(); }
    Vertex_handle insert(const Point &p, Face_handle start = Face_handle() ){ return Vertex_handle(); }
    Vertex_handle insert(const Point& p,
			 Locate_type lt,
			 Face_handle loc, int li ){ return Vertex_handle(); }
    Vertex_handle push_back(const Point& a) { return Vertex_handle(); }
  
    void remove_degree_3(Vertex_handle  v, Face_handle f = Face_handle()) {};
    void remove_first(Vertex_handle  v) {};
    void remove_second(Vertex_handle v) {};
    void remove(Vertex_handle  v) {};

    Vertex_handle move_if_no_collision(Vertex_handle v, const Point &p) { return Vertex_handle(); }
    Vertex_handle move(Vertex_handle v, const Point &p) { return Vertex_handle(); }

    Face_handle locate(const Point& p,
		       Locate_type& lt,
		       int& li,
		       Face_handle start = Face_handle()) const { return Face_handle(); }
    Face_handle locate(const Point &p,
		       Face_handle start = Face_handle()) const { return Face_handle(); }

    Finite_faces_iterator finite_faces_begin() const { return _tds.faces_begin(); }
    Finite_faces_iterator finite_faces_end() const { return _tds.faces_end(); }
    Finite_vertices_iterator finite_vertices_begin() const { return _tds.vertices_begin(); }
    Finite_vertices_iterator finite_vertices_end() const { return _tds.vertices_end(); }
    Finite_edges_iterator finite_edges_begin() const { return _tds.edges_begin(); }
    Finite_edges_iterator finite_edges_end() const { return _tds.edges_end(); }
    // Point_iterator points_begin() const { return Point_iterator(); }
    // Point_iterator points_end() const { return Point_iterator(); }
  
    All_faces_iterator all_faces_begin() const { return _tds.faces_begin(); }
    All_faces_iterator all_faces_end() const { return _tds.faces_end(); }
    All_vertices_iterator all_vertices_begin() const { return _tds.vertices_begin(); }
    All_vertices_iterator all_vertices_end() const { return _tds.vertices_end(); }
    All_edges_iterator all_edges_begin() const { return _tds.edges_begin(); }
    All_edges_iterator all_edges_end() const { return _tds.edges_end(); }
  
    Face_iterator faces_begin() const {return finite_faces_begin();}
    Face_iterator faces_end() const {return finite_faces_end();}
    Edge_iterator edges_begin() const {return finite_edges_begin();}
    Edge_iterator edges_end() const {return finite_edges_end();}
    Vertex_iterator vertices_begin() const {return finite_vertices_begin();}
    Vertex_iterator vertices_end() const {return finite_vertices_end();}
  
    Face_circulator incident_faces(Vertex_handle v,
				   Face_handle f =  Face_handle()) const { return Face_circulator(); }
    Vertex_circulator incident_vertices(Vertex_handle v,
					Face_handle f = Face_handle()) const { return Vertex_circulator(); }
    Edge_circulator incident_edges(Vertex_handle v,
				   Face_handle f = Face_handle()) const { return Edge_circulator(); }

    Vertex_handle mirror_vertex(Face_handle f, int i) const { return Vertex_handle(); }
    int mirror_index(Face_handle f, int i) const { return 0; }
    Edge mirror_edge(const Edge e) const { return Edge(); }

    Oriented_side
    oriented_side(const Point &p0, const Point &p1,
		  const Point &p2, const Point &p) const { return ON_ORIENTED_BOUNDARY; }
    Bounded_side
    bounded_side(const Point &p0, const Point &p1,
		 const Point &p2, const Point &p) const { return ON_BOUNDARY; }
    Oriented_side
    oriented_side(Face_handle f, const Point &p) const { return ON_ORIENTED_BOUNDARY; }
    Oriented_side
    side_of_oriented_circle(const Point &p0, const Point &p1, const Point &p2,
			    const Point &p, bool perturb) const { return ON_ORIENTED_BOUNDARY; }
    Oriented_side
    side_of_oriented_circle(Face_handle f, const Point & p, bool perturb = false) const { return ON_ORIENTED_BOUNDARY; }

    void make_hole(Vertex_handle v, std::list<Edge> & hole) {};
    Face_handle create_face(const Face& f = Face()) { return Face_handle(); }
    Face_handle create_face(Face_handle f) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2,
			    Face_handle f3, int i3) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1, Vertex_handle v) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3,
			    Face_handle f1,
			    Face_handle f2,
			    Face_handle f3) { return Face_handle(); }
    Face_handle create_face() { return Face_handle(); }
    void delete_face(Face_handle f) {};
    void delete_vertex(Vertex_handle v) {};

    template < class InputIterator >
    std::ptrdiff_t insert(InputIterator first, InputIterator last) { return 0; }
    // typedef typename std::vector< std::pair<Point,Info> >::iterator pinsert;
    // void insert(pinsert p1, pinsert p2) {};

    template<class EdgeIt>
    Vertex_handle star_hole( const Point& p,
			     EdgeIt edge_begin,
			     EdgeIt edge_end) { return Vertex_handle(); }
    template<class EdgeIt, class FaceIt>
    Vertex_handle star_hole( const Point& p,
			     EdgeIt edge_begin,
			     EdgeIt edge_end,
			     FaceIt face_begin,
			     FaceIt face_end) { return Vertex_handle(); }


    // Delaunay stuff
    int ccw(int i) const { return i; }
    int cw(int i) const { return i; }
    Point dual (Face_handle f) const { return Point(); }
    // Point circumcenter (Face_handle f) const { return Point(); }
    Vertex_handle nearest_vertex(Point p, Face_handle f = Face_handle()) const { return Vertex_handle(); }
    void flip_flippable(Face_handle x, int i) {}

    template <class OutputItFaces, class OutputItBoundaryEdges>
    std::pair<OutputItFaces,OutputItBoundaryEdges>
    get_conflicts_and_boundary(const Point  &p,
			       OutputItFaces fit,
			       OutputItBoundaryEdges eit,
			       Face_handle start = Face_handle(),
			       bool strict = true) const { return std::make_pair(fit, eit); }
    template <class OutputItFaces>
    OutputItFaces get_conflicts (const Point  &p,
				 OutputItFaces fit,
				 Face_handle start= Face_handle(),
				 bool strict = true) const { return fit; }
    template <class OutputItBoundaryEdges>
    OutputItBoundaryEdges
    get_boundary_of_conflicts(const Point  &p,
			      OutputItBoundaryEdges eit,
			      Face_handle start= Face_handle(),
			      bool strict = true) const { return eit; }
  };

  template <class Key_, class Data_>
  class Unique_hash_map {
  private:
    Data_ _data;
  public:
    typedef Key_                                     Key;
    typedef Data_                                    Data;
    Unique_hash_map() 
      : _data(0)
    {};
    const Data& operator[]( const Key& key) const { return _data; }
    Data& operator[]( const Key& key) { return _data; }
  };

  template <class K>
  // double squared_distance(K::Point& p1, K::Point& p2) {
  double squared_distance(K& p1, K& p2) {
  // double squared_distance(typename K::Point& p1, typename K::Point& p2) {
    double x = 0.0;
    return x;
  }
};
