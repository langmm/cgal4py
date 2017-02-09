#include "mpi.h"
#include <math.h>
#include <stdint.h>
#include <vector>
#include <set>
#include <map>
#include <stdexcept>
#include <exception>
#include <iostream>
#include <fstream>
// #include "c_kdtree.hpp"
#include "c_parallel_kdtree.hpp"
#include "c_tools.hpp"
#include "c_delaunay2.hpp"
#include "c_delaunay3.hpp"
#include "c_periodic_delaunay2.hpp"
#include "c_periodic_delaunay3.hpp"
#include "c_delaunayD.hpp"
#if (CGAL_VERSION_NR < 1040900900)
#define VALID_PERIODIC_2 0
#else
#define VALID_PERIODIC_2 1
#endif
#if (CGAL_VERSION_NR < 1030501000)
#define VALID_PERIODIC_3 0
#else
#define VALID_PERIODIC_3 1
#endif
#define VALID 1
#define MAXLEN_FILENAME  1000
#define DEBUG 1

void my_error(const char* msg = "") {
  printf(msg);
  // throw std::runtime_error(msg);
}

void *my_malloc(size_t size) {
  void *out = malloc(size);
  if (out == NULL)
    my_error("Failed to malloc\n");
  return out;
}

void *my_realloc(void *in, size_t size, const char* msg = "") {
  void *out = realloc(in, size);
  if (out == NULL) {
    char msg_tot[100];
    std::strcpy(msg_tot, "Failed to realloc: ");
    std::strcat(msg_tot, msg);
    std::strcat(msg_tot, "\n");
    my_error( msg_tot );
  }
  return out;
}



void print_array_double(double *arr, int nrow, int ncol) {
  int i, j;
  printf("[\n");
  for (i = 0; i < nrow; i++) {
    printf(" [");
    for (j = 0; j < ncol; j++) {
      printf("%f ", arr[ncol*i+j]);
    }
    printf("]\n");
  }
  printf("]\n");
}

template <typename I>
void print_array_I(I *arr, int nrow, int ncol) {
  int i, j;
  printf("[\n");
  for (i = 0; i < nrow; i++) {
    printf(" [");
    for (j = 0; j < ncol; j++) {
      printf("%lu ", (uint64_t)arr[ncol*i+j]);
    }
    printf("]\n");
  }
  printf("]\n");
}


template <typename Info_>
class CGeneralDelaunay
{
public:
  typedef Info_ Info;
  typedef Delaunay_with_info_2<Info> Delaunay2;
  typedef Delaunay_with_info_3<Info> Delaunay3;
  typedef PeriodicDelaunay_with_info_2<Info> PeriodicDelaunay2;
  typedef PeriodicDelaunay_with_info_3<Info> PeriodicDelaunay3;
  typedef Delaunay_with_info_D<Info> DelaunayD;
  int ndim = 0;
  bool periodic;
  void *T;

  CGeneralDelaunay(int ndim0, bool periodic0 = false,
		   const double *domain = NULL) {
    ndim = ndim0;
    periodic = periodic0;
    if (ndim == 2) {
      if (periodic)
	T = (void*)(new PeriodicDelaunay2(domain));
      else
	T = (void*)(new Delaunay2());
    } else if (ndim == 3) {
      if (periodic)
	T = (void*)(new PeriodicDelaunay3(domain));
      else
	T = (void*)(new Delaunay3());
    } else if (ndim == D) {
      T = (void*)(new DelaunayD());
    } else {
      char msg[100];
      sprintf(msg, "[CGeneralDelaunay] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
  }
  ~CGeneralDelaunay() {
    if (ndim == 2) {
      if (periodic)
	delete((PeriodicDelaunay2*)T);
      else
	delete((Delaunay2*)T);
    } else if (ndim == 3) {
      if (periodic)
	delete((PeriodicDelaunay3*)T);
      else
	delete((Delaunay3*)T);
    } else if (ndim == D) {
      delete((DelaunayD*)T);
    } else {
      char msg[100];
      sprintf(msg, "[~CGeneralDelaunay] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
  }

  uint32_t num_finite_verts() {
    uint32_t out = 0;
    if (ndim == 2) {
      if (periodic)
	out = ((PeriodicDelaunay2*)T)->num_finite_verts();
      else
	out = ((Delaunay2*)T)->num_finite_verts();
    } else if (ndim == 3) {
      if (periodic)
	out = ((PeriodicDelaunay3*)T)->num_finite_verts();
      else
	out = ((Delaunay3*)T)->num_finite_verts();
    } else if (ndim == D) {
      out = ((DelaunayD*)T)->num_finite_verts();
    } else {
      char msg[100];
      sprintf(msg, "[num_finite_verts] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
    return out;
  }
  uint32_t num_cells() {
    uint32_t out = 0;
    if (ndim == 2) {
      if (periodic)
	out = ((PeriodicDelaunay2*)T)->num_cells();
      else
	out = ((Delaunay2*)T)->num_cells();
    } else if (ndim == 3) {
      if (periodic)
	out = ((PeriodicDelaunay3*)T)->num_cells();
      else
	out = ((Delaunay3*)T)->num_cells();
    } else if (ndim == D) {
      out = ((DelaunayD*)T)->num_cells();
    } else {
      char msg[100];
      sprintf(msg, "[num_cells] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
    return out;
  }

  void insert(double *pts, Info *val, uint32_t n) {
    if (ndim == 2) {
      if (periodic)
	((PeriodicDelaunay2*)T)->insert(pts, val, n);
      else
	((Delaunay2*)T)->insert(pts, val, n);
    } else if (ndim == 3) {
      if (periodic)
	((PeriodicDelaunay3*)T)->insert(pts, val, n);
      else
	((Delaunay3*)T)->insert(pts, val, n);
    } else if (ndim == D) {
      ((DelaunayD*)T)->insert(pts, val, n);
    } else {
      char msg[100];
      sprintf(msg, "[insert] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
  }

  template <typename I>
  I serialize_info2idx(I &n, I &m, int32_t &d,
                       I* cells, I* neighbors,
                       Info max_info, I* idx,
		       int32_t* offsets = NULL,
                       double* domain = NULL,
		       int32_t* cover = NULL) const {
    I out = 0;
    if (ndim == 2) {
      if (periodic)
	out = ((PeriodicDelaunay2*)T)->serialize_info2idx(n, m, d, domain, cover,
							  cells, neighbors,
							  offsets, max_info, idx);
      else
	out = ((Delaunay2*)T)->serialize_info2idx(n, m, d, cells, neighbors,
						  max_info, idx);
    } else if (ndim == 3) {
      if (periodic)
	out = ((PeriodicDelaunay3*)T)->serialize_info2idx(n, m, d, domain, cover,
							  cells, neighbors,
							  offsets, max_info, idx);
      else
	out = ((Delaunay3*)T)->serialize_info2idx(n, m, d, cells, neighbors,
						  max_info, idx);
    } else if (ndim == D) {
      out = ((DelaunayD*)T)->serialize_info2idx(n, m, d, cells, neighbors,
						max_info, idx);
    } else {
      char msg[100];
      sprintf(msg, "[serialize_info2idx] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
    return out;
  }

  template <typename I>
  void deserialize(I n, I m, int32_t d,
                   double* vert_pos, Info* vert_info,
                   I* cells, I* neighbors, I idx_inf,
		   double* domain = NULL, int32_t* cover = NULL,
		   int32_t* offsets = NULL) {
    if (ndim == 2) {
      if (periodic)
	((PeriodicDelaunay2*)T)->deserialize(n, m, d, domain, cover,
					     vert_pos, vert_info,
					     cells, neighbors,
					     offsets, idx_inf);
      else
	((Delaunay2*)T)->deserialize(n, m, d, vert_pos, vert_info,
				     cells, neighbors, idx_inf);
    } else if (ndim == 3) {
      if (periodic)
	((PeriodicDelaunay3*)T)->deserialize(n, m, d, domain, cover,
					     vert_pos, vert_info,
					     cells, neighbors,
					     offsets, idx_inf);
      else
	((Delaunay3*)T)->deserialize(n, m, d, vert_pos, vert_info,
				     cells, neighbors, idx_inf);
    } else if (ndim == D) {
      ((DelaunayD*)T)->deserialize(n, m, d, vert_pos, vert_info,
				   cells, neighbors, idx_inf);
    } else {
      char msg[100];
      sprintf(msg, "[deserialize] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
  }

  std::vector<std::vector<Info>> outgoing_points(uint64_t nbox,
                                                 double *left_edges,
						 double *right_edges) const {
    std::vector<std::vector<Info>> out;
    if (ndim == 2) {
      if (periodic)
	out = ((PeriodicDelaunay2*)T)->outgoing_points(nbox, left_edges, right_edges);
      else
	out = ((Delaunay2*)T)->outgoing_points(nbox, left_edges, right_edges);
    } else if (ndim == 3) {
      if (periodic)
	out = ((PeriodicDelaunay3*)T)->outgoing_points(nbox, left_edges, right_edges);
      else
	out = ((Delaunay3*)T)->outgoing_points(nbox, left_edges, right_edges);
    } else if (ndim == D) {
      out = ((DelaunayD*)T)->outgoing_points(nbox, left_edges, right_edges);
    } else {
      char msg[100];
      sprintf(msg, "[outgoing_points] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
    return out;
  }

  void write_to_buffer(std::ofstream &os) {
    if (ndim == 2) {
      if (periodic)
	((PeriodicDelaunay2*)T)->write_to_buffer(os);
      else
	((Delaunay2*)T)->write_to_buffer(os);
    } else if (ndim == 3) {
      if (periodic)
	((PeriodicDelaunay3*)T)->write_to_buffer(os);
      else
	((Delaunay3*)T)->write_to_buffer(os);
    } else if (ndim == D) {
      ((DelaunayD*)T)->write_to_buffer(os);
    } else {
      char msg[100];
      sprintf(msg, "[write_to_buffer] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
  }

  void read_from_buffer(std::ifstream &os) {
    if (ndim == 2) {
      if (periodic)
	((PeriodicDelaunay2*)T)->read_from_buffer(os);
      else
	((Delaunay2*)T)->read_from_buffer(os);
    } else if (ndim == 3) {
      if (periodic)
	((PeriodicDelaunay3*)T)->read_from_buffer(os);
      else
	((Delaunay3*)T)->read_from_buffer(os);
    } else if (ndim == D) {
      ((DelaunayD*)T)->read_from_buffer(os);
    } else {
      char msg[100];
      sprintf(msg, "[read_from_buffer] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
  }

  void dual_volumes(double *vols) {
    if (ndim == 2) {
      if (periodic)
	((PeriodicDelaunay2*)T)->dual_areas(vols);
      else
	((Delaunay2*)T)->dual_areas(vols);
    } else if (ndim == 3) {
      if (periodic)
	((PeriodicDelaunay3*)T)->dual_volumes(vols);
      else
	((Delaunay3*)T)->dual_volumes(vols);
    } else if (ndim == D) {
      ((DelaunayD*)T)->dual_volumes(vols);
    } else {
      char msg[100];
      sprintf(msg, "[dual_volumes] Incorrect number of dimensions. %d", ndim);
      my_error(msg);
    }
  }

};


template <typename Info_>
class CParallelLeaf
{
public:
  typedef Info_ Info;
  typedef CGeneralDelaunay<Info> Delaunay;
  bool from_node;
  bool in_memory = false;
  bool tess_exists = false;
  int size;
  int rank;
  char unique_str[MAXLEN_FILENAME];
  uint32_t id;
  uint32_t nleaves;
  uint32_t ndim;
  uint64_t npts_orig = 0;
  uint64_t npts = 0;
  uint64_t ncells = 0;
  Info *idx = NULL;
  double *pts = NULL;
  double *le = NULL;
  double *re = NULL;
  int *periodic_le = NULL;
  int *periodic_re = NULL;
  double *domain_width = NULL;
  std::set<uint32_t> *neigh;
  double *leaves_le;
  double *leaves_re;
  Delaunay *T = NULL;
  std::set<uint32_t> *all_neigh;
  std::vector<std::set<uint32_t>> *lneigh;
  std::vector<std::set<uint32_t>> *rneigh;
  char OutputFile[MAXLEN_FILENAME];

  void begin_init(uint32_t nleaves0, uint32_t ndim0, const char *ustr) {
    uint32_t k;
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    nleaves = nleaves0;
    ndim = ndim0;
    std::strcpy(unique_str, ustr);
    le = (double*)my_malloc(ndim*sizeof(double));
    re = (double*)my_malloc(ndim*sizeof(double));
    periodic_le = (int*)my_malloc(ndim*sizeof(int));
    periodic_re = (int*)my_malloc(ndim*sizeof(int));
    domain_width = (double*)my_malloc(ndim*sizeof(double));
    neigh = new std::set<uint32_t>();
    leaves_le = (double*)my_malloc(nleaves*ndim*sizeof(double*));
    leaves_re = (double*)my_malloc(nleaves*ndim*sizeof(double*));
    all_neigh = new std::set<uint32_t>();
    lneigh = new std::vector<std::set<uint32_t>>();
    rneigh = new std::vector<std::set<uint32_t>>();
    for (k = 0; k < ndim; k++) {
      lneigh->push_back(std::set<uint32_t>());
      rneigh->push_back(std::set<uint32_t>());
    }
  }    

  void end_init() {
    sprintf(OutputFile, "%s_leafoutput%u.dat", unique_str, id);
    in_memory = true;
  }

  CParallelLeaf(uint32_t nleaves0, uint32_t ndim0, const char *ustr, int src) {
    from_node = false;
    begin_init(nleaves0, ndim0, ustr);
    // Receive leaf info from root process
    recv(src);
    if (DEBUG > 1)
      printf("%d: Initialized from transfer on %d\n", id, rank);
    end_init();
  };

  CParallelLeaf(uint32_t nleaves0, uint32_t ndim0, const char *ustr,
		KDTree* tree, int index) {
    from_node = true;
    begin_init(nleaves0, ndim0, ustr);
    // Transfer leaf information
    Node* node = tree->leaves[index];
    uint64_t j;
    uint32_t k;
    std::set<uint32_t>::iterator it;
    id = node->leafid;
    npts = node->children;
    idx = (Info*)my_malloc(npts*sizeof(Info));
    pts = (double*)my_malloc(ndim*npts*sizeof(double));
    memcpy(le, node->left_edge, ndim*sizeof(double));
    memcpy(re, node->right_edge, ndim*sizeof(double));
    memcpy(domain_width, tree->domain_width, ndim*sizeof(double));
    for (j = 0; j < npts; j++) {
      idx[j] = (Info)(tree->left_idx + node->left_idx + j);
      for (k = 0; k < ndim; k++) {
    	pts[ndim*j+k] = tree->all_pts[ndim*tree->all_idx[node->left_idx+j]+k];
      }
    }
    memcpy(leaves_le, tree->leaves_le, nleaves*ndim*sizeof(double));
    memcpy(leaves_re, tree->leaves_re, nleaves*ndim*sizeof(double));
    neigh->insert(node->all_neighbors.begin(), node->all_neighbors.end());
    for (k = 0; k < ndim; k++) {
      periodic_le[k] = node->periodic_left[k];
      periodic_re[k] = node->periodic_right[k];
    }
    for (k = 0; k < ndim; k++) {
      (*lneigh)[k].insert(node->left_neighbors[k].begin(),
    			  node->left_neighbors[k].end());
      (*rneigh)[k].insert(node->right_neighbors[k].begin(),
    			  node->right_neighbors[k].end());
    }
    // Shift edges of periodic neighbors
    for (k = 0; k < ndim; k++) {
      if (periodic_le[k]) {
	for (it = (*lneigh)[k].begin(); it != (*lneigh)[k].end(); it++) {
    	  leaves_le[*it, k] -= domain_width[k];
    	  leaves_re[*it, k] -= domain_width[k];
    	}
      }
      if (periodic_re[k]) {
	for (it = (*rneigh)[k].begin(); it != (*rneigh)[k].end(); it++) {
    	     
    	  leaves_le[*it, k] += domain_width[k];
    	  leaves_re[*it, k] += domain_width[k];
    	}
      }
    }
    if (DEBUG > 1)
      printf("%d: Initialized directly on %d\n", id, rank);
    end_init();
  }

  ~CParallelLeaf() {
    delete(neigh);
    free(leaves_le);
    free(leaves_re);
    delete(all_neigh);
    delete(lneigh);
    delete(rneigh);
    delete(T);
    if (pts != NULL)
      free(pts);
    if (idx != NULL)
      free(idx);
    free(periodic_le);
    free(periodic_re);
    free(le);
    free(re);
    free(domain_width);
  }

  void dump() {
    if (in_memory) { // Don't write empty pointers
      std::ofstream fd (OutputFile, std::ios::out | std::ios::binary);
      fd.write((char*)idx, npts*sizeof(Info));
      fd.write((char*)pts, npts*ndim*sizeof(double));
      free(idx);
      free(pts);
      if (tess_exists) {
	T->write_to_buffer(fd);
	delete(T);
	T = NULL;
      }
      idx = NULL;
      pts = NULL;
      fd.close();
      in_memory = false;
    }
  }
  void load() {
    if (!(in_memory)) { // Don't read if already loaded
      std::ifstream fd (OutputFile, std::ios::in | std::ios::binary);
      idx = (Info*)my_malloc(npts*sizeof(Info));
      pts = (double*)my_malloc(npts*ndim*sizeof(double));
      fd.read((char*)idx, npts*sizeof(Info));
      fd.read((char*)pts, npts*ndim*sizeof(double));
      if (tess_exists) {
	T = new Delaunay(ndim, false);
	T->read_from_buffer(fd);
      }
      fd.close();
      in_memory = true;
    }
  }

  uint32_t num_cells() {
    return T->num_cells();
  }

  void send(int dst) {
    int i = 0, j;
    uint32_t k = 0;
    MPI_Send(&id, 1, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    MPI_Send(&npts, 1, MPI_UNSIGNED_LONG, dst, i++, MPI_COMM_WORLD);
    if (sizeof(Info) == sizeof(uint32_t))
      MPI_Send(idx, npts, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    else
      MPI_Send(idx, npts, MPI_UNSIGNED_LONG, dst, i++, MPI_COMM_WORLD);
    MPI_Send(pts, ndim*npts, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(le, ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(re, ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(periodic_le, ndim, MPI_INT, dst, i++, MPI_COMM_WORLD);
    MPI_Send(periodic_re, ndim, MPI_INT, dst, i++, MPI_COMM_WORLD);
    MPI_Send(domain_width, ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(leaves_le, nleaves*ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(leaves_re, nleaves*ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    uint32_t *dummy = (uint32_t*)my_malloc(nleaves*sizeof(uint32_t));
    int ndum;
    // neighbors
    ndum = (int)(neigh->size());
    j = 0;
    for (std::set<uint32_t>::iterator it = neigh->begin();
	 it != neigh->end(); it++) {
      dummy[j] = *it;
      j++;
    }
    MPI_Send(&ndum, 1, MPI_INT, dst, i++, MPI_COMM_WORLD);
    MPI_Send(dummy, ndum, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    for (k = 0; k < ndim; k++) {
      // left neighbors
      ndum = (int)((*lneigh)[k].size());
      j = 0;
      for (std::set<uint32_t>::iterator it = (*lneigh)[k].begin();
    	   it != (*lneigh)[k].end(); it++) {
    	dummy[j] = *it;
    	j++;
      }
      MPI_Send(&ndum, 1, MPI_INT, dst, i++, MPI_COMM_WORLD);
      MPI_Send(dummy, ndum, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
      // right neighbors
      ndum = (int)((*rneigh)[k].size());
      j = 0;
      for (std::set<uint32_t>::iterator it = (*rneigh)[k].begin();
    	   it != (*rneigh)[k].end(); it++) {
    	dummy[j] = *it;
    	j++;
      }
      MPI_Send(&ndum, 1, MPI_INT, dst, i++, MPI_COMM_WORLD);
      MPI_Send(dummy, ndum, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    }
    free(dummy);
    if (DEBUG > 1)
      printf("%d: Sent to %d from %d\n", id, dst, rank);
  };

  void recv(int src) {
    int i = 0, j;
    uint32_t k = 0;
    MPI_Recv(&id, 1, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(&npts, 1, MPI_UNSIGNED_LONG, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    idx = (Info*)my_malloc(npts*sizeof(Info));
    pts = (double*)my_malloc(ndim*npts*sizeof(double));
    if (sizeof(Info) == sizeof(uint32_t))
      MPI_Recv(idx, npts, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    else
      MPI_Recv(idx, npts, MPI_UNSIGNED_LONG, src, i++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    MPI_Recv(pts, ndim*npts, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(le, ndim, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(re, ndim, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(periodic_le, ndim, MPI_INT, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(periodic_re, ndim, MPI_INT, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(domain_width, ndim, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(leaves_le, nleaves*ndim, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(leaves_re, nleaves*ndim, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    uint32_t *dummy = (uint32_t*)my_malloc(nleaves*sizeof(uint32_t));
    int ndum;
    MPI_Recv(&ndum, 1, MPI_INT, src, i++, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    MPI_Recv(dummy, ndum, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    for (j = 0; j < ndum; j++) {
      neigh->insert(dummy[j]);
    }
    // neighbors
    for (k = 0; k < ndim; k++) {
      // left neighbors
      MPI_Recv(&ndum, 1, MPI_INT, src, i++, MPI_COMM_WORLD,
    	       MPI_STATUS_IGNORE);
      MPI_Recv(dummy, ndum, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
    	       MPI_STATUS_IGNORE);
      for (j = 0; j < ndum; j++) {
    	(*lneigh)[k].insert(dummy[j]);
      }
      // right neighbors
      MPI_Recv(&ndum, 1, MPI_INT, src, i++, MPI_COMM_WORLD,
    	       MPI_STATUS_IGNORE);
      MPI_Recv(dummy, ndum, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
    	       MPI_STATUS_IGNORE);
      for (j = 0; j < ndum; j++) {
    	(*rneigh)[k].insert(dummy[j]);
      }
    }
    free(dummy);
    if (DEBUG > 1)
      printf("%d: Received from %d on %d\n", id, src, rank);
  }

  void init_triangulation() {
    T = new Delaunay(ndim, false);
    // Insert points using monotonic indices
    Info *idx_dum = (Info*)my_malloc(npts*sizeof(Info));
    for (Info i = 0; i < npts; i++)
      idx_dum[i] = i;
    T->insert(pts, idx_dum, npts);
    free(idx_dum);
    npts_orig = npts;
    ncells = (uint64_t)(T->num_cells());
    if (DEBUG > 1)
      printf("%d: Triangulation of %lu points initialized on %d\n", id, npts, rank);
    tess_exists = true;
  }

  void insert(double *pts_new, Info *idx_new, uint64_t npts_new) {
    // Insert points
    Info *idx_dum = (Info*)my_malloc(npts_new*sizeof(Info));
    for (Info i = 0, j = npts; i < npts_new; i++, j++)
      idx_dum[i] = j;
    T->insert(pts_new, idx_dum, npts_new);
    free(idx_dum);
    // Copy indices
    idx = (Info*)my_realloc(idx, (npts+npts_new)*sizeof(Info), "idx in insert");
    memcpy(idx+npts, idx_new, npts_new*sizeof(Info));
    // Copy points
    pts = (double*)my_realloc(pts, ndim*(npts+npts_new)*sizeof(double),
			      "pts in insert");
    memcpy(pts+ndim*npts, pts_new, ndim*npts_new*sizeof(double));
    // Advance count
    npts += npts_new;
    ncells = (uint64_t)(T->num_cells());
    if (DEBUG > 1)
      printf("%d: %lu points inserted on %d\n", id, npts_new, rank);
  }
  
  template <typename I>
  I serialize(I &n, I &m,
	      I *cells, I *neigh,
	      uint32_t *idx_verts, uint64_t *idx_cells,
	      bool sort = false) {
    n = (Info)(T->num_finite_verts());
    m = (Info)(T->num_cells());
    int32_t d = ndim;
    I idx_inf = T->serialize_info2idx(n, m, d, cells, neigh,
				      (Info)npts_orig, idx);
    if (sort) {
      sortSerializedTess(cells, neigh, m, d+1);
    } else {
      uint64_t j;
      uint32_t k;
      for (j = 0; j < (uint64_t)m; j++) {
    	idx_cells[j] = j;
    	for (k = 0; k < (uint32_t)(d+1); k++)
    	  idx_verts[(d+1)*j+k] = k;
      }
      arg_sortSerializedTess(cells, m, d+1, idx_verts, idx_cells);
    }
    if (DEBUG > 1)
      printf("%d: %lu cells serialized on %d\n", id, (uint64_t)m, rank);
    return idx_inf;
  };

  void outgoing_points(std::vector<std::vector<uint32_t>> &src_out,
		       std::vector<std::vector<uint32_t>> &dst_out,
		       std::vector<std::vector<uint32_t>> &cnt_out,
		       std::vector<std::vector<uint32_t>> &nct_out,
		       std::vector<Info*> &idx_out,
		       std::vector<double*> &pts_out,
		       std::vector<uint32_t*> &ngh_out) {
    int i, j;
    uint32_t k, n, dst, src=id;
    int task;
    typedef typename std::vector<Info> vect_Info;
    std::vector<uint32_t>::iterator it32;
    typename vect_Info::iterator it;
    std::set<uint32_t>::iterator sit;
    std::vector<vect_Info> out_leaves;
    // Select edges of neighbors
    double *neigh_le = (double*)my_malloc(neigh->size()*ndim*sizeof(double*));
    double *neigh_re = (double*)my_malloc(neigh->size()*ndim*sizeof(double*));
    for (sit = neigh->begin(), i = 0; sit != neigh->end(); sit++, i++) {
      n = *sit;
      memcpy(neigh_le+ndim*i, leaves_le+ndim*n, ndim*sizeof(double));
      memcpy(neigh_re+ndim*i, leaves_re+ndim*n, ndim*sizeof(double));
    }
    // Get outgoing to other leaves
    out_leaves = T->outgoing_points(neigh->size(), neigh_le, neigh_re);
    // Sort leaves to their host task
    uint32_t ntot = 0;
    uint32_t nold, nnew, nold_neigh, nnew_neigh;
    for (sit = neigh->begin(), i = 0; sit != neigh->end(); sit++, i++) {
      dst = *sit;
      task = dst % size;
      src_out[task].push_back(src);
      dst_out[task].push_back(dst);
      for (it = out_leaves[i].begin(); it != out_leaves[i].end(); ) {
	if (*it < npts_orig)
	  it++;
	else
	  it = out_leaves[i].erase(it);
      }
      nnew = (uint32_t)(out_leaves[i].size());
      nold = 0;
      for (it32 = cnt_out[task].begin();
	   it32 != cnt_out[task].end(); it32++)
	nold += *it32;
      if (nnew > 0)
	nnew_neigh = neigh->size();
      else
	nnew_neigh = 0;
      nold_neigh = 0;
      for (it32 = nct_out[task].begin();
	   it32 != nct_out[task].end(); it32++)
	nold_neigh += *it32;
      // TODO: Maybe move realloc outside of loop
      if (nnew > 0) {
	char msg[100];
	sprintf(msg, "nold = %d, nnew = %d", nold, nnew);
	idx_out[task] = (Info*)my_realloc(idx_out[task],
					  (nold+nnew)*sizeof(Info),
					  msg);
	// "idx in leaf outgoing points");
	pts_out[task] = (double*)my_realloc(pts_out[task],
					    ndim*(nold+nnew)*sizeof(double),
					    "pts in leaf outgoing points");
	ngh_out[task] = (uint32_t*)my_realloc(ngh_out[task],
					      (nold_neigh+nnew_neigh)*sizeof(uint32_t),
					      "ngh in leaf outgoing points");
      }
      cnt_out[task].push_back(nnew);
      nct_out[task].push_back(nnew_neigh);
      for (it = out_leaves[i].begin(), j = nold;
	   it != out_leaves[i].end(); it++, j++) {
	idx_out[task][j] = idx[*it];
	for (k = 0; k < ndim; k++)
	  pts_out[task][ndim*j+k] = pts[ndim*(*it)+k];
      }
      ntot += nnew;
      if (nnew_neigh > 0) {
	std::set<uint32_t>::iterator sit2;
	for (sit2 = neigh->begin(), k = 0; sit2 != neigh->end(); sit2++, k++)
	  ngh_out[task][nold_neigh+k] = *sit2;
      }
    }
    // Transfer neighbors to log & reset count to 0
    all_neigh->insert(neigh->begin(), neigh->end());
    neigh->clear();
    if (DEBUG > 1)
      printf("%d: %lu outgoing points on %d\n", id, (uint64_t)ntot, rank);
  }

  void incoming_points(uint32_t src, uint32_t npts_recv,
		       uint32_t nneigh_recv, Info *idx_recv,
		       double *pts_recv, uint32_t *neigh_recv) {
    if (npts_recv == 0)
      return;
    uint64_t j;
    uint32_t k;
    if (src == id) {
      for (k = 0; k < ndim; k++) {
	if (periodic_le[k] and periodic_re[k]) {
	  for (j = 0; j < npts_recv; j++) {
	    if ((pts_recv[ndim*j+k] - le[k]) < (re[k] - pts_recv[ndim*j+k])) {
	      pts_recv[ndim*j+k] += domain_width[k];
	    }
	    if ((re[k] - pts_recv[ndim*j+k]) < (pts_recv[ndim*j+k] - le[k])) {
	      pts_recv[ndim*j+k] -= domain_width[k];
	    }
	  }
	}
      }
    } else {
      for (k = 0; k < ndim; k++) {
	if (periodic_re[k] and ((*rneigh)[k].count(src) > 0)) {
	  for (j = 0; j < npts_recv; j++) {
	    if ((pts_recv[ndim*j+k] + domain_width[k] - re[k]) <
		(le[k] - pts_recv[ndim*j+k]))
	      pts_recv[ndim*j+k] += domain_width[k];
	  }
	}
	if (periodic_le[k] and ((*lneigh)[k].count(src) > 0)) {
	  for (j = 0; j < npts_recv; j++) {
	    if ((le[k] - pts_recv[ndim*j+k] + domain_width[k]) <
		(pts_recv[ndim*j+k] - re[k]))
	      pts_recv[ndim*j+k] -= domain_width[k];
	  }
	}
      }
    }
    // Add points to tessellation, then arrays
    insert(pts_recv, idx_recv, npts_recv);
    // Add neighbors
    uint32_t n;
    for (k = 0; k < nneigh_recv; k++) {
      n = neigh_recv[k];
      if ((n != id) and (all_neigh->count(n) == 0) and (neigh->count(n) == 0)) {
	neigh->insert(n);
      }
    }
    if (DEBUG > 1)
      printf("%d: %lu incoming points on %d\n", id, (uint64_t)npts_recv, rank);
  }

  uint64_t voronoi_volumes(double **vols) {
    (*vols) = (double*)my_realloc(*vols, T->num_finite_verts()*sizeof(double),
				  "leaf voronoi volums");
    T->dual_volumes(*vols);
    return npts;
  }

};


template <typename Info_>
class ParallelDelaunay_with_info_D
{
public:
  typedef Info_ Info;
  int rank;
  int size;
  uint32_t ndim;
  int tree_exists = 0;
  int limit_mem = 0;
  char unique_str[MAXLEN_FILENAME];
  // Things only valid for root
  double *le;
  double *re;
  bool *periodic;
  uint64_t npts_prev = 0;
  uint64_t npts_total;
  int nleaves_total;
  double *pts_total = NULL;
  uint64_t *idx_total = NULL;
  Info *info_total = NULL;
  KDTree *tree = NULL;
  ParallelKDTree *ptree = NULL;
  // Things for each process
  int nleaves;
  std::vector<CParallelLeaf<Info>*> leaves;
  std::map<int,uint32_t> map_id2idx;

  ParallelDelaunay_with_info_D() {}
  ParallelDelaunay_with_info_D(uint32_t ndim0, double *le0, double *re0,
			       bool *periodic0, int limit_mem0 = 0,
			       const char* unique_str0 = "") {
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    if (DEBUG)
      printf("%d: Beginning init\n", rank);
    ndim = ndim0;
    le = le0;
    re = re0;
    periodic = periodic0;
    limit_mem = limit_mem0;
    std::strcpy(unique_str, unique_str0);
    MPI_Bcast(&ndim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&limit_mem, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&unique_str, MAXLEN_FILENAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (DEBUG)
      printf("%d: Finishing init\n", rank);
  }

  ~ParallelDelaunay_with_info_D() {
    if (DEBUG)
      printf("%d: Beginning dealloc\n", rank);
    int i;
    if (idx_total != NULL)
      free(idx_total);
    if (info_total != NULL)
      free(info_total);
    for (i = 0; i < nleaves; i++) {
      delete(leaves.pop()); // leaves used
    }
    if (tree != NULL)
      delete(tree);
    if (ptree != NULL)
      delete(ptree);
    if (DEBUG)
      printf("%d: Finishing dealloc\n", rank);
  }

  void insert(uint64_t npts0, double *pts0) {
    if (DEBUG)
      printf("%d: Beginning insert\n", rank);
    int i;
    uint64_t j;
    uint32_t k;
    if (tree_exists == 0) {
      // Initial domain decomposition
      npts_total = npts0;
      pts_total = pts0;
      domain_decomp();
      for (i = 0; i < nleaves; i++) {
	if (limit_mem > 1)
	  leaves[i]->load();
	leaves[i]->init_triangulation(); // leaves used
	if (limit_mem > 1)
	  leaves[i]->dump();
      }
    } else {
      Info *iidx = NULL;
      double *ipts = NULL;
      // Assign points to leaves based on initial domain decomp
      if (rank == 0) {
	// Assign each point to an existing leaf
      	std::vector<std::vector<uint64_t>> dist;
      	for (i = 0; i < nleaves_total; i++) {
      	  dist.push_back(std::vector<uint64_t>());
      	  dist[i].reserve(npts0/nleaves_total+1);
      	}
	int cnt = 0;
      	for (j = 0; j < npts0; j++) {
	  Node *res = tree->search(pts0+ndim*j);
	  if (res != NULL)
	    dist[res->leafid].push_back(j);
	  else
	    cnt++;
	}
	if (cnt > 0) {
	    printf("%d points were not within the bounds of the original domain decomposition\n",
		   cnt);
	}
	// Send new points to leaf
      	int nsend, task;
	int iroot = 0;
      	for (i = 0; i < nleaves_total; i++) {
      	  task = i % size;
      	  nsend = (int)(dist[i].size());
	  iidx = (Info*)my_realloc(iidx, nsend*sizeof(Info));
	  ipts = (double*)my_realloc(ipts, ndim*nsend*sizeof(double));
	  for (j = 0; j < (uint64_t)nsend; j++) {
	    iidx[j] = dist[i][j] + npts_prev;
	    for (k = 0; k < ndim; k++) 
	      ipts[ndim*j+k] = pts0[ndim*dist[i][j]+k];
	  }
      	  if (task == rank) {
	    if (limit_mem > 1)
	      leaves[iroot]->load();
	    leaves[iroot]->insert(ipts, iidx, nsend); // leaves used
	    if (limit_mem > 1)
	      leaves[iroot]->dump();
	    iroot++;
      	  } else {
      	    MPI_Send(&nsend, 1, MPI_INT, task, 20+task, MPI_COMM_WORLD);
	    if (sizeof(Info) == sizeof(uint32_t))
	      MPI_Send(iidx, nsend, MPI_UNSIGNED, task, 21+task,
		       MPI_COMM_WORLD);
	    else
	      MPI_Send(iidx, nsend, MPI_UNSIGNED_LONG, task, 21+task,
		       MPI_COMM_WORLD);
	    MPI_Send(ipts, ndim*nsend, MPI_DOUBLE, task, 22+task,
		     MPI_COMM_WORLD);
      	  }
	  free(iidx);
	  free(ipts);
      	}
      } else {
      	int nrecv;
      	for (i = 0; i < nleaves; i++) {
      	  MPI_Recv(&nrecv, 1, MPI_INT, 0, 20+rank, MPI_COMM_WORLD,
      		   MPI_STATUS_IGNORE);
	  iidx = (Info*)my_realloc(iidx,nrecv*sizeof(Info));
	  ipts = (double*)my_realloc(ipts,ndim*nrecv*sizeof(double));
	  if (sizeof(Info) == sizeof(uint32_t))
	    MPI_Recv(iidx, nrecv, MPI_UNSIGNED, 0, 21+rank,
		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  else
	    MPI_Recv(iidx, nrecv, MPI_UNSIGNED_LONG, 0, 21+rank,
		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(ipts, ndim*nrecv, MPI_DOUBLE, 0, 22+rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  if (limit_mem > 0)
	    leaves[i]->load();
	  leaves[i]->insert(ipts, iidx, nrecv); // leaves used
	  if (limit_mem > 0)
	    leaves[i]->dump();
      	}
      }
      if (iidx != NULL)
	free(iidx);
      if (ipts != NULL)
	free(ipts);
    }
    // Exchange points
    exchange();
    npts_prev += npts0;
    if (DEBUG)
      printf("%d: Finishing insert\n", rank);
  }

  void exchange() {
    if (DEBUG)
      printf("%d: Beginning exchange\n", rank);
    uint64_t nrecv_total = 1;
    uint64_t nrecv;
    int nexch;
    uint32_t *src_recv = NULL;
    uint32_t *dst_recv = NULL;
    uint32_t *cnt_recv = NULL;
    uint32_t *nct_recv = NULL;
    Info *idx_recv = NULL;
    double *pts_recv = NULL;
    uint32_t *ngh_recv = NULL;
    int count_exch = 0;
    while (nrecv_total != 0) {
      nexch = outgoing_points(&src_recv, &dst_recv, &cnt_recv, &nct_recv,
			      &idx_recv, &pts_recv, &ngh_recv);
      nrecv = incoming_points(nexch, src_recv, dst_recv, cnt_recv, nct_recv,
      			      idx_recv, pts_recv, ngh_recv);
      free(src_recv);
      free(dst_recv);
      free(cnt_recv);
      free(nct_recv);
      // free(idx_recv);  // Memory moved to leaf
      // free(pts_recv);  // Memory moved to leaf
      free(ngh_recv);
      MPI_Allreduce(&nrecv, &nrecv_total, 1, MPI_UNSIGNED, MPI_SUM,
		    MPI_COMM_WORLD);
      count_exch++;
    }
    if (DEBUG)
      printf("%d: Finishing exchange (%d rounds)\n", rank, count_exch-1);
  }

  uint64_t incoming_points(int nexch, uint32_t *src_recv, uint32_t *dst_recv,
			   uint32_t *cnt_recv, uint32_t *nct_recv,
			   Info *idx_recv, double *pts_recv,
			   uint32_t *ngh_recv) {
    if (DEBUG)
      printf("%d: Beginning incoming_points\n", rank);
    uint64_t nrecv = 0;
    uint64_t nprev_pts = 0, nprev_ngh = 0;
    Info *iidx;
    double *ipts;
    uint32_t *ingh;
    int dst;
    for (int i = 0; i < nexch; i++) {
      iidx = idx_recv + nprev_pts;
      ipts = pts_recv + ndim*nprev_pts;
      ingh = ngh_recv + nprev_ngh;
      dst = map_id2idx[dst_recv[i]];
      if (cnt_recv[i] > 0) {
	if (limit_mem > 1)
	  leaves[dst]->load();
	leaves[dst]->incoming_points(src_recv[i], cnt_recv[i], nct_recv[i],
				     iidx, ipts, ingh); // leaves used
	if (limit_mem > 1)
	  leaves[dst]->dump();
      }
      nprev_pts += cnt_recv[i];
      nprev_ngh += nct_recv[i];
    }
    nrecv = nprev_pts;
    if (DEBUG)
      printf("%d: Finishing incoming_points\n", rank);
    return nrecv;
  }

  int outgoing_points(uint32_t **src_recv, uint32_t **dst_recv,
		      uint32_t **cnt_recv, uint32_t **nct_recv,
		      Info **idx_recv, double **pts_recv,
		      uint32_t **ngh_recv) {
    if (DEBUG)
      printf("%d: Beginning outgoing_points\n", rank);
    int i, j;
    // Get output from each leaf
    std::vector<std::vector<uint32_t>> src_out, dst_out, cnt_out, nct_out;
    std::vector<Info*> idx_out;
    std::vector<double*> pts_out;
    std::vector<uint32_t*> ngh_out;
    for (i = 0; i < size; i++) {
      src_out.push_back(std::vector<uint32_t>());
      dst_out.push_back(std::vector<uint32_t>());
      cnt_out.push_back(std::vector<uint32_t>());
      nct_out.push_back(std::vector<uint32_t>());
      idx_out.push_back(NULL);
      pts_out.push_back(NULL);
      ngh_out.push_back(NULL);
    }
    for (i = 0; i < nleaves; i++) {
      if (limit_mem > 1)
	leaves[i]->load();
      leaves[i]->outgoing_points(src_out, dst_out, cnt_out, nct_out,
				 idx_out, pts_out, ngh_out); // leaves used
      if (limit_mem > 1)
	leaves[i]->dump();
    }
    // Send expected counts
    int *count_send = (int*)my_malloc(size*sizeof(int));
    int *count_recv = (int*)my_malloc(size*sizeof(int));
    int *offset_send = (int*)my_malloc(size*sizeof(int));
    int *offset_recv = (int*)my_malloc(size*sizeof(int));
    for (i = 0; i < size; i++)
      count_send[i] = (int)(src_out[i].size());
    MPI_Alltoall(count_send, 1, MPI_INT,
    		 count_recv, 1, MPI_INT,
    		 MPI_COMM_WORLD);
    int count_send_tot = 0, count_recv_tot = 0;
    for (i = 0; i < size; i++) {
      count_send_tot += count_send[i];
      count_recv_tot += count_recv[i];
    }
    // Send extra info about each exchange
    uint32_t *src_send = (uint32_t*)my_malloc(count_send_tot*sizeof(uint32_t));
    uint32_t *dst_send = (uint32_t*)my_malloc(count_send_tot*sizeof(uint32_t));
    uint32_t *cnt_send = (uint32_t*)my_malloc(count_send_tot*sizeof(uint32_t));
    uint32_t *nct_send = (uint32_t*)my_malloc(count_send_tot*sizeof(uint32_t));
    (*src_recv) = (uint32_t*)my_malloc(count_recv_tot*sizeof(uint32_t));
    (*dst_recv) = (uint32_t*)my_malloc(count_recv_tot*sizeof(uint32_t));
    (*cnt_recv) = (uint32_t*)my_malloc(count_recv_tot*sizeof(uint32_t));
    (*nct_recv) = (uint32_t*)my_malloc(count_recv_tot*sizeof(uint32_t));
    int prev_send = 0, prev_recv = 0;
    for (i = 0; i < size; i++) {
      offset_send[i] = prev_send;
      offset_recv[i] = prev_recv;
      for (j = 0; j < count_send[i]; j++) {
    	src_send[prev_send] = src_out[i][j];
    	dst_send[prev_send] = dst_out[i][j];
    	cnt_send[prev_send] = cnt_out[i][j];
    	nct_send[prev_send] = nct_out[i][j];
    	prev_send++;
      }
      prev_recv += count_recv[i];
    }
    MPI_Alltoallv(src_send, count_send, offset_send, MPI_UNSIGNED,
    		  *src_recv, count_recv, offset_recv, MPI_UNSIGNED,
    		  MPI_COMM_WORLD);
    MPI_Alltoallv(dst_send, count_send, offset_send, MPI_UNSIGNED,
    		  *dst_recv, count_recv, offset_recv, MPI_UNSIGNED,
    		  MPI_COMM_WORLD);
    MPI_Alltoallv(cnt_send, count_send, offset_send, MPI_UNSIGNED,
    		  *cnt_recv, count_recv, offset_recv, MPI_UNSIGNED,
    		  MPI_COMM_WORLD);
    MPI_Alltoallv(nct_send, count_send, offset_send, MPI_UNSIGNED,
    		  *nct_recv, count_recv, offset_recv, MPI_UNSIGNED,
    		  MPI_COMM_WORLD);
    free(src_send);
    free(dst_send);
    free(offset_send);
    free(offset_recv);
    // Get counts/offsets for sending arrays
    int *count_idx_send = (int*)my_malloc(size*sizeof(int));
    int *count_idx_recv = (int*)my_malloc(size*sizeof(int));
    int *count_pts_send = (int*)my_malloc(size*sizeof(int));
    int *count_pts_recv = (int*)my_malloc(size*sizeof(int));
    int *count_ngh_send = (int*)my_malloc(size*sizeof(int));
    int *count_ngh_recv = (int*)my_malloc(size*sizeof(int));
    int *offset_idx_send = (int*)my_malloc(size*sizeof(int));
    int *offset_idx_recv = (int*)my_malloc(size*sizeof(int));
    int *offset_pts_send = (int*)my_malloc(size*sizeof(int));
    int *offset_pts_recv = (int*)my_malloc(size*sizeof(int));
    int *offset_ngh_send = (int*)my_malloc(size*sizeof(int));
    int *offset_ngh_recv = (int*)my_malloc(size*sizeof(int));
    prev_send = 0;
    prev_recv = 0;
    int prev_array_send = 0, prev_array_recv = 0;
    int prev_neigh_send = 0, prev_neigh_recv = 0;
    for (i = 0; i < size; i++) {
      count_idx_send[i] = 0;
      count_idx_recv[i] = 0;
      count_ngh_send[i] = 0;
      count_ngh_recv[i] = 0;
      offset_idx_send[i] = prev_array_send;
      offset_idx_recv[i] = prev_array_recv;
      offset_pts_send[i] = ndim*prev_array_send;
      offset_pts_recv[i] = ndim*prev_array_recv;
      offset_ngh_send[i] = prev_neigh_send;
      offset_ngh_recv[i] = prev_neigh_recv;
      for (j = 0; j < count_send[i]; j++) {
    	count_idx_send[i] += cnt_send[prev_send];
    	count_ngh_send[i] += nct_send[prev_send];
    	prev_send++;
      }
      count_pts_send[i] = ndim*count_idx_send[i];
      prev_array_send += count_idx_send[i];
      prev_neigh_send += count_ngh_send[i];
      for (j = 0; j < count_recv[i]; j++) {
    	count_idx_recv[i] += (*cnt_recv)[prev_recv];
    	count_ngh_recv[i] += (*nct_recv)[prev_recv];
    	prev_recv++;
      }
      count_pts_recv[i] = ndim*count_idx_recv[i];
      prev_array_recv += count_idx_recv[i];
      prev_neigh_recv += count_ngh_recv[i];
    }
    free(count_send);
    free(count_recv);
    free(cnt_send);
    free(nct_send);
    // Allocate for arrays
    (*idx_recv) = (Info*)my_malloc(prev_array_recv*sizeof(Info));
    (*pts_recv) = (double*)my_malloc(ndim*prev_array_recv*sizeof(double));
    (*ngh_recv) = (uint32_t*)my_malloc(prev_neigh_recv*sizeof(uint32_t));
    Info *idx_send = (Info*)my_malloc(prev_array_send*sizeof(Info));
    double *pts_send = (double*)my_malloc(ndim*prev_array_send*sizeof(double));
    uint32_t *ngh_send = (uint32_t*)my_malloc(prev_neigh_send*sizeof(uint32_t));
    Info *idx_send_curr = idx_send;
    double *pts_send_curr = pts_send;
    uint32_t *ngh_send_curr = ngh_send;
    for (i = 0; i < size; i++) {
      if (count_idx_send[i] > 0) {
    	memmove(idx_send_curr, idx_out[i], count_idx_send[i]*sizeof(Info));
    	memmove(pts_send_curr, pts_out[i], count_pts_send[i]*sizeof(double));
    	idx_send_curr += count_idx_send[i];
    	pts_send_curr += count_pts_send[i];
      }
      if (count_ngh_send[i] > 0) {
    	memmove(ngh_send_curr, ngh_out[i], count_ngh_send[i]*sizeof(uint32_t));
    	ngh_send_curr += count_ngh_send[i];
      }
      idx_out[i] = NULL;
      pts_out[i] = NULL;
      ngh_out[i] = NULL;
    }
    // Send arrays
    if (sizeof(Info) == sizeof(uint32_t))
      MPI_Alltoallv(idx_send, count_idx_send, offset_idx_send, MPI_UNSIGNED,
		    *idx_recv, count_idx_recv, offset_idx_recv, MPI_UNSIGNED,
		    MPI_COMM_WORLD);
    else
      MPI_Alltoallv(idx_send, count_idx_send, offset_idx_send, MPI_UNSIGNED_LONG,
		    *idx_recv, count_idx_recv, offset_idx_recv, MPI_UNSIGNED_LONG,
		    MPI_COMM_WORLD);
    MPI_Alltoallv(pts_send, count_pts_send, offset_pts_send, MPI_DOUBLE,
    		  *pts_recv, count_pts_recv, offset_pts_recv, MPI_DOUBLE,
    		  MPI_COMM_WORLD);
    MPI_Alltoallv(ngh_send, count_ngh_send, offset_ngh_send, MPI_UNSIGNED,
    		  *ngh_recv, count_ngh_recv, offset_ngh_recv, MPI_UNSIGNED,
    		  MPI_COMM_WORLD);
    free(count_idx_send);
    free(count_pts_send);
    free(count_ngh_send);
    free(count_idx_recv);
    free(count_pts_recv);
    free(count_ngh_recv);
    free(offset_idx_send);
    free(offset_pts_send);
    free(offset_ngh_send);
    free(offset_idx_recv);
    free(offset_pts_recv);
    free(offset_ngh_recv);
    free(idx_send);
    free(pts_send);
    free(ngh_send);
    if (DEBUG)
      printf("%d: Finishing outgoing_points\n", rank);
    return count_recv_tot;
  }

  void domain_decomp() {
    int i;
    uint64_t j;
    uint32_t k;
    int *nleaves_per_proc = NULL;
    int leafsize_limit = 0;
    if (DEBUG)
      printf("%d: Beginning domain decomposition\n", rank);
    if (rank == 0) {
      // Create KDtree
      uint32_t leafsize;
      nleaves_total = size;
      nleaves_total = (int)(pow(2,ceil(log2((float)(nleaves_total)))));
      if (limit_mem > 1)
	nleaves_total *= limit_mem;
      leafsize = std::max((uint32_t)(npts_total/nleaves_total + 1), (ndim+1));
      idx_total = (uint64_t*)my_malloc(npts_total*sizeof(uint64_t));
      for (j = 0; j < npts_total; j++)
	idx_total[j] = j;
      tree = new KDTree(pts_total, idx_total, npts_total, ndim,
		        leafsize, le, re, periodic, false);
      tree->consolidate_edges();
      // info_total = (Info*)my_malloc(npts_total*sizeof(Info));
      // for (j = 0; j < npts_total; j++)
      // 	info_total[j] = idx_total[j];
      nleaves_total = tree->num_leaves;
    }
    MPI_Bcast(&nleaves_total, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Send number of leaves
    if (rank == 0) {
      nleaves_per_proc = (int*)my_malloc(sizeof(int)*size);
      for (i = 0; i < size; i++)
	nleaves_per_proc[i] = 0;
      for (k = 0; k < tree->num_leaves; k++) {
	nleaves_per_proc[k % size]++;
      }
    }
    MPI_Scatter(nleaves_per_proc, 1, MPI_INT,
		&nleaves, 1, MPI_INT,
		0, MPI_COMM_WORLD);
    if (nleaves == 1)
      limit_mem = 1;
    // Make sure leaves meet minimum criteria
    if (rank == 0) {
      for (i = 0; i < nleaves_total; i++) {
	if (tree->leaves[i]->children < (ndim+1)) {
	  leafsize_limit = tree->leaves[i]->children;
	  break;
	}
      }
    }
    MPI_Bcast(&leafsize_limit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (leafsize_limit)
      printf("Leafsize is too small (%d in %dD).", leafsize_limit, ndim);
      // my_error("Leafsize is too small (%d in %dD).",
      // 			       leafsize_limit, );
    // Send leaves
    if (rank == 0) {
      int task;
      int iroot = 0;
      for (i = 0; i < nleaves_total; i++) {
	task = i % size;
	if (task == rank) {
	  // leaves used
	  leaves.push_back(new CParallelLeaf<Info>(nleaves_total, ndim,
						   unique_str,
						   tree, i));
	  if (limit_mem > 1)
	    leaves[iroot]->dump();
	  map_id2idx[leaves[iroot]->id] = iroot;
	  iroot++;
	} else {
	  CParallelLeaf<Info> ileaf(nleaves_total, ndim, unique_str, tree, i);
	  ileaf.send(task);
	}
      }
      tree_exists = 1;
      for (task = 1; task < size; task++)
	MPI_Send(&tree_exists, 1, MPI_INT, task, 33, MPI_COMM_WORLD);
    } else {
      for (i = 0; i < nleaves; i++) {
	// leaves used
	leaves.push_back(new CParallelLeaf<Info>(nleaves_total, ndim,
						 unique_str, 0)); // calls recv
	if (limit_mem > 1)
	  leaves[i]->dump();
	map_id2idx[leaves[i]->id] = i;
      }
      MPI_Recv(&tree_exists, 1, MPI_INT, 0, 33, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }
    if (nleaves_per_proc != NULL)
      free(nleaves_per_proc);
    if (DEBUG)
      printf("%d: Finished domain decomposition\n", rank);
  }

  void parallel_domain_decomp() {
    int i;
    uint64_t j;
    uint32_t k;
    int leafsize_limit = 0, tot_leafsize_limit = 0;
    if (DEBUG)
      printf("%d: Beginning parallel domain decomposition\n", rank);
    // Determine leafsize
    uint32_t leafsize = 0;
    if (rank == 0) {
      // Create KDtree
      nleaves_total = size;
      nleaves_total = (int)(pow(2,ceil(log2((float)(nleaves_total)))));
      if (limit_mem > 1)
	nleaves_total *= limit_mem;
      leafsize = std::max((uint32_t)(npts_total/nleaves_total + 1), (ndim+1));
      idx_total = (uint64_t*)my_malloc(npts_total*sizeof(uint64_t));
      for (j = 0; j < npts_total; j++)
	idx_total[j] = j;
    }
    // Create tree
    ptree = new ParallelKDTree(pts_total, idx_total, npts_total, ndim,
			       leafsize, le, re, periodic, false);
    nleaves_total = ptree->tot_num_leaves;
    nleaves = ptree->tree->num_leaves;
    if (nleaves == 1)
      limit_mem = 1;
    // Create version of indices in correct format
    // if (rank == 0) {
    //   info_total = (Info*)my_malloc(npts_total*sizeof(Info));
    //   for (j = 0; j < npts_total; j++)
    //   	info_total[j] = idx_total[j];
    // }
    // Make sure leaves meet minimum criteria
    for (i = 0; i < nleaves; i++) {
      if (ptree->leaves[i]->children < (ndim+1)) {
	leafsize_limit = ptree->leaves[i]->children;
	break;
      }
    }
    MPI_Allreduce(&leafsize_limit, &tot_leafsize_limit, 1, MPI_INT, MPI_MAX,
		  MPI_COMM_WORLD);
    if (leafsize_limit)
      printf("Leafsize is too small (%d in %dD).", leafsize_limit, ndim);
      // my_error("Leafsize is too small (%d in %dD).",
      // 			       leafsize_limit, );
    // Create leaves from tree nodes
    for (i = 0; i < nleaves; i++) {
      leaves.push_back(new CParallelLeaf<Info>(nleaves_total, ndim,
					       unique_str, ptree, i));
      if (limit_mem > 1)
	leaves[i]->dump();
      map_id2idx[leaves[i]->id] = i;
      
    }
    tree_exists = 1;
    if (DEBUG)
      printf("%d: Finished parallel domain decomposition\n", rank);
  }

  uint32_t num_cells() {
    int i;
    uint32_t tot_ncells = 0;
    uint32_t tot_ncells_total = 0;
    if (DEBUG)
      printf("%d: Begining num_cells\n", rank);
    for (i = 0; i < nleaves; i++)
      tot_ncells += leaves[i]->ncells; // leaves used
    MPI_Reduce(&tot_ncells, &tot_ncells_total, 1, MPI_UNSIGNED,
	       MPI_SUM, 0, MPI_COMM_WORLD);
    if (DEBUG)
      printf("%d: Finished num_cells\n", rank);
    return tot_ncells_total;
  }

  void consolidate_vols(double *vols) {
    if (DEBUG)
      printf("%d: Beginning consolidate_vols\n", rank);
    int i, iroot, task;
    double *ivols = NULL;
    int j;
    int nvols;
    if (rank == 0) {
      iroot = 0;
      for (i = 0; i < nleaves_total; i++) {
	nvols = tree->leaves[i]->children;
	task = i % size;
	if (task == rank) {
	  // Local
	  if (limit_mem > 1)
	    leaves[iroot]->load();
	  leaves[iroot]->voronoi_volumes(&ivols);
	  if (limit_mem > 1)
	    leaves[iroot]->dump();
	  iroot++;
	} else {
	  ivols = (double*)my_realloc(ivols, nvols*sizeof(double),
				      "volumes being received");
	  MPI_Recv(ivols, nvols, MPI_DOUBLE, task, 0, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	}
	for (j = 0; j < nvols; j++) {
	  vols[tree->all_idx[tree->leaves[i]->left_idx+j]] = ivols[j];
	}
      }
    } else {
      for (i = 0; i < nleaves; i++) {
	nvols = leaves[i]->npts_orig;
	if (limit_mem > 1)
	  leaves[i]->load();
	leaves[i]->voronoi_volumes(&ivols);
	if (limit_mem > 1)
	  leaves[i]->dump();
	MPI_Send(ivols, nvols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
    }
    if (DEBUG)
      printf("%d: Finished consolidate_vols\n", rank);
  }

  uint64_t consolidate_tess(uint64_t tot_ncells_total, Info *tot_idx_inf,
			    Info *allverts, Info *allneigh) {
    if (DEBUG)
      printf("%d: Beginning consolidate_tess\n", rank);
    int i, task, s;
    uint64_t j;
    Info tn = 0, tm = 0;
    Info *verts = NULL, *neigh = NULL;
    uint32_t *idx_verts = NULL;
    uint64_t *idx_cells = NULL;
    Info idx_inf = 0;
    Info *header = (Info*)my_malloc(3*sizeof(Info));
    // Get counts
    uint64_t max_ncells = 0;
    uint64_t max_ncells_total;
    uint64_t incells;
    uint64_t out = 0;
    for (i = 0; i < nleaves; i++) {
      incells = leaves[i]->ncells; // leaves used
      if (incells > max_ncells)
	max_ncells = incells;
    }
    MPI_Allreduce(&max_ncells, &max_ncells_total, 1, MPI_UNSIGNED_LONG,
		  MPI_MAX, MPI_COMM_WORLD);
    // Preallocate
    verts = (Info*)my_malloc(max_ncells_total*(ndim+1)*sizeof(Info));
    neigh = (Info*)my_malloc(max_ncells_total*(ndim+1)*sizeof(Info));
    idx_verts = (uint32_t*)my_malloc(max_ncells_total*(ndim+1)*sizeof(uint32_t));
    idx_cells = (uint64_t*)my_malloc(max_ncells_total*sizeof(uint64_t));
    // Send serialized info
    if (rank == 0) {
      // Prepare object to hold consolidated tess info
      idx_inf = std::numeric_limits<Info>::max();
      for (j = 0; j < tot_ncells_total*(ndim+1); j++) {
    	allverts[j] = idx_inf;
    	allneigh[j] = idx_inf;
      }
      ConsolidatedLeaves<Info> cons;
      SerializedLeaf<Info> sleaf;
      uint64_t idx_start, idx_stop;
      cons = ConsolidatedLeaves<Info>(ndim, idx_inf,
				      (int64_t)tot_ncells_total,
				      allverts, allneigh);
      // Receive other leaves
      for (i = 0; i < nleaves_total; i++) {
    	task = i % size;
    	if (task == rank) {
	  // leaves used
	  if (limit_mem > 1)
	    leaves[i/size]->load();
    	  idx_inf = leaves[i/size]->serialize(tn, tm, verts, neigh,
					      idx_verts, idx_cells);
	  if (limit_mem > 1)
	    leaves[i/size]->dump();
    	} else {
    	  s = 0;
	  if (sizeof(Info) == sizeof(uint32_t))
	    MPI_Recv(header, 3, MPI_UNSIGNED, task, s++, MPI_COMM_WORLD,
		     MPI_STATUS_IGNORE);
	  else
	    MPI_Recv(header, 3, MPI_UNSIGNED_LONG, task, s++, MPI_COMM_WORLD,
		     MPI_STATUS_IGNORE);
    	  tn = header[0], tm = header[1], idx_inf = header[2];
	  if (sizeof(Info) == sizeof(uint32_t)) {
	    MPI_Recv(verts, tm*(ndim+1), MPI_UNSIGNED, task, s++,
		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Recv(neigh, tm*(ndim+1), MPI_UNSIGNED, task, s++,
		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  } else {
	    MPI_Recv(verts, tm*(ndim+1), MPI_UNSIGNED_LONG, task, s++,
		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Recv(neigh, tm*(ndim+1), MPI_UNSIGNED_LONG, task, s++,
		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  }
    	  MPI_Recv(idx_verts, tm*(ndim+1), MPI_UNSIGNED, task, s++,
    		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	  MPI_Recv(idx_cells, tm, MPI_UNSIGNED_LONG, task, s++,
    		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	}
    	// Insert serialized leaf
	// leaves used
    	idx_start = tree->leaves[i]->left_idx;
    	idx_stop = idx_start + tree->leaves[i]->children;
    	sleaf = SerializedLeaf<Info>(i, ndim, (int64_t)tm, idx_inf,
				     verts, neigh,
				     idx_verts, idx_cells,
				     idx_start, idx_stop);
    	cons.add_leaf(sleaf);
      }
      // Finalize consolidated object
      cons.add_inf();
      out = (uint64_t)cons.ncells;
      (*tot_idx_inf) = cons.idx_inf;
      // CGeneralDelaunay<Info> *Tout = new CGeneralDelaunay<Info>(ndim, false);
      // Tout->deserialize(npts_total, (uint64_t)cons.ncells, (int64_t)ndim,
      // 			pts_total, info_total, allverts, allneigh,
      // 			cons.idx_inf);
      // out = Tout->T;
      // free(allverts);
      // free(allneigh);
    } else {
      (*tot_idx_inf) = 0;
      // Send leaves to root
      for (i = 0; i < nleaves; i++) {
	// leaves used
	if (limit_mem > 1)
	  leaves[i]->load();
    	idx_inf = leaves[i]->serialize(tn, tm, verts, neigh,
    				       idx_verts, idx_cells);
	if (limit_mem > 1)
	  leaves[i]->dump();
    	header[0] = tn, header[1] = tm, header[2] = idx_inf;
    	s = 0;
	if (sizeof(Info) == sizeof(uint32_t)) {
	  MPI_Send(header, 3, MPI_UNSIGNED, 0, s++, MPI_COMM_WORLD);
	  MPI_Send(verts, tm*(ndim+1), MPI_UNSIGNED, 0, s++, MPI_COMM_WORLD);
	  MPI_Send(neigh, tm*(ndim+1), MPI_UNSIGNED, 0, s++, MPI_COMM_WORLD);
	} else {
	  MPI_Send(header, 3, MPI_UNSIGNED_LONG, 0, s++, MPI_COMM_WORLD);
	  MPI_Send(verts, tm*(ndim+1), MPI_UNSIGNED_LONG, 0, s++, MPI_COMM_WORLD);
	  MPI_Send(neigh, tm*(ndim+1), MPI_UNSIGNED_LONG, 0, s++, MPI_COMM_WORLD);
	}
    	MPI_Send(idx_verts, tm*(ndim+1), MPI_UNSIGNED, 0, s++, MPI_COMM_WORLD);
    	MPI_Send(idx_cells, tm, MPI_UNSIGNED_LONG, 0, s++, MPI_COMM_WORLD);
      }

    }
    free(header);
    free(verts);
    free(neigh);
    free(idx_verts);
    free(idx_cells);
    if (DEBUG)
      printf("%d: Finished consolidate_tess\n", rank);
    return out;
  }

};
