#include "mpi.h"
#include <math.h>
#include <stdint.h>
#include "c_delaunay2.hpp"
#include "c_kdtree.hpp"


typedef uint64_t Info;
typedef Delaunay_with_info_2<Info> Delaunay;


class Leaf
{
public:
  uint32_t nleaves;
  uint32_t ndim;
  uint64_t npts_orig = 0;
  uint64_t npts = 0;
  uint64_t *idx = NULL;
  double *pts = NULL;
  double *le = NULL;
  double *re = NULL;
  int nneigh;
  uint32_t *neigh = NULL;
  double *neigh_le;
  double *neigh_re;
  Delaunay *T = NULL;
  Leaf(uint32_t nleaves0, uint32_t ndim0, int src) {
    nleaves = nleaves0;
    ndim = ndim0;
    neigh = (uint32_t*)malloc(nleaves*sizeof(uint32_t));
    neigh_le = (double*)malloc(nleaves*ndim*sizeof(double*));
    neigh_re = (double*)malloc(nleaves*ndim*sizeof(double*));
    recv(src);
  };
  Leaf(uint32_t nleaves0, uint32_t ndim0, uint64_t npts0,
       uint64_t *idx0, double *pts0,
       double *le0, double *re0, std::vector<uint32_t> neigh0,
       std::vector<double*> neigh_le0, std::vector<double*> neigh_re0) {
    nleaves = nleaves0;
    ndim = ndim0;
    neigh = (uint32_t*)malloc(nleaves*sizeof(uint32_t));
    neigh_le = (double*)malloc(nleaves*ndim*sizeof(double*));
    neigh_re = (double*)malloc(nleaves*ndim*sizeof(double*));
    npts = npts0;
    idx = idx0;
    pts = pts0;
    le = le0;
    re = re0;
    nneigh = (int)(neigh0.size());
    for (int i = 0; i < nneigh; i++) {
      neigh[i] = neigh0[i];
      for (uint32_t j = 0; j < ndim; j++) {
	neigh_le[ndim*i+j] = neigh_le0[i][j];
	neigh_re[ndim*i+j] = neigh_re0[i][j];
      }
    }
  }

  void send(int dst) {
    MPI_Send(&npts, 1, MPI_UNSIGNED_LONG, dst, 0, MPI_COMM_WORLD);
    MPI_Send(idx, npts, MPI_UNSIGNED_LONG, dst, 1, MPI_COMM_WORLD);
    MPI_Send(pts, ndim*npts, MPI_DOUBLE, dst, 2, MPI_COMM_WORLD);
    MPI_Send(le, ndim, MPI_DOUBLE, dst, 3, MPI_COMM_WORLD);
    MPI_Send(re, ndim, MPI_DOUBLE, dst, 4, MPI_COMM_WORLD);
    MPI_Send(&nneigh, 1, MPI_INT, dst, 5, MPI_COMM_WORLD);
    MPI_Send(neigh, nneigh, MPI_UNSIGNED, dst, 6, MPI_COMM_WORLD);
    MPI_Send(neigh_le, nneigh*ndim, MPI_DOUBLE, dst, 7, MPI_COMM_WORLD);
    MPI_Send(neigh_re, nneigh*ndim, MPI_DOUBLE, dst, 7, MPI_COMM_WORLD);
  };

  void recv(int src) {
    MPI_Recv(&npts, 1, MPI_UNSIGNED_LONG, src, 0, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    idx = (uint64_t*)malloc(npts*sizeof(uint64_t));
    pts = (double*)malloc(ndim*npts*sizeof(double));
    le = (double*)malloc(ndim*sizeof(double));
    re = (double*)malloc(ndim*sizeof(double));
    MPI_Recv(idx, npts, MPI_UNSIGNED_LONG, src, 1, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(pts, ndim*npts, MPI_DOUBLE, src, 2, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(le, ndim, MPI_DOUBLE, src, 3, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(re, ndim, MPI_DOUBLE, src, 4, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(&nneigh, 1, MPI_INT, src, 5, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(neigh, nneigh, MPI_UNSIGNED, src, 6, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(neigh_le, nneigh*ndim, MPI_DOUBLE, src, 7, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(neigh_re, nneigh*ndim, MPI_DOUBLE, src, 7, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
  }

  void tessellate() {
    T = new Delaunay();
    T->insert(pts, idx, npts);
    npts_orig = npts;
  }

  std::vector<std::vector<Info>> outgoing_points() {

  }
};


//template <typename Info_>
class CParallelDelaunay
{
public:
  int rank;
  int size;
  uint32_t ndim;
  // Things only valid for root
  uint64_t npts_total;
  int nleaves_total;
  double *pts_total = NULL;
  uint64_t *idx_total = NULL;
  KDTree *tree = NULL;
  // Things for each process
  int nleaves;
  Leaf *leaves = NULL;
  CParallelDelaunay() {
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    printf("Hello from %d of %d.\n", rank, size);
  }

  void insert(uint64_t npts0, uint32_t ndim0, double *pts0,
	      double *le, double *re, bool *periodic) {
    ndim = ndim0;
    npts_total = npts0;
    pts_total = pts0;
    if (rank == 0) {
      MPI_Send(&ndim, 1, MPI_UNSIGNED, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&ndim, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }
    // Domain decomp
    domain_decomp(le, re, periodic);
    // Insert points into leaves
    tessellate();
    // Exchange points
    exchange();
    printf("Inserted points on %d of %d\n", rank, size);
  }

  void tessellate() {
    for (int i = 0; i < nleaves; i++)
      leaves[i].tessellate();
  }

  void exchange() {
    uint64_t nrecv_total = 1;
    uint64_t nrecv;
    while (nrecv_total != 0) {
      outgoing_points();
      nrecv = incoming_points();
      MPI_Allreduce(&nrecv, &nrecv_total, 1, MPI_UNSIGNED, MPI_SUM,
		    MPI_COMM_WORLD);
    }
  }

  void outgoing_points() {

  }

  uint64_t incoming_points() {
    uint64_t nrecv = 0;
    return nrecv;
  }

  void domain_decomp(double *le, double *re, bool *periodic) {
    int i;
    uint64_t j;
    uint32_t k;
    int *nleaves_per_proc = NULL;
    if (rank == 0) {
      // Create KDtree
      uint32_t leafsize;
      nleaves_total = size;
      nleaves_total = (int)(pow(2,ceil(log2((float)(nleaves_total)))));
      leafsize = (uint32_t)(npts_total/nleaves_total + 1);
      idx_total = (uint64_t*)malloc(npts_total*sizeof(uint64_t));
      for (j = 0; j < npts_total; j++)
	idx_total[j] = j;
      tree = new KDTree(pts_total, idx_total, npts_total, ndim,
		        leafsize, le, re, periodic);
      nleaves_total = tree->num_leaves;
      printf("%d leaves in total\n", nleaves_total);
    }
    MPI_Bcast(&nleaves_total, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Send number of leaves
    if (rank == 0) {
      nleaves_per_proc = (int*)malloc(sizeof(int)*size);
      for (i = 0; i < size; i++)
	nleaves_per_proc[i] = 0;
      for (k = 0; k < tree->num_leaves; k++) {
	nleaves_per_proc[k % size]++;
      }
    }
    MPI_Scatter(nleaves_per_proc, 1, MPI_INT,
		&nleaves, 1, MPI_INT,
		0, MPI_COMM_WORLD);
    leaves = (Leaf*)malloc(nleaves*sizeof(Leaf));

    // Send leaves
    if (rank == 0) {
      int task;
      double *ipts;
      int iroot = 0;
      uint64_t sidx = 0;
      for (i = 0; i < nleaves_total; i++) {
	task = i % size;
	sidx = tree->leaves[i]->left_idx;
	ipts = (double*)malloc(ndim*tree->leaves[i]->children*sizeof(double));
	std::vector<double*> neigh_le;
	std::vector<double*> neigh_re;
	std::vector<uint32_t>::iterator it;
	for (j = 0; j < tree->leaves[i]->children; j++) {
	  for (k = 0; k < ndim; k++) {
	    ipts[ndim*j+k] = pts_total[ndim*idx_total[sidx+j]+k];
	  }
	}
	for (it = tree->leaves[i]->all_neighbors.begin();
	     it != tree->leaves[i]->all_neighbors.end(); it++) {
	  neigh_le.push_back(tree->leaves[*it]->left_edge);
	  neigh_re.push_back(tree->leaves[*it]->right_edge);
	}
	Leaf ileaf(nleaves_total, ndim, tree->leaves[i]->children,
		   idx_total+sidx, ipts,
		   tree->leaves[i]->left_edge, tree->leaves[i]->right_edge,
		   tree->leaves[i]->all_neighbors, neigh_le, neigh_re);
	if (task == rank) {
	  leaves[iroot] = ileaf;
	  iroot++;
	} else {
	  ileaf.send(task);
	  free(ipts);
	}
      }
    } else {
      for (i = 0; i < nleaves; i++) {
	leaves[i] = Leaf(nleaves_total, ndim, 0); // calls recv
      }
    }
    printf("Received %d leaves on %d of %d\n", nleaves, rank, size);
  }

};
