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
  uint64_t n = 0;
  uint64_t *idx = NULL;
  double *pts = NULL;
  Delaunay *T = NULL;
  Leaf() {};
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
    int i;
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
    for (i = 0; i < nleaves; i++) {
      leaves[i].T = new Delaunay();
      leaves[i].T->insert(leaves[i].pts, leaves[i].idx, leaves[i].n);
    }
    printf("Inserted points on %d of %d\n", rank, size);
  }

  void exchange() {

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
    MPI_Scatter(nleaves_per_proc, 1, MPI_INT, &nleaves, 1, MPI_INT,
		0, MPI_COMM_WORLD);
    leaves = (Leaf*)malloc(nleaves*sizeof(Leaf));
    for (i = 0; i < nleaves; i++) {
      leaves[i] = Leaf();
    }

    // Send size of leaves
    uint64_t *leaf_sizes = NULL;
    uint64_t *leaf_sizes_total = NULL;
    int *offset_leaves = NULL;
    int *counts_leaves = NULL;
    leaf_sizes = (uint64_t*)malloc(nleaves*sizeof(uint64_t));
    if (rank == 0) {
      offset_leaves = (int*)malloc(size*sizeof(int));
      counts_leaves = (int*)malloc(size*sizeof(int));
      leaf_sizes_total = (uint64_t*)malloc(nleaves_total*sizeof(uint64_t));
      i = 0, j = 0;
      for (int p = 0; p < size; p++) {
	counts_leaves[p] = nleaves_per_proc[p];
	offset_leaves[p] = i;
	j = p;
	while (j < (uint64_t)(nleaves_total)) {
	  leaf_sizes_total[i] = tree->leaves[j]->children;
	  j += size;
	  i++;
	}
      }
    }
    MPI_Scatterv(leaf_sizes_total, counts_leaves, offset_leaves,
		 MPI_UNSIGNED_LONG, leaf_sizes, nleaves,
		 MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    for (i = 0; i < nleaves; i++) {
      leaves[i].n = leaf_sizes[i];
    }
    free(leaf_sizes);
    if (rank == 0) {
      free(leaf_sizes_total);
      free(offset_leaves);
      free(counts_leaves);
    }

    // Send indexes and points
    if (rank == 0) {
      int task;
      uint64_t nprev = 0;
      double *ipts;
      for (i = 0; i < nleaves_total; i++) {
	task = i % size;
	ipts = (double*)malloc(ndim*tree->leaves[i]->children*sizeof(double));
	for (j = 0; j < tree->leaves[i]->children; j++) {
	  for (k = 0; k < ndim; k++) {
	    ipts[ndim*j+k] = pts_total[ndim*idx_total[nprev+j]+k];
	  }
	}
	if (task == rank) {
	  leaves[i].idx = idx_total+nprev;
	  leaves[i].pts = ipts;
	} else {
	  MPI_Send(idx_total+nprev, tree->leaves[i]->children,
		   MPI_UNSIGNED_LONG, task, 0, MPI_COMM_WORLD);
	  MPI_Send(ipts, ndim*tree->leaves[i]->children,
		   MPI_DOUBLE, task, 0, MPI_COMM_WORLD);
	  free(ipts);
	}
	nprev += tree->leaves[i]->children;
      }
    } else {
      for (i = 0; i < nleaves; i++) {
	leaves[i].idx = (uint64_t*)malloc(leaves[i].n*sizeof(uint64_t));
	leaves[i].pts = (double*)malloc(ndim*leaves[i].n*sizeof(double));
	MPI_Recv(leaves[i].idx, leaves[i].n, MPI_UNSIGNED_LONG,
		 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(leaves[i].pts, ndim*leaves[i].n, MPI_DOUBLE,
		 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    printf("Received indexes for %d leaves on %d of %d\n", nleaves, rank, size);
  }

};
