#include "mpi.h"
#include <math.h>
#include <stdint.h>
#include <vector>
#include <set>
#include <map>
#include "c_delaunay2.hpp"
#include "c_kdtree.hpp"


typedef uint64_t Info;
typedef Delaunay_with_info_2<Info> Delaunay;


class Leaf
{
public:
  bool from_node;
  int size;
  int rank;
  uint32_t id;
  uint32_t nleaves;
  uint32_t ndim;
  uint64_t npts_orig = 0;
  uint64_t npts = 0;
  uint64_t *idx = NULL;
  double *pts = NULL;
  double *le = NULL;
  double *re = NULL;
  int *periodic_le = NULL;
  int *periodic_re = NULL;
  double *domain_width = NULL;
  int nneigh;
  uint32_t *neigh = NULL;
  double *neigh_le;
  double *neigh_re;
  double *leaves_le;
  double *leaves_re; 
  Delaunay *T = NULL;
  // std::set<uint32_t> *all_neigh;
  // std::vector<std::set<uint32_t>> *lneigh;
  // std::vector<std::set<uint32_t>> *rneigh;

  Leaf(uint32_t nleaves0, uint32_t ndim0, int src) {
    uint32_t k;
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    from_node = false;
    nleaves = nleaves0;
    ndim = ndim0;
    neigh = (uint32_t*)malloc(nleaves*sizeof(uint32_t));
    neigh_le = (double*)malloc(nleaves*ndim*sizeof(double*));
    neigh_re = (double*)malloc(nleaves*ndim*sizeof(double*));
    leaves_le = (double*)malloc(nleaves*ndim*sizeof(double*));
    leaves_re = (double*)malloc(nleaves*ndim*sizeof(double*));
    // all_neigh = new std::set<uint32_t>();
    // lneigh = new std::vector<*std::set<uint32_t>>();
    // rneigh = new std::vector<*std::set<uint32_t>>();
    // all_neigh->reserve(nleaves);
    // for (k = 0; k < ndim; k++) {
    //   lneigh->push_back(new std::set<uint32_t>());
    //   (*lneigh)[k]->reserve(nleaves);
    //   rneigh->push_back(new std::set<uint32_t>());
    //   (*rneigh)[k]->reserve(nleaves);
    // }
    // Receive leaf info from root process
    recv(src);
    int i;
    uint32_t n;
    for (i = 0; i < nneigh; i++) {
      n = neigh[i];
      memcpy(neigh_le+ndim*i, leaves_le+ndim*n, ndim*sizeof(double));
      memcpy(neigh_re+ndim*i, leaves_re+ndim*n, ndim*sizeof(double));
    }
  };

  Leaf(uint32_t nleaves0, uint32_t ndim0, KDTree* tree, int index) {
    uint32_t k;
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    from_node = true;
    nleaves = nleaves0;
    ndim = ndim0;
    neigh = (uint32_t*)malloc(nleaves*sizeof(uint32_t));
    neigh_le = (double*)malloc(nleaves*ndim*sizeof(double*));
    neigh_re = (double*)malloc(nleaves*ndim*sizeof(double*));
    leaves_le = (double*)malloc(nleaves*ndim*sizeof(double*));
    leaves_re = (double*)malloc(nleaves*ndim*sizeof(double*));
    // all_neigh = new std::set<uint32_t>();
    // lneigh = new std::vector<std::set<uint32_t>>();
    // rneigh = new std::vector<std::set<uint32_t>>();
    // for (k = 0; k < ndim; k++) {
    //   lneigh->push_back(std::set<uint32_t>());
    //   rneigh->push_back(std::set<uint32_t>());
    // }
    // Transfer leaf information
    Node* node = tree->leaves[index];
    std::vector<uint32_t>::iterator it;
    int i;
    uint64_t j;
    uint32_t n;
    id = node->leafid;
    npts = node->children;
    idx = tree->all_idx + node->left_idx;
    pts = (double*)malloc(ndim*npts*sizeof(double));
    le = node->left_edge;
    re = node->right_edge;
    periodic_le = (int*)malloc(ndim*sizeof(int));
    periodic_re = (int*)malloc(ndim*sizeof(int));
    domain_width = tree->domain_width;
    nneigh = (int)(node->all_neighbors.size());
    for (j = 0; j < npts; j++) {
      for (k = 0; k < ndim; k++) {
    	pts[ndim*j+k] = tree->all_pts[ndim*tree->all_idx[node->left_idx+j]+k];
      }
    }
    for (k = 0; k < nleaves; k++) {
      memcpy(leaves_le+ndim*k, tree->leaves[k]->left_edge,
    	     ndim*sizeof(double));
      memcpy(leaves_re+ndim*k, tree->leaves[k]->right_edge,
    	     ndim*sizeof(double));
    }
    for (i = 0; i < nneigh; i++) {
      neigh[i] = node->all_neighbors[i];
    }
    for (k = 0; k < ndim; k++) {
      periodic_le[k] = node->periodic_left[k];
      periodic_re[k] = node->periodic_right[k];
    }
    // for (k = 0; k < ndim; k++) {
    //   (*lneigh)[k].insert(node->left_neighbors[k].begin(),
    // 			  node->left_neighbors[k].end());
    //   (*rneigh)[k].insert(node->right_neighbors[k].begin(),
    // 			  node->right_neighbors[k].end());
    // }
    // Shift edges of periodic neighbors
    // for (k = 0; k < ndim; k++) {
    //   if (periodic_le[k]) {
    // 	for (std::set<uint32_t>::iterator it = (*lneigh)[k].begin();
    // 	     it != (*lneigh)[k].end(); it++) {
    // 	  leaves_le[*it, k] -= domain_width[k];
    // 	  leaves_re[*it, k] -= domain_width[k];
    // 	}
    //   }
    //   if (periodic_re[k]) {
    // 	for (std::set<uint32_t>::iterator it = (*rneigh)[k].begin();
    // 	     it != (*rneigh)[k].end(); it++) {
    // 	  leaves_le[*it, k] += domain_width[k];
    // 	  leaves_re[*it, k] += domain_width[k];
    // 	}
    //   }
    // }
    // Select edges of neighbors
    for (i = 0; i < nneigh; i++) {
      n = neigh[i];
      memcpy(neigh_le+ndim*i, leaves_le+ndim*n, ndim*sizeof(double));
      memcpy(neigh_re+ndim*i, leaves_re+ndim*n, ndim*sizeof(double));
    }
  }

  void send(int dst) {
    int i = 0, j;
    uint32_t k = 0;
    MPI_Send(&id, 1, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    MPI_Send(&npts, 1, MPI_UNSIGNED_LONG, dst, i++, MPI_COMM_WORLD);
    MPI_Send(idx, npts, MPI_UNSIGNED_LONG, dst, i++, MPI_COMM_WORLD);
    MPI_Send(pts, ndim*npts, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(le, ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(re, ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(periodic_le, ndim, MPI_INT, dst, i++, MPI_COMM_WORLD);
    MPI_Send(periodic_re, ndim, MPI_INT, dst, i++, MPI_COMM_WORLD);
    MPI_Send(domain_width, ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(&nneigh, 1, MPI_INT, dst, i++, MPI_COMM_WORLD);
    MPI_Send(neigh, nneigh, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    MPI_Send(leaves_le, nleaves*ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    MPI_Send(leaves_re, nleaves*ndim, MPI_DOUBLE, dst, i++, MPI_COMM_WORLD);
    // uint32_t *dummy = (uint32_t*)malloc(nneigh*sizeof(uint32_t));
    // int ndum;
    // for (k = 0; k < ndim; k++) {
    //   // left neighbors
    //   ndum = (int)((*lneigh)[k].size());
    //   j = 0;
    //   for (std::set<uint32_t>::iterator it = (*lneigh)[k].begin();
    // 	   it != (*lneigh)[k].end(); it++) {
    // 	dummy[j] = *it;
    // 	j++;
    //   }
    //   MPI_Send(&ndum, 1, MPI_INT, dst, i++, MPI_COMM_WORLD);
    //   MPI_Send(dummy, ndum, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    //   // right neighbors
    //   ndum = (int)((*rneigh)[k].size());
    //   j = 0;
    //   for (std::set<uint32_t>::iterator it = (*rneigh)[k].begin();
    // 	   it != (*rneigh)[k].end(); it++) {
    // 	dummy[j] = *it;
    // 	j++;
    //   }
    //   MPI_Send(&ndum, 1, MPI_INT, dst, i++, MPI_COMM_WORLD);
    //   MPI_Send(dummy, ndum, MPI_UNSIGNED, dst, i++, MPI_COMM_WORLD);
    // }
    // free(dummy);
  };

  void recv(int src) {
    int i = 0, j;
    uint32_t k = 0;
    MPI_Recv(&id, 1, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(&npts, 1, MPI_UNSIGNED_LONG, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    idx = (uint64_t*)malloc(npts*sizeof(uint64_t));
    pts = (double*)malloc(ndim*npts*sizeof(double));
    le = (double*)malloc(ndim*sizeof(double));
    re = (double*)malloc(ndim*sizeof(double));
    periodic_le = (int*)malloc(ndim*sizeof(int));
    periodic_re = (int*)malloc(ndim*sizeof(int));
    domain_width = (double*)malloc(ndim*sizeof(double));
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
    MPI_Recv(&nneigh, 1, MPI_INT, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(neigh, nneigh, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(leaves_le, nleaves*ndim, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    MPI_Recv(leaves_re, nleaves*ndim, MPI_DOUBLE, src, i++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    // uint32_t *dummy = (uint32_t*)malloc(nneigh*sizeof(uint32_t));
    // int ndum;
    // for (k = 0; k < ndim; k++) {
    //   // left neighbors
    //   MPI_Recv(&ndum, 1, MPI_INT, src, i++, MPI_COMM_WORLD,
    // 	       MPI_STATUS_IGNORE);
    //   MPI_Recv(dummy, ndum, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
    // 	       MPI_STATUS_IGNORE);
    //   for (j = 0; j < ndum; j++) {
    // 	(*lneigh)[k].insert(dummy[j]);
    //   }
    //   // right neighbors
    //   MPI_Recv(&ndum, 1, MPI_INT, src, i++, MPI_COMM_WORLD,
    // 	       MPI_STATUS_IGNORE);
    //   MPI_Recv(dummy, ndum, MPI_UNSIGNED, src, i++, MPI_COMM_WORLD,
    // 	       MPI_STATUS_IGNORE);
    //   for (j = 0; j < ndum; j++) {
    // 	(*rneigh)[k].insert(dummy[j]);
    //   }
    // }
    // free(dummy);
  }

  void tessellate() {
    T = new Delaunay();
    printf("%d: npts = %d\n", id, npts);
    T->insert(pts, idx, npts);
    npts_orig = npts;
  }

  void outgoing_points(std::vector<std::vector<uint32_t>> &src_out,
		       std::vector<std::vector<uint32_t>> &dst_out,
		       std::vector<std::vector<uint32_t>> &cnt_out,
		       std::vector<std::vector<uint32_t>> &nct_out,
		       std::vector<uint64_t*> &idx_out,
		       std::vector<double*> &pts_out,
		       std::vector<uint32_t*> &ngh_out) {
    int i, j;
    uint32_t k, n, dst, src=id;
    int task;
    std::vector<uint64_t>::iterator it64;
    std::vector<uint32_t>::iterator it32;
    std::vector<std::vector<uint64_t>> out_leaves;
    out_leaves = T->outgoing_points(nneigh, neigh_le, neigh_re);
    uint32_t ntot = 0;
    uint32_t nold, nnew, nold_neigh, nnew_neigh;
    for (i = 0; i < nneigh; i++) {
      dst = neigh[i];
      task = dst % size;
      src_out[task].push_back(src);
      dst_out[task].push_back(dst);
      for (it64 = out_leaves[i].begin(); it64 != out_leaves[i].end(); ) {
	if (*it64 < npts_orig)
	  it64++;
	else
	  it64 = out_leaves[i].erase(it64);
      }
      nnew = (uint32_t)(out_leaves[i].size());
      nold = 0;
      for (it32 = cnt_out[task].begin();
	   it32 != cnt_out[task].end(); it32++)
	nold += *it32;
      if (nnew > 0)
	nnew_neigh = nneigh;
      else
	nnew_neigh = 0;
      nold_neigh = 0;
      for (it32 = nct_out[task].begin();
	   it32 != nct_out[task].end(); it32++)
	nold_neigh += *it32;
      // TODO: Maybe move realloc outside of loop
      idx_out[task] = (uint64_t*)realloc(idx_out[task],
					 (nold+nnew)*sizeof(uint64_t));
      pts_out[task] = (double*)realloc(pts_out[task],
				       ndim*(nold+nnew)*sizeof(double));
      ngh_out[task] = (uint32_t*)realloc(ngh_out[task],
					 (nold_neigh+nnew_neigh)*sizeof(uint32_t));
      cnt_out[task].push_back(nnew);
      nct_out[task].push_back(nnew_neigh);
      for (it64 = out_leaves[i].begin(), j = nold;
	   it64 != out_leaves[i].end(); it64++, j++) {
	idx_out[task][j] = idx[*it64];
	for (k = 0; k < ndim; k++)
	  pts_out[task][ndim*j+k] = pts[ndim*(*it64)+k];
      }
      ntot += nnew;
      if (nnew_neigh > 0) {
	memcpy(ngh_out[task]+nold_neigh, neigh, nnew_neigh*sizeof(uint32_t));
      }
    }
    // Transfer neighbors to log
    for (i = 0; i < nneigh; i++) {
      n = neigh[i];
      // all_neigh->insert(n);
    }
    nneigh = 0;
  }

  void incoming_points(uint32_t src, uint32_t nnpts_recv,
		       uint32_t nneigh_recv, uint64_t *idx_recv,
		       double *pts_recv, uint32_t *neigh_recv) {
    if (nnpts_recv == 0)
      return;
    int i;
    uint64_t j;
    uint32_t k;
    if (src == id) {
      for (k = 0; k < ndim; k++) {
	// if (periodic_le[k] and periodic_re[k]) {
	//   for (j = 0; j < nnpts_recv; j++) {
	//     if ((pts_recv[ndim*j+k] - le[k]) < (re[k] - pts_recv[ndim*j+k])) {
	//       pts_recv[ndim*j+k] += domain_width[k];
	//     }
	//     if ((re[k] - pts_recv[ndim*j+k]) < (pts_recv[ndim*j+k] - le[k])) {
	//       pts_recv[ndim*j+k] -= domain_width[k];
	//     }
	//   }
	// }
      }
    } else {
      for (k = 0; k < ndim; k++) {
	// if (periodic_re[k] and 
      }
    }
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
  std::map<int,uint32_t> map_id2idx;

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
    printf("Done on %d of %d\n", rank, size);
  }

  void tessellate() {
    for (int i = 0; i < nleaves; i++)
      leaves[i].tessellate();
  }

  void exchange() {
    uint64_t nrecv_total = 1;
    uint64_t nrecv;
    int nexch;
    uint32_t *src_recv;
    uint32_t *dst_recv;
    uint32_t *cnt_recv;
    uint32_t *nct_recv;
    uint64_t *idx_recv;
    double *pts_recv;
    uint32_t *ngh_recv;
    while (nrecv_total != 0) {
      src_recv = NULL;
      dst_recv = NULL;
      cnt_recv = NULL;
      nct_recv = NULL;
      idx_recv = NULL;
      pts_recv = NULL;
      ngh_recv = NULL;
      nexch = outgoing_points(src_recv, dst_recv, cnt_recv, nct_recv,
			      idx_recv, pts_recv, ngh_recv);
      // nrecv = incoming_points(nexch, src_recv, dst_recv, cnt_recv, nct_recv,
      // 			      idx_recv, pts_recv, ngh_recv);
      MPI_Allreduce(&nrecv, &nrecv_total, 1, MPI_UNSIGNED, MPI_SUM,
		    MPI_COMM_WORLD);
    }
  }

  uint64_t incoming_points(int nexch, uint32_t *src_recv, uint32_t *dst_recv,
			   uint32_t *cnt_recv, uint32_t *nct_recv,
			   uint64_t *idx_recv, double *pts_recv,
			   uint32_t *ngh_recv) {
    uint64_t nrecv = 0;
    uint64_t nprev_pts = 0, nprev_ngh;
    uint64_t *iidx;
    double *ipts;
    uint32_t *ingh;
    int dst;
    for (int i = 0; i < nexch; i++) {
      iidx = idx_recv + nprev_pts;
      ipts = pts_recv + ndim*nprev_pts;
      ingh = ngh_recv + nprev_ngh;
      dst = map_id2idx[dst_recv[i]];
      leaves[dst].incoming_points(src_recv[i], cnt_recv[i], nct_recv[i],
				  iidx, ipts, ingh);
      nprev_pts += cnt_recv[i];
      nprev_ngh += nct_recv[i];
    }
    nrecv = nprev_pts;
    return nrecv;
  }

  int outgoing_points(uint32_t *src_recv, uint32_t *dst_recv,
		      uint32_t *cnt_recv, uint32_t *nct_recv,
		      uint64_t *idx_recv, double *pts_recv,
		      uint32_t *ngh_recv) {
    int i, j;
    // Get output from each leaf
    std::vector<std::vector<uint32_t>> src_out, dst_out, cnt_out, nct_out;
    std::vector<uint64_t*> idx_out;
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
    for (i = 0; i < nleaves; i++)
      leaves[i].outgoing_points(src_out, dst_out, cnt_out, nct_out,
				idx_out, pts_out, ngh_out);
    // Send expected counts
    int *count_send = (int*)malloc(size*sizeof(int));
    int *count_recv = (int*)malloc(size*sizeof(int));
    int *offset_send = (int*)malloc(size*sizeof(int));
    int *offset_recv = (int*)malloc(size*sizeof(int));
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
    uint32_t *src_send = (uint32_t*)malloc(count_send_tot*sizeof(uint32_t));
    uint32_t *dst_send = (uint32_t*)malloc(count_send_tot*sizeof(uint32_t));
    uint32_t *cnt_send = (uint32_t*)malloc(count_send_tot*sizeof(uint32_t));
    uint32_t *nct_send = (uint32_t*)malloc(count_send_tot*sizeof(uint32_t));
    src_recv = (uint32_t*)malloc(count_recv_tot*sizeof(uint32_t));
    dst_recv = (uint32_t*)malloc(count_recv_tot*sizeof(uint32_t));
    cnt_recv = (uint32_t*)malloc(count_recv_tot*sizeof(uint32_t));
    nct_recv = (uint32_t*)malloc(count_recv_tot*sizeof(uint32_t));
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
		  src_recv, count_recv, offset_recv, MPI_UNSIGNED,
		  MPI_COMM_WORLD);
    MPI_Alltoallv(dst_send, count_send, offset_send, MPI_UNSIGNED,
		  dst_recv, count_recv, offset_recv, MPI_UNSIGNED,
		  MPI_COMM_WORLD);
    MPI_Alltoallv(cnt_send, count_send, offset_send, MPI_UNSIGNED,
		  cnt_recv, count_recv, offset_recv, MPI_UNSIGNED,
		  MPI_COMM_WORLD);
    MPI_Alltoallv(nct_send, count_send, offset_send, MPI_UNSIGNED,
		  nct_recv, count_recv, offset_recv, MPI_UNSIGNED,
		  MPI_COMM_WORLD);
    // Send arrays
    int *count_idx_send = (int*)malloc(size*sizeof(int));
    int *count_idx_recv = (int*)malloc(size*sizeof(int));
    int *count_pts_send = (int*)malloc(size*sizeof(int));
    int *count_pts_recv = (int*)malloc(size*sizeof(int));
    int *count_ngh_send = (int*)malloc(size*sizeof(int));
    int *count_ngh_recv = (int*)malloc(size*sizeof(int));
    int *offset_idx_send = (int*)malloc(size*sizeof(int));
    int *offset_idx_recv = (int*)malloc(size*sizeof(int));
    int *offset_pts_send = (int*)malloc(size*sizeof(int));
    int *offset_pts_recv = (int*)malloc(size*sizeof(int));
    int *offset_ngh_send = (int*)malloc(size*sizeof(int));
    int *offset_ngh_recv = (int*)malloc(size*sizeof(int));
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
	count_idx_recv[i] += cnt_recv[prev_recv];
	count_ngh_recv[i] += nct_recv[prev_recv];
	prev_recv++;
      }
      count_pts_recv[i] = ndim*count_idx_recv[i];
      prev_array_recv += count_idx_recv[i];
      prev_neigh_recv += count_ngh_recv[i];
    }
    idx_recv = (uint64_t*)malloc(prev_array_recv*sizeof(uint64_t));
    pts_recv = (double*)malloc(ndim*prev_array_recv*sizeof(double));
    ngh_recv = (uint32_t*)malloc(prev_neigh_recv*sizeof(uint32_t));
    uint64_t *idx_send = (uint64_t*)malloc(prev_array_send*sizeof(uint64_t));
    double *pts_send = (double*)malloc(ndim*prev_array_send*sizeof(double));
    uint32_t *ngh_send = (uint32_t*)malloc(prev_neigh_send*sizeof(uint32_t));
    uint64_t *idx_send_curr = idx_send;
    double *pts_send_curr = pts_send;
    uint32_t *ngh_send_curr = ngh_send;
    for (i = 0; i < size; i++) {
      if (count_idx_send[i] > 0) {
	memmove(idx_send_curr, idx_out[i], count_idx_send[i]*sizeof(uint64_t));
	memmove(pts_send_curr, pts_out[i], count_pts_send[i]*sizeof(double));
	idx_send_curr += count_idx_send[i];
	pts_send_curr += count_pts_send[i];
      }
      if (count_ngh_send[i] > 0) {
	memmove(ngh_send_curr, ngh_out[i], count_ngh_send[i]*sizeof(uint32_t));
	ngh_send_curr += count_ngh_send[i];
      }
    }
    MPI_Alltoallv(idx_send, count_idx_send, offset_idx_send, MPI_UNSIGNED_LONG,
    		  idx_recv, count_idx_recv, offset_idx_recv, MPI_UNSIGNED_LONG,
    		  MPI_COMM_WORLD);
    MPI_Alltoallv(pts_send, count_pts_send, offset_pts_send, MPI_DOUBLE,
    		  pts_recv, count_pts_recv, offset_pts_recv, MPI_DOUBLE,
    		  MPI_COMM_WORLD);
    MPI_Alltoallv(ngh_send, count_ngh_send, offset_ngh_send, MPI_UNSIGNED,
		  ngh_recv, count_ngh_recv, offset_ngh_recv, MPI_UNSIGNED,
		  MPI_COMM_WORLD);
    // Free things
    free(count_send);
    free(offset_send);
    free(src_send);
    free(dst_send);
    free(cnt_send);
    free(nct_send);
    free(count_idx_send);
    free(count_pts_send);
    free(count_ngh_send);
    free(offset_idx_send);
    free(offset_pts_send);
    free(offset_ngh_send);
    free(count_recv);
    free(offset_recv);
    free(src_recv);
    free(dst_recv);
    free(cnt_recv);
    free(nct_recv);
    free(count_idx_recv);
    free(count_pts_recv);
    free(count_ngh_recv);
    free(offset_idx_recv);
    free(offset_pts_recv);
    free(offset_ngh_recv);
    free(idx_send);
    free(pts_send);
    free(ngh_send);
    free(src_send);
    free(dst_send);
    free(cnt_send);
    free(nct_send);
    return count_recv_tot;
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
		        leafsize, le, re, periodic, false);
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
      int iroot = 0;
      for (i = 0; i < nleaves_total; i++) {
	task = i % size;
	Leaf ileaf(nleaves_total, ndim, tree, i);
	if (task == rank) {
	  printf("%d of %d\n", iroot, nleaves);
	  leaves[iroot] = ileaf;
	  map_id2idx[leaves[iroot].id] = iroot;
	  iroot++;
	} else {
	  ileaf.send(task);
	}
      }
    } else {
      for (i = 0; i < nleaves; i++) {
	leaves[i] = Leaf(nleaves_total, ndim, 0); // calls recv
	map_id2idx[leaves[i].id] = i;
      }
    }
    printf("Received %d leaves on %d of %d\n", nleaves, rank, size);
  }

};
