from cgal4py.domain_decomp import Leaf

import cython
import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

def kdtree(np.ndarray[double, ndim=2] pts,
           np.ndarray[double, ndim=1] left_edge, 
           np.ndarray[double, ndim=1] right_edge, 
           int leafsize = 10000):
    r"""Get the leaves in a KDTree constructed for a set of points.

    Args:
        pts (np.ndarray of double): (n,m) array of n coordinates in a 
            m-dimensional domain.
        left_edge (np.ndarray of double): (m,) domain minimum in each dimension.
        right_edge (np.ndarray of double): (m,) domain maximum in each dimension.
        leafsize (int, optional): The maximum number of points that should be in 
            a leaf. Defaults to 10000.
        
    Returns:
        list of :class:`domain_decomp.Leaf`s: Leaves in the KDTree.

    Raises:
        ValueError: If `leafsize < 2`. This currectly segfaults.

    """
    if (leafsize < 2):
        # This is here to prevent segfault. The cpp code needs modified to 
        # support leafsize = 1
        raise ValueError("'leafsize' cannot be smaller than 2.")
    cdef uint32_t k,i
    cdef uint64_t npts = <uint64_t>pts.shape[0]
    cdef uint32_t ndim = <uint32_t>pts.shape[1]
    cdef np.ndarray[np.uint64_t] idx = np.arange(npts).astype('uint64')
    cdef KDTree* tree = new KDTree(&pts[0,0], &idx[0], npts, ndim, 
                                   <uint32_t>leafsize, 
                                   &left_edge[0], &right_edge[0])
    cdef object leaves = []
    cdef np.ndarray[np.float64_t] leaf_left_edge = np.zeros(ndim, 'float64')
    cdef np.ndarray[np.float64_t] leaf_right_edge = np.zeros(ndim, 'float64')
    for k in xrange(tree.leaves.size()):
        leafnode = tree.leaves[k]
        leaf_idx = idx[leafnode.left_idx:(leafnode.left_idx + leafnode.children)] 
        assert(len(leaf_idx) == <int>leafnode.children)
        for i in range(ndim):
            leaf_left_edge[i] = leafnode.left_edge[i]
            leaf_right_edge[i] = leafnode.right_edge[i]
        leaves.append(Leaf(k, leaf_idx, leaf_left_edge, leaf_right_edge))
    return leaves
