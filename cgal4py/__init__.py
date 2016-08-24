import plot
import utils
import domain_decomp
from delaunay import Delaunay

import numpy as np
import sys, os
import warnings

FLAG_MULTIPROC = False
try:
    import parallel
    FLAG_MULTIPROC = True
except:
    warnings.warn("Package for parallel triangulation could not be imported. "+
                      "Parallel features will be disabled.")

class Triangulate(object):
    def __init__(self, pts, left_edge=None, right_edge=None, periodic=False,
                 dd_method='kdtree', dd_kwargs={}, nproc=0, limit_mem=False):
        r"""Triangulation of points.

        Args:
            pts (np.ndarray of float64): (n,m) Array of n mD points.
            left_edge (np.ndarray of float64, optional): (m,) lower limits on 
                the domain. If None, this is set to np.min(pts, axis=0). 
                Defaults to None.
            right_edge (np.ndarray of float64, optional): (m,) upper limits on 
                the domain. If None, this is set to np.max(pts, axis=0). 
                Defaults to None.
            periodic (bool optional): If True, the domain is assumed to be 
                periodic at its left and right edges. Defaults to False.
            dd_method (str, optional): String specifier for a domain 
                decomposition method. See :meth:`cgal4py.domain_decomp.leaves` 
                for available values. Defaults to 'kdtree'.
            dd_kwargs (dict, optional): Dictionary of keyword arguments for the 
                selected domain decomposition method. Defaults to empty dict.
            nproc (int, optional): The number of MPI processes that should be 
                spawned. If <2, no processes are spawned. Defaults to 0.
            limit_mem (bool, optional): If True, memory usage is limited by 
                writing things to file at a cost to performance. Defaults to 
                False.

        Raises:
            ValueError: If `pts` is not a 2D array.
            ValueError: If `left_edge` is not a 1D array with `pts.shape[1]` 
                elements.
            ValueError: If `right_edge` is not a 1D array with `pts.shape[1]` 
                elements.

        Attributes:

        """
        # Check input
        if (pts.ndim != 2):
            raise ValueError("pts must be a 2D array of coordinates")
        npts = pts.shape[0]
        ndim = pts.shape[1]
        if left_edge is None:
            left_edge = np.min(pts, axis=0)
        else:
            if (left_edge.ndim != 1) or (len(left_edge) != ndim):
                raise ValueError("left_edge must be a 1D array with {} elements.".format(ndim))
        if right_edge is None:
            right_edge = np.max(pts, axis=0)
        else:
            if (right_edge.ndim != 1) or (len(right_edge) != ndim):
                raise ValueError("right_edge must be a 1D array with {} elements.".format(ndim))
        # Assign attributes
        self.pts = pts
        self.npts = npts
        self.ndim = ndim
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.periodic = periodic
        # Parallel 
        if nproc > 1 and FLAG_MULTIPROC:
            leaves = domain_decomp.leaves(dd_method, pts, left_edge, right_edge, 
                                          periodic, **dd_kwargs)
            self.T = parallel.ParallelDelaunay(leaves, pts, nproc)
        # Serial
        else:
            if periodic:
                raise NotImplementedError
            else:
                self.T = Delaunay(pts)
    

__all__ = ["delaunay","plot","utils","domain_decomp"]
