import numpy as np

from delaunay2 import Delaunay2
from delaunay3 import Delaunay3

FLAG_DOUBLE_AVAIL = False
try:
    from delaunay2_64bit import Delaunay2 as Delaunay2_64bit
    from delaunay3_64bit import Delaunay3 as Delaunay3_64bit
    FLAG_DOUBLE_AVAIL = True
except:
    warnings.warn("Could not import packages using long indices. This feature will be disabled.")


def Delaunay(pts, use_double=False):
    r"""Get a triangulation for a set of points with arbitrary dimensionality.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates.
        use_double (bool): If True, the triangulation is forced to use 64bit 
            integers reguardless of if there are too many points for 32bit.
            Otherwise 32bit integers are used so long as the number of points is 
            <=4294967295. Defaults to False.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:
            2D or 3D triangulation class.

    Raises:
        ValueError: If pts is not a 2D array.
        NotImplementedError: If pts.shape[1] is not 2 or 3.
        RuntimeError: If there are >=4294967295 points or `use_double == True` 
            and the 64bit integer triangulation packages could not be imported.

    """
    if (pts.ndim != 2):
        raise ValueError("pts must be a 2D array of coordinates")
    npts = pts.shape[0]
    ndim = pts.shape[1]
    if npts >= np.iinfo('uint32').max or use_double:
        if not FLAG_DOUBLE_AVAIL:
            raise RuntimeError("The 64bit triangulation package couldn't "+
                               "be imported and there are {} points.".format(npts))
        use_double = True
    if ndim == 2:
        if use_double:
            T = Delaunay2_64bit()
        else:
            T = Delaunay2()
    elif ndim == 3:
        if use_double:
            T = Delaunay3_64bit()
        else:
            T = Delaunay3()
    else:
        raise NotImplementedError("Only 2D & 3D triangulations are currently supported.")
    T.insert(pts)
    return T

class DelaunayLeaf(object):
    r"""Wraps triangulation of a single leaf in a domain decomposition.

    Args:
        leaf (:class:`cgal4py.domain_decomp.Leaf`): Leaf in domain decomposition.
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates 
            that will be selected from by `leaf.idx`.

    Attributes:
        leaf (:class:`cgal4py.domain_decomp.Leaf`): Leaf in domain decomposition.
        T (:class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:):
            2D or 3D triangulation class. 

    """
    def __init__(self, leaf, pts):
        self.leaf = leaf
        self.T = Delaunay(pts[leaf.idx,:])
        self.wrapped = np.zeros(len(leaf.idx), 'bool')
    @property
    def id(self):
        r"""int: Unique index of this leaf."""
        return self.leaf.id
    @property
    def idx(self):
        r"""np.ndarray of uint64: Indices of points in this leaf."""
        return self.leaf.idx
    @property
    def ndim(self):
        r"""ndim: Number of dimensions in the domain."""
        return self.leaf.ndim
    @property
    def left_edge(self):
        r"""np.ndarray of float64: Domain minimum along each dimension."""
        return self.leaf.left_edge
    @property
    def right_edge(self):
        r"""np.ndarray of float64: Domain maximum along each dimension."""
        return self.leaf.right_edge
    @property
    def periodic_left(self):
        r"""np.ndarray of bool: Truth of left edge being periodic in each 
        dimension."""
        return self.leaf.periodic_left
    @property
    def periodic_right(self):
        r"""np.ndarray of bool: Truth of right edge being periodic in each 
        dimension."""
        return self.leaf.periodic_right
    @property
    def neighbors(self):
        r"""list of dict: Indices of neighboring leaves in each dimension."""
        return self.leaf.neighbors
    @property
    def num_leaves(self):
        r"""int: Number of leaves in the domain decomposition."""
        return self.leaf.num_leaves

    def incoming_points(self, leafid, idx, pos):
        r"""Add incoming points from other leaves.

        Args:
            leafid (int): ID for the leaf that points came from.
            idx (np.ndarray of int): Indices of points being recieved.
            pos (np.ndarray of float): Positions of points being recieved.

        """
        if len(idx) == 0: return
        # Wrap points
        wrapped = np.zeros(len(idx), 'bool')
        if self.id == leafid:
            for i in range(self.m):
                if self.periodic_left[i] and self.periodic_right[i]:
                    idx_left = (pos[:,i] - self.left_edge[i]) < (self.right_edge[i] - pos[:,i])
                    idx_right = (self.right_edge[i] - pos[:,i]) < (pos[:,i] - self.left_edge[i])
                    pos[idx_left,i] += self.domain_width[i]
                    pos[idx_right,i] -= self.domain_width[i]
                    wrapped[idx_left] = True
                    wrapped[idx_right] = True
        else:
            for i in range(self.m):
                if leafid in self.neighbors[i]['right_periodic']:
                    idx_left = (pos[:,i] + self.domain_width[i] - self.right_edge[i]) < (self.left_edge[i] - pos[:,i]) 
                    pos[idx_left,i] += self.domain_width[i]
                    wrapped[idx_left] = True
                if leafid in self.neighbors[i]['left_periodic']:
                    idx_right = (self.left_edge[i] - pos[:,i] + self.domain_width[i]) < (pos[:,i] - self.right_edge[i])
                    pos[idx_right,i] -= self.domain_width[i]
                    wrapped[idx_right] = True
        # Concatenate arrays
        self.idx = np.concatenate([self.idx, idx])
        # self.pos = np.concatenate([self.pos, pos])
        self.wrapped = np.concatenate([self.wrapped, wrapped])
        # Insert points
        self.T.insert(pos)

    def outgoing_points(self):
        r"""Get indices of points that should be sent to each neighbor."""
        # TODO: Check that iind does not matter. iind contains points in tets 
        # that are infinite. For non-periodic boundary conditions, these points 
        # may need to be sent to distant leaves for an accurate convex hull. 
        # Currently points on an edge without a bordering leaf are sent to all 
        # leaves, but it is possible this could miss a few points...
        lind, rind, iind = self.T.boundary_points(self.left_edge,
                                                  self.right_edge,
                                                  True)
        # Count for preallocation
        Nind = np.zeros(self.num_leaves, 'uint32')
        for i in range(self.ndim):
            l_neighbors = self.neighbors[i]['left'] + \
              self.neighbors[i]['left_periodic']
            r_neighbors = self.neighbors[i]['right'] + \
              self.neighbors[i]['right_periodic']
            if len(l_neighbors) == 0:
                l_neighbors = range(self.num_leaves)
            if len(r_neighbors) == 0:
                r_neighbors = range(self.num_leaves)
            Nind[l_neighbors] += len(lind[i])
            Nind[r_neighbors] += len(rind[i])
        # Add points
        hvall = [np.zeros(Nind[k], iind.dtype) for k in xrange(self.num_leaves)]
        Cind = np.zeros(self.num_leaves, 'uint32')
        for i in range(self.ndim):
            l_neighbors = self.neighbors[i]['left'] + \
              self.neighbors[i]['left_periodic']
            r_neighbors = self.neighbors[i]['right'] + \
              self.neighbors[i]['right_periodic']
            if len(l_neighbors) == 0:
                l_neighbors = range(self.num_leaves)
            if len(r_neighbors) == 0:
                r_neighbors = range(self.num_leaves)
            ilN = len(lind[i])
            irN = len(rind[i])
            for k in l_neighbors:
                hvall[k][Cind[k]:(Cind[k]+ilN)] = lind[i]
                Cind[k] += ilN
            for k in r_neighbors:
                hvall[k][Cind[k]:(Cind[k]+irN)] = rind[i]
                Cind[k] += irN
        # Ensure unique values (overlap can happen if a point is at a corner)
        for k in xrange(num_leaves):
            hvall[k] = self.idx[np.unique(hvall[k])]
        return hvall

            
__all__ = ["Delaunay", "DelaunayLeaf", "Delaunay2", "Delaunay3"]        

if FLAG_DOUBLE_AVAIL:
    __all__ += ["Delaunay2_64bit", "Delaunay3_64bit"]
