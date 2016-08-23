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
        use_double (bool, optional): If True, 64bit integers are used to track 
            points and up to 18446744073709551615 points can be triangulated.
            Otherwise, 32bit integers are used and only 4294967295 points can be 
            triangulated. Defaults to `False`. This option is only available if 
            the 64bit versions could be succesfully imported.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:
            2D or 3D triangulation class.

    Raises:
        ValueError: If pts is not a 2D array.
        NotImplementedError: If pts.shape[1] is not 2 or 3.

    """
    if (pts.ndim != 2):
        raise ValueError("pts must be a 2D array of coordinates")
    ndim = pts.shape[1]
    if ndim == 2:
        if use_double and FLAG_DOUBLE_AVAIL:
            T = Delaunay2_64bit()
        else:
            T = Delaunay2()
    elif ndim == 3:
        if use_double and FLAG_DOUBLE_AVAIL:
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
            hvall[k] = np.unique(hvall[k])
        return hvall
            

# class Delaunay(object):#Delaunay2,Delaunay3):

#     @staticmethod
#     def _make_delaunay_method(fname, ndim=None):
#         def wrapped_func(solf, *args, **kwargs):
#             return getattr(solf._T, fname)(*args, **kwargs)
#         wrapped_func.__name__ = fname
#         wrapped_func.__doc__ = getattr(Delaunay2, fname).__doc__
#         wrapped_func.__doc__.replace('Delaunay2','Delaunay')
#         if ndim == 2:
#             wrapped_func.__doc__ += "\n\nONLY VALID FOR 2D TRIANGULATIONS"
#         elif ndim == 3:
#             wrapped_func.__doc__ += "\n\nONLY VALID FOR 3D TRIANGULATIONS"
#         return wrapped_func

#     _delaunay_methods = ['is_valid','write_to_file','read_from_file','plot',
#                          'insert','clear','remove','move','move_if_no_collision',
#                          'flip','flip_flippable','get_vertex','locate',
#                          'is_edge','is_cell','nearest_edge','mirror_index','mirror_vertex',
#                          'get_boundary_of_conflicts','get_conflicts','get_conflicts_and_boundary']
#     _delaunay_properties = ['num_finite_verts','num_finite_edges','num_finite_cells',
#                             'num_infinite_verts','num_infinite_edges','num_infinite_cells',
#                             'num_verts','num_edges','num_cells','vertices','edges',
#                             'all_verts_begin','all_verts_end','all_verts','finite_verts',
#                             'all_edges_begin','all_edges_end','all_edges','finite_edges',
#                             'all_cells_begin','all_cells_end','all_cells','finite_cells']
#     _delaunay2_methods = ['includes_edge','mirror_edge','line_walk']
#     _delaunay2_properties = []
#     _delaunay3_methods = ['mirror_facet']
#     _delaunay3_properties = ['num_finite_facets','num_infinite_facets','num_facets',
#                              'all_facets_begin','all_facets_end','all_facets','finite_facets']

#     for k in _delaunay_methods:
#         setattr(Delaunay, k, Delaunay._make_delaunay_method(k))
#     for k in _delaunay2_methods:
#         setattr(Delaunay, k, Delaunay._make_delaunay_method(k, ndim=2))
#     for k in _delaunay3_methods:
#         setattr(Delaunay, k, Delaunay._make_delaunay_method(k, ndim=3))
#     for k in _delaunay_properties:
#         property(setattr(Delaunay, k, Delaunay._make_delaunay_method(k)))
#     for k in _delaunay2_properties:
#         property(setattr(Delaunay, k, Delaunay._make_delaunay_method(k, ndim=2)))
#     for k in _delaunay3_properties:
#         property(setattr(Delaunay, k, Delaunay._make_delaunay_method(k, ndim=3)))

#     def __init__(self, pts, parallel=False, domain_decomp='kdtree'):
#         r"""Initialize a Delaunay triangulation.

#         Args:
#             pts (np.ndarray of float64): Points to be triangulated.
#             parallel (bool, optional): If True, the triangulation is done in 
#                 parallel. Defaults to False.
#             domain_decomp (str, optional): Specifies the method that should be 
#                 used to decompose the domain for parallel triangulation. If 
#                 `parallel` is False, `domain_decomp` is not used. Defaults to 
#                 'kdtree'. Options include:
#                     'kdtree': A KD-tree is used to split the triangulation 
#                         between processors.
                        
#         Attributes:
#             ndim (int): The number of dimensions that the points occupy.

#         """
#         if parallel:
#             raise NotImplementedError
#         else:
#             self.ndim = pts.shape[-1]
#             if self.ndim == 2:
#                 self._T = Delaunay2()
#                 self._T.insert(pts)
#             elif self.ndim == 3:
#                 self._T = Delaunay3()
#                 self._T.insert(pts)
#             else:
#                 raise NotImplementedError("Only 2D & 3D triangulations are currently supported.")


            
__all__ = ["Delaunay", "Delaunay2", "Delaunay3"]        
