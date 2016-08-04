
class Leaf(object):
    def __init__(self, leafid, idx, left_edge, right_edge):
        r"""A container for leaf info.

        Args:
            leafid (int): Unique index of this leaf.
            idx (np.ndarray of uint64): Indices of points in this leaf.
            left_edge (np.ndarray of float64): Domain minimum along each 
                dimension.
            right_edge (np.ndarray of float64): Domain maximum along each 
                dimension.

        Attributes:
            id (int): Unique index of this leaf.
            idx (np.ndarray of uint64): Indices of points in this leaf.
            left_edge (np.ndarray of float64): Domain minimum along each 
                dimension.
            right_edge (np.ndarray of float64): Domain maximum along each 
                dimension.

        """
        self.id = leafid
        self.idx = idx
        self.left_edge = left_edge
        self.right_edge = right_edge

from kdtree import kdtree

def leaves(method, pts, left_edge, right_edge, *args, **kwargs):
    r"""Get list of leaves for a given domain decomposition.

    Args:
        method (str): Domain decomposition method. Supported options are:
            'kdtree': KDTree based on median position along the dimension 
                with the greatest domain width. See 
                :meth:`cgal4py.domain_decomp.kdtree` for details on 
                accepted keyword arguments.
        pts (np.ndarray of float64): (n,m) array of n coordinates in a 
            m-dimensional domain. 
        left_edge (np.ndarray of float64): (m,) domain minimum in each dimension. 
        right_edge (np.ndarray of float64): (m,) domain maximum in each dimension. 
        *args: Variable argument list. Passed to the selected domain 
            decomposition method.
        **kwargs: Variable keyword argument list. Passed to the selected domain 
            decomposition method.

    Returns:
        list of :class:`cgal4py.domain_decomp.Leaf`s: Leaves returned by the 
            domain decomposition.

    Raises:
        ValueError: If `method` is not one of the supported domain decomposition 
            methods listed above.

    """
    if method.lower() == 'kdtree':
        return kdtree(pts, left_edge, right_edge, *args, **kwargs)
    else:
        raise ValueError("'{}' is not a supported domain decomposition.".format(method))

__all__ = ["Leaf", "kdtree"]

