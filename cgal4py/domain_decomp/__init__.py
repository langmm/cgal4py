import numpy as np
import cykdtree

def tree(method, pts, left_edge, right_edge, periodic, *args, **kwargs):
    r"""Get tree for a given domain decomposition schema.

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
        periodic (bool): True if domain is periodic, False otherwise.
        *args: Variable argument list. Passed to the selected domain 
            decomposition method.
        **kwargs: Variable keyword argument list. Passed to the selected domain 
            decomposition method.

    Returns:
        object: Tree of type specified by `method`.

    Raises:
        ValueError: If `method` is not one of the supported domain decomposition 
            methods listed above.

    """
    # Get leaves
    if method.lower() == 'kdtree':
        tree = cykdtree.PyKDTree(pts, left_edge, right_edge, *args, **kwargs)
    else:
        raise ValueError("'{}' is not a supported domain decomposition.".format(method))
    # Return tree
    return tree

class Leaf(object):
    def __init__(self, leafid, idx, left_edge, right_edge,
                 periodic_left=None, periodic_right=None,
                 domain_width=None, neighbors=None, num_leaves=None,
                 start_idx=None):
        r"""A container for leaf info.

        Args:
            leafid (int): Unique index of this leaf.
            idx (np.ndarray of uint64): Indices of points in this leaf.
            left_edge (np.ndarray of float64): Leaf min along each dimension.
            right_edge (np.ndarray of float64): Leaf max along each dimension.
            periodic_left (np.ndarray of bool, optional): Truth of left edge 
                being periodic in each dimension. Defaults to None. Is set by
                :meth:`cgal4py.domain_decomp.leaves`.
            periodic_right (np.ndarray of bool, optional): Truth of right edge 
                being periodic in each dimension. Defaults to None. Is set by 
                :meth:`cgal4py.domain_decomp.leaves`.
            domain_width (np.ndarray of float64, optional): Domain width along 
                each dimension. Defaults to None. Is set by 
                :meth:`cgal4py.domain_decomp.leaves`.
            neighbors (list of dict, optional): Indices of neighboring leaves in 
                each dimension. Defaults to None. Is set by 
                :meth:`cgal4py.domain_decomp.leaves`.
            num_leaves (int, optional): Number of leaves in the domain 
                decomposition. Defaults to None. Is set by 
                :meth:`cgal4py.domain_decomp.leaves`.  
            start_idx (int, optional): Number of points on previous leaves or 
                starting  index of this leaf in tree idx. Defaults to None. Is 
                set by :meth:`cgal4py.domain_decomp.leaves`.

        Attributes:
            id (int): Unique index of this leaf.
            idx (np.ndarray of uint64): Indices of points in this leaf.
            ndim (int): Number of dimensions in the domain.
            left_edge (np.ndarray of float64): Domain minimum along each 
                dimension.
            right_edge (np.ndarray of float64): Domain maximum along each 
                dimension.
            periodic_left (np.ndarray of bool): Truth of left edge being 
                periodic in each dimension. 
            periodic_right (np.ndarray of bool): Truth of right edge being 
                periodic in each dimension. 
            domain_width (np.ndarray of float64): Domain width along each 
                dimension.
            neighbors (list of dict): Indices of neighboring leaves in each 
                dimension. 
            num_leaves (int): Number of leaves in the domain decomposition. 
            norig (int): Number of points initially on this leaf.
            start_idx (int): Number of points on previous leaves or starting 
                index of this leaf in tree idx.

        """
        self.id = leafid
        self.idx = idx
        self.ndim = len(left_edge)
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.periodic_left = periodic_left
        self.periodic_right = periodic_right
        self.domain_width = domain_width
        self.neighbors = neighbors
        self.num_leaves = num_leaves
        self.norig = len(idx)
        self.start_idx = start_idx

from kdtree import kdtree

def leaves(method, pts, left_edge, right_edge, periodic, *args, **kwargs):
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
        periodic (bool): True if domain is periodic, False otherwise.
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
    m = pts.shape[1]
    # Get leaves
    if method.lower() == 'kdtree':
        leaves = kdtree(pts, left_edge, right_edge, *args, **kwargs)
    else:
        raise ValueError("'{}' is not a supported domain decomposition.".format(method))
    # Set total number of leaves
    if leaves[0].num_leaves is None:
        num_leaves = len(leaves)
        for leaf in leaves:
            leaf.num_leaves = num_leaves
    # Set starting index
    if leaves[0].start_idx is None:
        nprev = 0
        for leaf in leaves:
            leaf.start_idx = nprev
            nprev += leaf.norig
    # Set domain width
    if leaves[0].domain_width is None:
        domain_width = right_edge - left_edge
        for leaf in leaves:
            leaf.domain_width = domain_width
    # Determine if leaves are on periodic boundaries
    if leaves[0].periodic_left is None:
        if periodic:
            for leaf in leaves:
                leaf.periodic_left = np.isclose(leaf.left_edge, left_edge)
                leaf.periodic_right = np.isclose(leaf.right_edge, right_edge)
        else:
            for leaf in leaves:
                leaf.periodic_left = np.zeros(leaf.ndim, 'bool')
                leaf.periodic_right = np.zeros(leaf.ndim, 'bool')
    # Add neighbors
    if leaves[0].neighbors is None:
        for leaf in leaves:
            leaf.neighbors = [
                {'left':[],'left_periodic':[],
                 'right':[],'right_periodic':[]} for i in range(leaf.ndim)]
            for prev in leaves[:(leaf.id+1)]:
                matches = [None for _ in range(m)]
                match = True
                for i in range(m):
                    if leaf.left_edge[i] > prev.right_edge[i]:
                        if not (leaf.periodic_right[i] and prev.periodic_left[i]):
                            match = False
                            break
                    if leaf.right_edge[i] < prev.left_edge[i]:
                        if not (prev.periodic_right[i] and leaf.periodic_left[i]):
                            match = False
                            break
                if match:
                    for i in range(m):
                        if np.isclose(leaf.left_edge[i], prev.right_edge[i]):
                            leaf.neighbors[i]['left'].append(prev.id)
                            prev.neighbors[i]['right'].append(leaf.id)
                        elif np.isclose(leaf.right_edge[i], prev.left_edge[i]):
                            leaf.neighbors[i]['right'].append(prev.id)
                            prev.neighbors[i]['left'].append(leaf.id)
                        if periodic:
                            if leaf.periodic_right[i] and prev.periodic_left[i]:
                                leaf.neighbors[i]['right_periodic'].append(prev.id)
                                prev.neighbors[i]['left_periodic'].append(leaf.id)
                            if prev.periodic_right[i] and leaf.periodic_left[i]:
                                leaf.neighbors[i]['left_periodic'].append(prev.id)
                                prev.neighbors[i]['right_periodic'].append(leaf.id)
    # Return leaves
    return leaves

__all__ = ["Leaf", "kdtree", "tree"]

