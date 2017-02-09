import numpy as np
import cykdtree as kdtree
from cgal4py import PY_MAJOR_VERSION

def tree(method, pts, left_edge, right_edge, periodic, *args, **kwargs):
    r"""Get tree for a given domain decomposition schema.

    Args:
        method (str): Domain decomposition method. Supported options are:
            'kdtree': KDTree based on median position along the dimension
                with the greatest domain width. See
                :meth:`cgal4py.domain_decomp.kdtree` for details on
                accepted keyword arguments.
        pts (np.ndarray of float64): (n, m) array of n coordinates in a
            m-dimensional domain.
        left_edge (np.ndarray of float64): (m,) domain minimum in each
            dimension.
        right_edge (np.ndarray of float64): (m,) domain maximum in each
            dimension.
        *args: Variable argument list. Passed to the selected domain
            decomposition method.
        **kwargs: Variable keyword argument list. Passed to the selected
            domain decomposition method.

    Returns:
        object: Tree of type specified by `method`.

    Raises:
        ValueError: If `method` is not one of the supported domain
            decomposition methods listed above.

    """
    # Get leaves
    if method.lower() == 'kdtree':
        tree = kdtree.PyKDTree(pts, left_edge, right_edge, *args, **kwargs)
    else:
        raise ValueError("'{}' is not a supported ".format(method) +
                         "domain decomposition.")
    # Return tree
    return tree


class GenericLeaf(object):
    def __init__(self, npts, left_edge, right_edge):
        r"""A generic container for leaf info with the minimum required info.
        These leaves must still be processed to add additional properties using
        :meth:`cgal4py.domain_decomp.process_leaves`, but can serve as a base
        class for supplemental domain decompositions and be provided to
        :class:`cgal4py.domain_decomp.GenericTree`.

        Args:
            npts (int): Number of points on this leaf.
            left_edge (np.ndarray of float64): Leaf min along each dimension.
            right_edge (np.ndarray of float64): Leaf max along each dimension.

        Attributes:
            npts (int): Number of points on this leaf, including those added
                during communication.
            left_edge (np.ndarray of float64): Domain minimum along each
                dimension.
            right_edge (np.ndarray of float64): Domain maximum along each
                dimension.

        """
        self.npts = npts
        self.left_edge = left_edge
        self.right_edge = right_edge

    @classmethod
    def from_leaf(cls, leaf):
        r"""Construct a GenericLeaf from a non-generic leaf.

        Args:
            leaf (Leaf): A leaf object.
       
        Returns:
            :class:`cgal4py.domain_decomp.GenericLeaf`: Generic version of the
                input leaf.

        """
        out = cls(leaf.npts, leaf.left_edge, leaf.right_edge)
        other_attr = ['id', 'ndim', 'num_leaves', 'start_idx', 'stop_idx',
                      'domain_width', 'periodic_left', 'periodic_right',
                      'left_neighbors', 'right_neighbors', 'neighbors']
        for k in other_attr:
            if hasattr(leaf, k):
                setattr(out, k, getattr(leaf, k))
        return out


class GenericTree(object):
    def __init__(self, idx, leaves, left_edge, right_edge, periodic):
        r"""Generic container for domain decomposition with the minimal
        required info. The leaves must have at least the following
        attributes:
            npts (int): Number of points on the leaf.
            left_edge (np.ndarray of float): min of leaf extent in each
                dimension.
            right_edge (np.ndarray of float): max of leaf extent in each
                dimension.

        Args:
            idx (np.ndarray of int): Indices sorting points in the tree by the
                leaf that contains them.
            leaves (list of leaf objects): Leaves in an arbitrary domain
                decomposition.
            left_edge (np.ndarray of float): domain minimum in each dimension.
            right_edge (np.ndarray of float): domain maximum in each dimension.
            periodic (bool): True if domain is periodic, False otherwise.

        Attributes:
            idx (np.ndarray of int): Indices sorting points in the tree by the
                leaf that contains them.
            leaves (list of leaf objects): Leaves in an arbitrary domain
                decomposition.
            left_edge (np.ndarray of float): domain minimum in each dimension.
            right_edge (np.ndarray of float): domain maximum in each dimension.
            periodic (bool): True if domain is periodic, False otherwise.
            num_leaves (int): Number of leaves in the tree.
            domain_width (np.ndarray of float): Domain width in each dimension.

        """
        self.idx = idx
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.periodic = periodic
        self.domain_width = right_edge - left_edge
        self.num_leaves = len(leaves)
        self.leaves = process_leaves(leaves, left_edge, right_edge, periodic)

    @classmethod
    def from_tree(cls, tree):
        r"""Construct a GenericTree from a non-generic tree.

        Args:
            tree (Tree): A tree object.
       
        Returns:
            :class:`cgal4py.domain_decomp.GenericTree`: Generic version of the
                input tree.

        """
        leaves = [GenericLeaf.from_leaf(leaf) for leaf in tree.leaves]
        out = cls(tree.idx, leaves, tree.left_edge, tree.right_edge,
                  tree.periodic)
        other_attr = []
        for k in other_attr:
            if hasattr(leaf, k):
                setattr(out, k, getattr(leaf, k))
        return out


def process_leaves(leaves, left_edge, right_edge, periodic):
    r"""Process leaves ensuring they have the necessary information/methods
    for parallel tessellation. Each leaf must have at least the following
    attributes:
        npts (int): Number of points on the leaf.
        left_edge (np.ndarray of float): min of leaf extent in each dimension.
        right_edge (np.ndarray of float): max of leaf extent in each dimension.

    Args:
        leaves (list of leaf objects): Leaves in an arbitrary domain
            decomposition.
        left_edge (np.ndarray of float64): domain minimum in each dimension.
        right_edge (np.ndarray of float64): domain maximum in each dimension.
        periodic (bool): True if domain is periodic, False otherwise.

    Returns:
        list of leaf objects: Leaves process with additional attributes added
            if they do not exist and can be added.

    Raises:
        AttributeError: If a leaf does not have one of the required attributes.
        TypeError: If a leaf has a required attribute, but of the wrong type.
        ValueError: If a leaf has a required attribute, but of the wrong size.

    """
    ndim = len(left_edge)
    req_attr = {'npts': [int, np.int32, np.uint32, np.int64, np.uint64],
                'left_edge': ([float, np.float32, np.float64], (ndim,)),
                'right_edge': ([float, np.float32, np.float64], (ndim,))}
    if PY_MAJOR_VERSION < 3:
        req_attr['npts'].append(long)
    # Check for necessary attributes
    for k, v in req_attr.items():
        if isinstance(v, tuple):
            for i, leaf in enumerate(leaves):
                if not hasattr(leaf, k):
                    raise AttributeError(
                        "Leaf {} does not have attribute {}.".format(i, k))
                lv = getattr(leaf, k)
                if not isinstance(lv, np.ndarray):
                    raise TypeError("Attribute {} ".format(k) +
                                    "of leaf {} ".format(i) +
                                    "is not an array.\n" +
                                    "It is type {}.".format(type(lv)))
                if lv.dtype not in v[0]:
                    raise TypeError("Attribute {} ".format(k) +
                                    "of leaf {} ".format(i) +
                                    "is not an array with dtype " +
                                    "{}.\n".format(v[0]) +
                                    "It is type {}.".format(lv.dtype))
                if v[1] is not None and lv.shape != v[1]:
                    raise ValueError("Attribute {} ".format(k) +
                                     "of leaf {} ".format(i) +
                                     "is not an array with shape " +
                                     "{}.\n".format(v[1]) +
                                     "It is shape {}.".format(lv.shape))
        else:
            for i, leaf in enumerate(leaves):
                if not hasattr(leaf, k):
                    raise AttributeError("Leaf {} does not ".format(i) +
                                         "have attribute {}.".format(k))
                lv = getattr(leaf, k)
                if not isinstance(lv, tuple(v)):
                    raise TypeError("Attribute {} ".format(k) +
                                    "of leaf {} is not ".format(i) +
                                    "of type {}.\n".format(v) +
                                    "It is type {}.".format(type(lv)))
    # Set id & ensure leaves are sorted
    if getattr(leaves[0], 'id', None) is None:
        for i, leaf in enumerate(leaves):
            leaf.id = i
    else:
        leaves = sorted(leaves, key=lambda l: l.id)
    # Set number of dimensions
    if getattr(leaves[0], 'ndim', None) is None:
        for leaf in leaves:
            leaf.ndim = ndim
    # Set total number of leaves
    if getattr(leaves[0], 'num_leaves', None) is None:
        num_leaves = len(leaves)
        for leaf in leaves:
            leaf.num_leaves = num_leaves
    # Set starting index
    if getattr(leaves[0], 'start_idx', None) is None:
        nprev = 0
        for leaf in leaves:
            leaf.start_idx = nprev
            nprev += leaf.npts
    # Set stopping index
    if getattr(leaves[0], 'stop_idx', None) is None:
        for leaf in leaves:
            leaf.stop_idx = leaf.start_idx + leaf.npts
    # Set domain width
    if getattr(leaves[0], 'domain_width', None) is None:
        domain_width = right_edge - left_edge
        for leaf in leaves:
            leaf.domain_width = domain_width
    # Determine if leaves are on periodic boundaries
    if getattr(leaves[0], 'periodic_left', None) is None:
        if periodic:
            for leaf in leaves:
                leaf.periodic_left = np.isclose(leaf.left_edge, left_edge)
                leaf.periodic_right = np.isclose(leaf.right_edge, right_edge)
        else:
            for leaf in leaves:
                leaf.periodic_left = np.zeros(leaf.ndim, 'bool')
                leaf.periodic_right = np.zeros(leaf.ndim, 'bool')
    # Add neighbors
    if getattr(leaves[0], 'left_neighbors', None) is None:
        for j, leaf in enumerate(leaves):
            leaf.left_neighbors = [[] for _ in range(ndim)]
            leaf.right_neighbors = [[] for _ in range(ndim)]
            for prev in leaves[:(j+1)]:
                match = True
                for i in range(ndim):
                    if leaf.left_edge[i] > prev.right_edge[i]:
                        if not (leaf.periodic_right[i] and
                                prev.periodic_left[i]):
                            match = False
                            break
                    if leaf.right_edge[i] < prev.left_edge[i]:
                        if not (prev.periodic_right[i] and
                                leaf.periodic_left[i]):
                            match = False
                            break
                if match:
                    for i in range(ndim):
                        if np.isclose(leaf.left_edge[i], prev.right_edge[i]):
                            leaf.left_neighbors[i].append(prev.id)
                            prev.right_neighbors[i].append(leaf.id)
                        elif np.isclose(leaf.right_edge[i], prev.left_edge[i]):
                            leaf.right_neighbors[i].append(prev.id)
                            prev.left_neighbors[i].append(leaf.id)
                        if periodic:
                            if (leaf.periodic_right[i] and
                                    prev.periodic_left[i]):
                                leaf.right_neighbors[i].append(prev.id)
                                prev.left_neighbors[i].append(leaf.id)
                            if (prev.periodic_right[i] and
                                    leaf.periodic_left[i]):
                                leaf.left_neighbors[i].append(prev.id)
                                prev.right_neighbors[i].append(leaf.id)
    if getattr(leaves[0], 'neighbors', None) is None:
        for leaf in leaves:
            neighbors = [leaf.id]
            for i in range(ndim):
                neighbors += leaf.left_neighbors[i]
                neighbors += leaf.right_neighbors[i]
            leaf.neighbors = list(set(neighbors))
    # Return leaves
    return leaves


__all__ = ["tree", "kdtree", "GenericLeaf", "GenericTree", "process_leaves"]
