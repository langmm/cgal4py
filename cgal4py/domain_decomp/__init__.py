
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

__all__ = ["Leaf", "kdtree"]

