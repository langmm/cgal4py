import numpy as np

from delaunay2 import Delaunay2
from delaunay3 import Delaunay3

def Delaunay(pts):
    r"""Get a triangulation for a set of points with arbitrary dimensionality.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates.

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
        T = Delaunay2()
    elif ndim == 3:
        T = Delaunay3()
    else:
        raise NotImplementedError("Only 2D & 3D triangulations are currently supported.")
    T.insert(pts)
    return T    


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
