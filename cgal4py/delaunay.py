import numpy as np

from delaunay2 import Delaunay2
from delaunay3 import Delaunay3

class Delaunay(object):

    def __init__(self, pts, parallel=False, domain_decomp='kdtree'):
        r"""Initialize a Delaunay triangulation.

        Args:
            pts (np.ndarray of float64): Points to be triangulated.
            parallel (bool, optional): If True, the triangulation is done in 
                parallel. Defaults to False.
            domain_decomp (str, optional): Specifies the method that should be 
                used to decompose the domain for parallel triangulation. If 
                `parallel` is False, `domain_decomp` is not used. Defaults to 
                'kdtree'. Options include:
                    'kdtree': A KD-tree is used to split the triangulation 
                        between processors.

        Returns:
            Delaunay: Triangulation object.

        """
        if parallel:
            raise NotImplementedError
        else:
            ndim = pts.shape[-1]
            if ndim == 2:
                self.T = Delaunay2()
                self.T.insert(pts)
            elif ndim == 3:
                self.T = Delaunay3()
                self.T.insert(pts)
            else:
                raise NotImplementedError("Only 2D & 3D triangulations are currently supported.")
        
        
