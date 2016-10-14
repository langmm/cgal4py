import numpy as np

import tools
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
        use_double (bool, optional): If True, the triangulation is forced to use 
            64bit integers reguardless of if there are too many points for 32bit.
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
    # Check if 64bit integers need/can be used
    if npts >= np.iinfo('uint32').max or use_double:
        if not FLAG_DOUBLE_AVAIL:
            raise RuntimeError("The 64bit triangulation package couldn't "+
                               "be imported and there are {} points.".format(npts))
        use_double = True
    # Initialize correct tessellation
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
    # Insert points into tessellation
    if npts > 0:
        T.insert(pts)
    return T

            
def VoronoiVolumes(pts, use_double=False):
    r"""Get the volumes of the voronoi cells associated with a set of points.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates.
        use_double (bool, optional): If True, the triangulation is forced to use 
            64bit integers reguardless of if there are too many points for 32bit.
            Otherwise 32bit integers are used so long as the number of points is 
            <=4294967295. Defaults to False.

    Returns:
        np.ndarray of float64: Volumes of voronoi cells. Negative values 
            indicate infinite cells.

    """
    T = Delaunay(pts, use_double=use_double)
    return T.voronoi_volumes()

            
__all__ = ["tools", "Delaunay", "VoronoiVolumes", "Delaunay2", "Delaunay3"]

if FLAG_DOUBLE_AVAIL:
    __all__ += ["Delaunay2_64bit", "Delaunay3_64bit"]
