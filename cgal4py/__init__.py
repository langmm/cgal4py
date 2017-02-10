r"""Package for performing Delaunay triangulations in Python using Cython
wrapped CGAL C++ libraries."""
import sys
PY_MAJOR_VERSION = sys.version_info[0]
from cgal4py import plot
from cgal4py import domain_decomp
from cgal4py import delaunay
from cgal4py.delaunay import Delaunay
import numpy as np
import warnings
FLAG_MULTIPROC = False
try:
    from cgal4py import parallel
    FLAG_MULTIPROC = True
except:
    warnings.warn("Package for parallel triangulation could not be " +
                  "imported. Parallel features will be disabled.")
    raise


def triangulate(pts, left_edge=None, right_edge=None, periodic=False,
                use_double=False, nproc=0, dd_method='kdtree', dd_kwargs={},
                limit_mem=False, **kwargs):
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
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        nproc (int, optional): The number of MPI processes that should be
            spawned. If <2, no processes are spawned. Defaults to 0.
        dd_method (str, optional): String specifier for a domain
            decomposition method. See :meth:`cgal4py.domain_decomp.leaves`
            for available values. Defaults to 'kdtree'.
        dd_kwargs (dict, optional): Dictionary of keyword arguments for the
            selected domain decomposition method. Defaults to empty dict.
        limit_mem (bool, optional): If True, memory usage is limited by
            writing things to file at a cost to performance. Defaults to
            False.
        \*\*kwargs: Additiona keyword arguments are passed to the appropriate
            class for constructuing the triangulation.

    Returns:
        T (:class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`:): 2D or 3D triangulation
            object.

    Raises:
        ValueError: If `pts` is not a 2D array.
        ValueError: If `left_edge` is not a 1D array with `pts.shape[1]`
            elements.
        ValueError: If `right_edge` is not a 1D array with `pts.shape[1]`
            elements.

    """
    # Check input
    if (pts.ndim != 2):
        raise ValueError("pts must be a 2D array of coordinates")
    ndim = pts.shape[1]
    if left_edge is None:
        left_edge = np.min(pts, axis=0)
    else:
        if (left_edge.ndim != 1) or (len(left_edge) != ndim):
            raise ValueError("left_edge must be a 1D array with " +
                             "{} elements.".format(ndim))
    if right_edge is None:
        right_edge = np.max(pts, axis=0)
    else:
        if (right_edge.ndim != 1) or (len(right_edge) != ndim):
            raise ValueError("right_edge must be a 1D array with " +
                             "{} elements.".format(ndim))
    # Parallel
    if nproc > 1 and FLAG_MULTIPROC:
        if (not 'nleaves' in dd_kwargs) and (not 'leafsize' in dd_kwargs):
            if limit_mem:
                dd_kwargs['nleaves'] = 2*nproc
            else:
                dd_kwargs['nleaves'] = nproc
        tree = domain_decomp.tree(dd_method, pts, left_edge, right_edge,
                                  periodic, **dd_kwargs)
        T = parallel.ParallelDelaunay(pts, tree, nproc, limit_mem=limit_mem,
                                      use_double=use_double, **kwargs)
    # Serial
    else:
        T = Delaunay(pts, use_double=use_double, periodic=periodic,
                     left_edge=left_edge, right_edge=right_edge, **kwargs)
    return T


def voronoi_volumes(pts, left_edge=None, right_edge=None, periodic=False,
                    use_double=False, nproc=0, dd_method='kdtree',
                    dd_kwargs={}, limit_mem=False, **kwargs):
    r"""Volume of voronoi cells for each point.

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
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        nproc (int, optional): The number of MPI processes that should be
            spawned. If <2, no processes are spawned. Defaults to 0.
        dd_method (str, optional): String specifier for a domain
            decomposition method. See :meth:`cgal4py.domain_decomp.leaves`
            for available values. Defaults to 'kdtree'.
        dd_kwargs (dict, optional): Dictionary of keyword arguments for the
            selected domain decomposition method. Defaults to empty dict.
        limit_mem (bool, optional): If True, memory usage is limited by
            writing things to file at a cost to performance. Defaults to
            False.
        \*\*kwargs: Additiona keyword arguments are passed to the appropriate
            class for constructuing the triangulation.

    Returns:
        np.ndarray of float64: (n,) array of n voronoi cell mD volumes. A value
            of -1 indicates the cell is infinite.

    Raises:
        ValueError: If `pts` is not a 2D array.
        ValueError: If `left_edge` is not a 1D array with `pts.shape[1]`
            elements.
        ValueError: If `right_edge` is not a 1D array with `pts.shape[1]`
            elements.

    """
    # Check input
    if (pts.ndim != 2):
        raise ValueError("pts must be a 2D array of coordinates")
    ndim = pts.shape[1]
    if left_edge is None:
        left_edge = np.min(pts, axis=0)
    else:
        if (left_edge.ndim != 1) or (len(left_edge) != ndim):
            raise ValueError("left_edge must be a 1D array with " +
                             "{} elements.".format(ndim))
    if right_edge is None:
        right_edge = np.max(pts, axis=0)
    else:
        if (right_edge.ndim != 1) or (len(right_edge) != ndim):
            raise ValueError("right_edge must be a 1D array with " +
                             "{} elements.".format(ndim))
    # Parallel
    if nproc > 1 and FLAG_MULTIPROC:
        if (not 'nleaves' in dd_kwargs) and (not 'leafsize' in dd_kwargs):
            if limit_mem:
                dd_kwargs['nleaves'] = nproc
            else:
                dd_kwargs['nleaves'] = 2*nproc
        tree = domain_decomp.tree(dd_method, pts, left_edge, right_edge,
                                  periodic, **dd_kwargs)
        vols = parallel.ParallelVoronoiVolumes(pts, tree, nproc,
                                               limit_mem=limit_mem,
                                               use_double=use_double, **kwargs)
    # Serial
    else:
        T = Delaunay(pts, use_double=use_double, periodic=periodic,
                     left_edge=left_edge, right_edge=right_edge, **kwargs)
        vols = T.voronoi_volumes()
    return vols

# Must go here to support tests of triangulate & voronoi_volumes
from cgal4py import tests  # noqa: E402

__all__ = ["triangulate", "voronoi_volumes", "delaunay", "plot",
           "domain_decomp", "tests"]
if FLAG_MULTIPROC:
    __all__.append("parallel")
