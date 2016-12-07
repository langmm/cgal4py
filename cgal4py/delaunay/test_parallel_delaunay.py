from cgal4py.tests.test_cgal4py import make_points
from cgal4py.delaunay.parallel_delaunay import ParallelDelaunay
from cgal4py import triangulate

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

periodic = False
npts = int(1e4)
ndim = 3

if rank == 0:
    pts, le, re = make_points(npts, ndim)
    pts2, le, re = make_points(10, ndim)
else:
    pts = np.empty((0,ndim), 'float64')
    le = np.empty(0, 'float64')
    re = np.empty(0, 'float64')
    pts2 = np.empty((0,ndim), 'float64')
    
TP = ParallelDelaunay(le, re, periodic=periodic)
TP.insert(pts)
# TP.insert(pts2)
T_new = TP.consolidate_tess()
if rank == 0:
    T_old = triangulate(pts, le, re, nproc=size)
    c_old, n_old, inf_old = T_old.serialize(sort=True)
    c_new, n_new, inf_new = T_new.serialize(sort=True)
    try:
        assert(np.all(c_old == c_new))
        assert(np.all(n_old == n_new))
        assert(T_new.is_equivalent(T_old))
    except:
        for name, T in zip(['New','Old'],[T_new, T_old]):
            print(name)
            print('\t verts: {}, {}, {}'.format(
                T.num_verts, T.num_finite_verts, T.num_infinite_verts))
            print('\t cells: {}, {}, {}'.format(
                T.num_cells, T.num_finite_cells, T.num_infinite_cells))
            print('\t edges: {}, {}, {}'.format(
                T.num_edges, T.num_finite_edges, T.num_infinite_edges))
            if ndim == 3:
                print('\t facets: {}, {}, {}'.format(
                    T.num_facets, T.num_finite_facets,
                    T.num_infinite_facets))
        raise
