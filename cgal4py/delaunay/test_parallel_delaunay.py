from cgal4py.tests.test_cgal4py import make_points
from cgal4py.delaunay.parallel_delaunay import ParallelDelaunay

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

periodic = False
npts = int(1e2)
ndim = 3

if rank == 0:
    pts, le, re = make_points(npts, ndim)
    pts2, le, re = make_points(10, ndim)
else:
    pts = np.empty((0,ndim), 'float64')
    le = np.empty(0, 'float64')
    re = np.empty(0, 'float64')
    pts2 = np.empty((0,ndim), 'float64')
    
T = ParallelDelaunay(le, re, periodic=periodic)
T.insert(pts)
# T.insert(pts2)
T.consolidate_tess()
