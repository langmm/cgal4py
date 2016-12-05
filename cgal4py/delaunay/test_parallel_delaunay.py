from cgal4py.tests.test_cgal4py import make_points
from cgal4py.delaunay.parallel_delaunay import ParallelDelaunay

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

periodic = False
npts = 150
ndim = 2

if rank == 0:
    pts, le, re = make_points(npts, ndim)
else:
    pts = np.empty((0,ndim), 'float64')
    le = np.empty(0, 'float64')
    re = np.empty(0, 'float64')
    
T = ParallelDelaunay()
T.run(pts, le, re, periodic=periodic)
