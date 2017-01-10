from cgal4py.tests.test_cgal4py import make_points
from cgal4py.delaunay import _get_Delaunay
from cgal4py import triangulate, voronoi_volumes
import sys
from datetime import datetime

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

limit_mem = 32
use_double = True

periodic = True
npts = int(1e9)
ndim = 3
test_result = False
task = 'vols'
unique_str = datetime.today().strftime("%Y%j%H%M%S")

ParallelDelaunay = _get_Delaunay(ndim, bit64=use_double, parallel=True,
                                 comm=comm)

if rank == 0:
    pts, le, re = make_points(npts, ndim)
    pts2, le, re = make_points(10, ndim)
else:
    pts = None
    le = None
    re = None
    pts2 = None

TP = ParallelDelaunay(le, re, periodic=periodic, limit_mem=limit_mem,
                      unique_str=unique_str)
TP.insert(pts)
# TP.insert(pts2)
if rank == 0:
    print("Consolidating")
if task == 'tess':
    T_new = TP.consolidate_tess()
else:
    V_new = TP.consolidate_vols()
if rank == 0:
    if test_result:
        if task == 'tess':
            T_old = triangulate(pts, le, re, nproc=1)#size)
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
        else:
            V_old = voronoi_volumes(pts, le, re, nproc=1)
            assert(np.allclose(V_new, V_old))
    print("Success!!")
