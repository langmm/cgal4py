import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
from cgal4py import Triangulate
from test_delaunay2 import pts as pts2
from test_delaunay3 import pts as pts3

# TODO: Testing of parallel code with coverage

def test_parallel_Triangulate():
    T2 = Triangulate(pts2, dd_kwargs={'leafsize':2}, nproc=5)
    T3 = Triangulate(pts3, dd_kwargs={'leafsize':2}, nproc=5)
    T2 = Triangulate(pts2, dd_kwargs={'leafsize':2}, nproc=5, periodic=True)
    T3 = Triangulate(pts3, dd_kwargs={'leafsize':2}, nproc=5, periodic=True)
