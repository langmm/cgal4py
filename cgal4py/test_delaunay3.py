from nose import with_setup
import numpy as np
import os
from delaunay3 import Delaunay3

pts = np.array([[ 0,  0,  0],
                [-1, -1, -1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1,  1,  1],
                [ 1, -1, -1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [ 1,  1,  1]], 'float64')
pts_dup = np.concatenate([pts, np.reshape(pts[0,:],(1,pts.shape[1]))])
ncells = 24


def test_create():
    T = Delaunay3()

def test_insert():
    T = Delaunay3()
    T.insert(pts)

def test_insert_dup():
    T = Delaunay3()
    T.insert(pts_dup)

def test_num_verts():
    T = Delaunay3()
    T.insert(pts)
    assert(T.num_verts() == pts.shape[0])

def test_num_verts_dup():
    T = Delaunay3()
    T.insert(pts_dup)
    assert(T.num_verts() == pts.shape[0])

def test_num_cells():
    T = Delaunay3()
    T.insert(pts)
    assert(T.num_cells() == ncells)
    
def test_num_cells_dup():
    T = Delaunay3()
    T.insert(pts_dup)
    print T.num_cells(), ncells
    assert(T.num_cells() == ncells)

def test_verts():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_verts():
        assert(np.allclose(v.point, pts[v.index,:]))
        count += 1
    assert(count == T.num_verts())

def test_io():
    fname = 'test_io2348_3.dat'
    Tout = Delaunay3()
    Tout.insert(pts)
    Tout.write_to_file(fname)
    Tin = Delaunay3()
    Tin.read_from_file(fname)
    assert(Tout.num_verts() == Tin.num_verts())
    assert(Tout.num_cells() == Tin.num_cells())
    os.remove(fname)
