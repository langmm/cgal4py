from nose import with_setup
import numpy as np
import os
from delaunay2 import Delaunay2

pts = np.array([[-0.4941988586954018 , -0.07594397977563715],
                [-0.06448037284989526,  0.4958248496365813 ],
                [ 0.4911154367094632 ,  0.09383830681375946],
                [-0.348353580869097  , -0.3586778257652367 ],
                [-1,     -1],
                [-1,      1],
                [ 1,     -1],
                [ 1,      1]], 'float64')
pts_dup = np.concatenate([pts, np.reshape(pts[0,:],(1,pts.shape[1]))])
nverts_fin = pts.shape[0]
nverts_inf = 1
nverts = nverts_fin + nverts_inf
ncells_fin = 10
ncells_inf = 4
ncells = ncells_fin + ncells_inf

def test_create():
    T = Delaunay2()

def test_insert():
    T = Delaunay2()
    T.insert(pts)

def test_insert_dup():
    T = Delaunay2()
    T.insert(pts_dup)

def test_num_verts():
    T = Delaunay2()
    T.insert(pts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)

def test_num_verts_dup():
    T = Delaunay2()
    T.insert(pts_dup)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)

def test_num_cells():
    T = Delaunay2()
    T.insert(pts)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)
    
def test_num_cells_dup():
    T = Delaunay2()
    T.insert(pts_dup)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)

def test_verts():
    T = Delaunay2()
    T.insert(pts)
    count_fin = count_inf = 0
    for v in T.all_verts:
        if v.is_infinite():
            count_inf += 1
        else:
            assert(np.allclose(v.point, pts[v.index,:]))
            count_fin += 1
    count = count_fin + count_inf
    assert(count_fin == T.num_finite_verts)
    assert(count_inf == T.num_infinite_verts)
    assert(count == T.num_verts)

def test_cells():
    T = Delaunay2()
    T.insert(pts)
    count_fin = count_inf = 0
    for c in T.all_cells:
        if c.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    assert(count_fin == T.num_finite_cells)
    assert(count_inf == T.num_infinite_cells)
    assert(count == T.num_cells)

def test_io():
    fname = 'test_io2348_2.dat'
    Tout = Delaunay2()
    Tout.insert(pts)
    Tout.write_to_file(fname)
    Tin = Delaunay2()
    Tin.read_from_file(fname)
    assert(Tout.num_verts == Tin.num_verts)
    assert(Tout.num_cells == Tin.num_cells)
    os.remove(fname)
