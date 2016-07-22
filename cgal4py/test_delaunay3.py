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
nverts_fin = pts.shape[0]
nverts_inf = 1
nverts = nverts_fin + nverts_inf
nedges_fin = 26
nedges_inf = 8
nedges = nedges_fin + nedges_inf
ncells_fin = 12
ncells_inf = 12
ncells = ncells_fin + ncells_inf
# vert_incident_cells = [12, 

def test_create():
    T = Delaunay3()

def test_insert():
    T = Delaunay3()
    T.insert(pts)

def test_insert_dup():
    T = Delaunay3()
    T.insert(pts_dup)

def test_get_vertex():
    T = Delaunay3()
    T.insert(pts)
    for i in range(nverts_fin):
        v = T.get_vertex(i)
        assert(np.allclose(v.point, pts[i,:]))

def test_remove():
    T = Delaunay3()
    T.insert(pts)
    v = T.get_vertex(0)
    T.remove(v)
    assert(T.num_verts == (nverts-1))

def test_move():
    T = Delaunay3()
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(3,'float64')
    v = T.move(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(T.num_verts == (nverts-1))

def test_move_if_no_collision():
    T = Delaunay3()
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(3,'float64')
    v = T.move_if_no_collision(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move_if_no_collision(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v1.point, pts[1,:]))
    assert(T.num_verts == nverts)

def test_num_verts():
    T = Delaunay3()
    T.insert(pts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)

def test_num_verts_dup():
    T = Delaunay3()
    T.insert(pts_dup)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)

def test_num_edges():
    T = Delaunay3()
    T.insert(pts)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)

def test_num_edges_dup():
    T = Delaunay3()
    T.insert(pts_dup)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)

def test_num_cells():
    T = Delaunay3()
    T.insert(pts)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)
    
def test_num_cells_dup():
    T = Delaunay3()
    T.insert(pts_dup)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)

def test_all_verts():
    T = Delaunay3()
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

def test_finite_verts():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.finite_verts:
        assert((not v.is_infinite()))
        count += 1
    assert(count == T.num_finite_verts)

def test_all_edges():
    T = Delaunay3()
    T.insert(pts)
    count_fin = count_inf = 0
    for e in T.all_edges:
        if e.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    print(count_fin, count_inf)
    assert(count_fin == T.num_finite_edges)
    assert(count_inf == T.num_infinite_edges)
    assert(count == T.num_edges)

def test_finite_edges():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for e in T.finite_edges:
        assert((not e.is_infinite()))
        count += 1
    print(count)
    assert(count == T.num_finite_edges)

def test_all_cells():
    T = Delaunay3()
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

def test_finite_cells():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for c in T.finite_cells:
        assert((not c.is_infinite()))
        count += 1
    assert(count == T.num_finite_cells)

def test_io():
    fname = 'test_io2348_3.dat'
    Tout = Delaunay3()
    Tout.insert(pts)
    Tout.write_to_file(fname)
    Tin = Delaunay3()
    Tin.read_from_file(fname)
    assert(Tout.num_verts == Tin.num_verts)
    assert(Tout.num_cells == Tin.num_cells)
    os.remove(fname)

def test_circumcenter():
    # TODO: value test
    T = Delaunay3()
    T.insert(pts)
    for c in T.all_cells:
        out = c.circumcenter()

def test_vert_incident_cells():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_cells():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 96)

def test_vert_incident_verts():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_vertices():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 68)
    
def test_nearest_vertex():
    idx_test = 8
    T = Delaunay3()
    T.insert(pts)
    v = T.nearest_vertex(pts[idx_test,:]-0.1)
    assert(v.index == idx_test)

def test_dual_volume():
    T = Delaunay3()
    T.insert(pts)
    for v in T.finite_verts:
        print(v.index,v.volume)
        if v.index == 0:
            assert(np.isclose(v.volume, 4.5))
        else:
            assert(np.isclose(v.volume, -1.0))

def test_edge_length():
    T = Delaunay3()
    T.insert(pts)
    for e in T.all_edges:
        if e.is_infinite():
            assert(np.isclose(e.length, -1.0))
        else:
            l = np.sqrt(np.sum((pts[e.vertex1.index,:]-pts[e.vertex2.index,:])**2.0))
            assert(np.isclose(e.length, l))
