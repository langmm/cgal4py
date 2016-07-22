from nose import with_setup
import numpy as np
import os
from delaunay2 import Delaunay2

# TODO:
# - value test circumcenter

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
nedges_fin = 17
nedges_inf = 4
nedges = nedges_fin + nedges_inf
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

def test_num_edges():
    T = Delaunay2()
    T.insert(pts)
    print(T.num_finite_edges, T.num_infinite_edges)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)

def test_num_edges_dup():
    T = Delaunay2()
    T.insert(pts_dup)
    print(T.num_finite_edges, T.num_infinite_edges)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)

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

def test_all_verts():
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

def test_finite_verts():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.finite_verts:
        assert((not v.is_infinite()))
        count += 1
    assert(count == T.num_finite_verts)

def test_all_edges():
    T = Delaunay2()
    T.insert(pts)
    count_fin = count_inf = 0
    for e in T.all_edges:
        if e.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    print(count_fin, count_inf)
    count = count_fin + count_inf
    assert(count_fin == T.num_finite_edges)
    assert(count_inf == T.num_infinite_edges)
    assert(count == T.num_edges)

def test_finite_edges():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for e in T.finite_edges:
        assert((not e.is_infinite()))
        count += 1
    print(count)
    assert(count == T.num_finite_edges)

def test_all_cells():
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

def test_finite_cells():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for c in T.finite_cells:
        assert((not c.is_infinite()))
        count += 1
    assert(count == T.num_finite_cells)

def test_clear():
    T = Delaunay2()
    T.insert(pts)
    T.clear()
    print(T.num_finite_verts, T.num_cells)
    assert(T.num_finite_verts == 0)
    assert(T.num_cells == 0)

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

def test_get_vertex():
    T = Delaunay2()
    T.insert(pts)
    for i in range(nverts_fin):
        v = T.get_vertex(i)
        assert(np.allclose(v.point, pts[i,:]))

def test_remove():
    T = Delaunay2()
    T.insert(pts)
    v = T.get_vertex(0)
    T.remove(v)
    assert(T.num_verts == (nverts-1))

def test_vertex():
    pass
def test_edge():
    pass
def test_cell():
    T = Delaunay2()
    T.insert(pts)
    for c in T.all_cells:
        v1 = c.vertex(0)
        v2 = c.vertex(1)
        v3 = c.vertex(2)
        print(c.has_vertex(v1))
        print(c.has_vertex(v1, return_index = True))
        print(c.index_vertex(v1))

        c.reset_vertices()
        c.set_vertex(0, v1)
        c.set_vertices(v3, v2, v1)

        n1 = c.neighbor(0)
        n2 = c.neighbor(1)
        n3 = c.neighbor(2)
        print(c.has_neighbor(n1))
        print(c.has_neighbor(n1, return_index = True))
        print(c.index_neighbor(n1))

        c.reset_neighbors()
        c.set_neighbor(0, n1)
        c.set_neighbors(n3, n2, n1)

        c.reorient()
        c.ccw_permute()
        c.cw_permute()
        print(c.dimension)
        print(c.circumcenter)

def test_move():
    T = Delaunay2()
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(2,'float64')
    v = T.move(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(T.num_verts == (nverts-1))

def test_move_if_no_collision():
    T = Delaunay2()
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(2,'float64')
    v = T.move_if_no_collision(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move_if_no_collision(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v1.point, pts[1,:]))
    assert(T.num_verts == nverts)

def test_flip():
    T = Delaunay2()
    T.insert(pts)
    for c in T.all_cells:
        out = T.flip(c, 0)
        assert(out == True)
    assert(T.num_edges == nedges)

def test_flippable():
    T = Delaunay2()
    T.insert(pts)
    for c in T.all_cells:
        T.flip(c, 0)
    assert(T.num_edges == nedges)

def test_vert_incident_verts():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_vertices():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 42)

def test_vert_incident_edges():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 42)

def test_vert_incident_cells():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_cells():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 42)

def test_edge_incident_verts():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 42)

def test_edge_incident_edges():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 156)

def test_edge_incident_cells():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 100)

def test_nearest_vertex():
    idx_test = 7
    T = Delaunay2()
    T.insert(pts)
    v = T.nearest_vertex(pts[idx_test,:]-0.1)
    assert(v.index == idx_test)

def test_dual_volume():
    T = Delaunay2()
    T.insert(pts)
    for v in T.finite_verts:
        print(v.index,v.volume)
        if v.index >= 4:
            assert(np.isclose(v.volume, -1.0))

def test_edge_length():
    T = Delaunay2()
    T.insert(pts)
    for e in T.all_edges:
        if e.is_infinite():
            assert(np.isclose(e.length, -1.0))
        else:
            l = np.sqrt(np.sum((pts[e.vertex1.index,:]-pts[e.vertex2.index,:])**2.0))
            assert(np.isclose(e.length, l))
