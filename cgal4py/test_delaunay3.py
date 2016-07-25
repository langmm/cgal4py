from nose import with_setup
import numpy as np
import os
from delaunay3 import Delaunay3

# TODO:
# - answer test circumcenter
# - answer test ability to flip


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

def test_clear():
    T = Delaunay3()
    T.insert(pts)
    T.clear()
    print(T.num_finite_verts, T.num_cells)
    assert(T.num_finite_verts == 0)
    assert(T.num_cells == 0)

def test_vert():
    T = Delaunay3()
    T.insert(pts)
    for v in T.finite_verts:
        idx = v.index
        pnt = v.point
        vol = v.dual_volume
        print(idx,pnt,vol)
        if idx == 0:
            assert(np.isclose(vol, 4.5))
        else:
            assert(np.isclose(vol, -1.0))
        c = v.cell
        v.set_cell(c)
        v.set_point(pnt)

def test_edge():
    T = Delaunay3()
    T.insert(pts)
    for e in T.all_edges:
        v1 = e.vertex(0)
        v2 = e.vertex(1)
        assert(v1 == e.vertex1)
        assert(v2 == e.vertex2)
        c = e.cell
        i1 = e.ind1
        i2 = e.ind2
        elen = e.length
        if e.is_infinite():
            assert(np.isclose(elen, -1.0))
        else:
            print(v1.index, v2.index, elen)
            l = np.sqrt(np.sum((pts[v1.index,:]-pts[v2.index,:])**2.0))
            assert(np.isclose(elen, l))

def test_facet():
    T = Delaunay3()
    T.insert(pts)
    for f in T.all_facets:
        v1 = f.vertex(0)
        v2 = f.vertex(1)
        v3 = f.vertex(2)
        c = f.cell
        i = f.ind

def test_cell():
    T = Delaunay3()
    T.insert(pts)
    for c in T.all_cells:
        v1 = c.vertex(0)
        v2 = c.vertex(1)
        v3 = c.vertex(2)
        v4 = c.vertex(3)
        print(c.has_vertex(v1))
        print(c.has_vertex(v1, return_index = True))
        print(c.ind_vertex(v1))

        c.reset_vertices()
        c.set_vertex(0, v1)
        c.set_vertices(v4, v3, v2, v1)

        n1 = c.neighbor(0)
        n2 = c.neighbor(1)
        n3 = c.neighbor(2)
        n4 = c.neighbor(3)
        print(c.has_neighbor(n1))
        print(c.has_neighbor(n1, return_index = True))
        print(c.ind_neighbor(n1))

        c.reset_neighbors()
        c.set_neighbor(0, n1)
        c.set_neighbors(n4, n3, n2, n1)

        print(c.circumcenter)

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

def test_flip():
    T = Delaunay3()
    T.insert(pts)
    for c in T.all_cells:
        out = T.flip(c, 0)
        # assert(out == True)
    assert(T.num_edges == nedges)

def test_flippable():
    T = Delaunay3()
    T.insert(pts)
    for c in T.all_cells:
        T.flip(c, 0)
    assert(T.num_edges == nedges)

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
    
def test_vert_incident_edges():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 68)

def test_vert_incident_facets():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 144)

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

def test_edge_incident_verts():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 68)

def test_edge_incident_edges():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 404)

def test_edge_incident_facets():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 144)

def test_edge_incident_cells():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 144)

def test_facet_incident_verts():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 144)

def test_facet_incident_edges():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 144)

def test_facet_incident_facets():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 450)

def test_facet_incident_cells():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 96)

def test_cell_incident_verts():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 96)

def test_cell_incident_edges():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 144)

def test_cell_incident_facets():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 96)

def test_cell_incident_cells():
    T = Delaunay3()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 72)

def test_nearest_vertex():
    idx_test = 8
    T = Delaunay3()
    T.insert(pts)
    v = T.nearest_vertex(pts[idx_test,:]-0.1)
    assert(v.index == idx_test)

def test_vertices():
    T = Delaunay3()
    T.insert(pts)
    v = T.vertices
    assert(v.shape[0] == pts.shape[0])
    assert(v.shape[1] == pts.shape[1])
    assert(np.allclose(pts, v))

def test_edges():
    T = Delaunay3()
    T.insert(pts)
    e = T.edges
    assert(e.shape[0] == T.num_finite_edges)
    assert(e.shape[1] == 2)

def test_plot():
    fname_test = "test_plot3D.png"
    T = Delaunay3()
    T.insert(pts)
    axs = T.plot(plotfile=fname_test)
    os.remove(fname_test)
    T.plot(axs=axs, title='Test')
