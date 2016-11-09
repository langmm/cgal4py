r"""Tests for the 2D Periodic Delaunay Extension.

.. todo:
   * value test circumcenter
   * test includes_edge

"""
import numpy as np
import os
from cgal4py.delaunay import PeriodicDelaunay2 as Delaunay2

left_edge = -2*np.ones(2, 'float64')
right_edge = 2*np.ones(2, 'float64')
pts = np.array([[-0.49419885869540180, -0.07594397977563715],
                [-0.06448037284989526,  0.49582484963658130],
                [+0.49111543670946320,  0.09383830681375946],
                [-0.34835358086909700, -0.35867782576523670],
                [-1,     -1],
                [-1,      1],
                [+1,     -1],
                [+1,      1]], 'float64')
pts_dup = np.concatenate([pts, np.reshape(pts[0, :], (1, pts.shape[1]))])
nsheets = np.array([3, 3], 'int32')
nverts_fin = pts.shape[0]
nverts_inf = 0
nverts = nverts_fin + nverts_inf
nedges_fin = 24  # 17 in non-periodic
nedges_inf = 0   # 4 in non-periodic
nedges = nedges_fin + nedges_inf
ncells_fin = 16  # 10 in non-periodic
ncells_inf = 0   # 4 in non-periodic
ncells = ncells_fin + ncells_inf


def test_create():
    T = Delaunay2(left_edge, right_edge)
    del T


def test_insert():
    # without duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    assert(T.is_valid())
    # with duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts_dup)
    assert(T.is_valid())


def test_equal():
    T1 = Delaunay2(left_edge, right_edge)
    T1.insert(pts)
    T2 = Delaunay2(left_edge, right_edge)
    T2.insert(pts)
    assert(T1.is_equivalent(T2))


def test_num_sheets():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    print(T.num_sheets, T.num_sheets_total)
    assert(np.allclose(T.num_sheets, nsheets))
    assert(T.num_sheets_total == nsheets.prod())


def test_num_verts():
    # without duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    print(T.num_finite_verts, T.num_infinite_verts, T.num_verts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)
    assert(T.num_stored_verts == T.num_sheets_total*nverts)
    # with duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts_dup)
    print(T.num_finite_verts, T.num_infinite_verts, T.num_verts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)
    assert(T.num_stored_verts == T.num_sheets_total*nverts)


def test_num_edges():
    # without duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    print(T.num_finite_edges, T.num_infinite_edges, T.num_edges)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)
    assert(T.num_stored_edges == T.num_sheets_total*nedges)
    # with duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts_dup)
    print(T.num_finite_edges, T.num_infinite_edges, T.num_edges)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)
    assert(T.num_stored_edges == T.num_sheets_total*nedges)


def test_num_cells():
    # without duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    print(T.num_finite_cells, T.num_infinite_cells, T.num_cells)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)
    assert(T.num_stored_cells == T.num_sheets_total*ncells)
    # with duplicates
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts_dup)
    print(T.num_finite_cells, T.num_infinite_cells, T.num_cells)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)
    assert(T.num_stored_cells == T.num_sheets_total*ncells)


def test_all_verts():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count_fin = count_inf = 0
    for v in T.all_verts:
        print(v.index, v.point)
        if v.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    assert(count_fin == T.num_stored_verts)
    assert(count_inf == T.num_infinite_verts)
    assert(count == T.num_stored_verts)


def test_finite_verts():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.finite_verts:
        assert((not v.is_infinite()))
        count += 1
    assert(count == T.num_stored_verts)


def test_all_edges():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count_fin = count_inf = 0
    for e in T.all_edges:
        if e.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    assert(count_fin == T.num_stored_edges)
    assert(count_inf == T.num_infinite_edges)
    assert(count == T.num_stored_edges)


def test_finite_edges():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for e in T.finite_edges:
        assert((not e.is_infinite()))
        count += 1
    print(count)
    assert(count == T.num_stored_edges)


def test_all_cells():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count_fin = count_inf = 0
    for c in T.all_cells:
        if c.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    assert(count_fin == T.num_stored_cells)
    assert(count_inf == T.num_infinite_cells)
    assert(count == T.num_stored_cells)


def test_finite_cells():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for c in T.finite_cells:
        assert((not c.is_infinite()))
        count += 1
    assert(count == T.num_stored_cells)


def test_clear():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    T.clear()
    print(T.num_finite_verts, T.num_cells)
    assert(T.num_finite_verts == 0)
    assert(T.num_cells == 0)


def test_io():
    fname = 'test_io2348_2.dat'
    Tout = Delaunay2(left_edge, right_edge)
    Tout.insert(pts)
    Tout.write_to_file(fname)
    Tin = Delaunay2()
    Tin.read_from_file(fname)
    assert(Tout.num_verts == Tin.num_verts)
    assert(Tout.num_cells == Tin.num_cells)
    os.remove(fname)


def test_serialize():
    Tout = Delaunay2(left_edge, right_edge)
    Tout.insert(pts)
    out = Tout.serialize()
    Tin = Delaunay2()
    Tin.deserialize(pts, *out)
    print(Tout.num_cells, Tin.num_cells)
    assert(Tout.num_verts == Tin.num_verts)
    assert(Tout.num_cells == Tin.num_cells)


def test_locate():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    for c in T.finite_cells:
        p = c.center
        print('{}\n{}\n{}'.format(c, p, T.locate(p)))
        assert(c == T.locate(p, c))
        assert(c == T.locate(p))
        assert(c.vertex(0) == T.locate(c.vertex(0).point))
        assert(c.vertex(0) == T.locate(c.vertex(0).point, c))
        # assert(c.edge(0) == T.locate(c.edge(0).midpoint))
        # assert(c.edge(0) == T.locate(c.edge(0).midpoint, c))
        break


def test_get_vertex():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    for i in range(nverts_fin):
        v = T.get_vertex(i)
        assert(np.allclose(v.point, pts[i, :]))


def test_remove():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    T.remove(v)
    assert(T.num_verts == (nverts-1))


def test_is_edge():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    assert(T.is_edge(T.get_vertex(0), T.get_vertex(1)))
    assert(not T.is_edge(T.get_vertex(0), T.get_vertex(nverts_fin-1)))


def test_is_cell():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    assert(T.is_cell(T.get_vertex(0), T.get_vertex(1), T.get_vertex(3)))
    assert(not T.is_cell(T.get_vertex(0), T.get_vertex(1),
                         T.get_vertex(nverts_fin-1)))


def test_vert():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    vold = None
    for v in T.all_verts:
        pnt = v.point
        per = v.periodic_point
        idx = v.index
        vol = v.dual_volume
        print(pts[idx, :])
        print(v, idx, vol, pnt, per)
        assert(v == v)
        if vold is not None:
            assert(v != vold)
        if v.is_infinite():
            assert(np.isinf(per).all())
            assert(idx == np.iinfo(np.uint32).max)
            assert(np.isclose(vol, -1))
        else:
            assert(np.allclose(per, pts[idx, :]))
            c = v.cell
            v.set_cell(c)
            v.set_point(per)
        vold = v


def test_edge():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    eold = None
    for e in T.all_edges:
        v1 = e.vertex1
        v2 = e.vertex2
        assert(e.vertex(0) == v1)
        assert(e.vertex(1) == v2)
        elen = e.length
        print(e, v1.index, v2.index, elen, e.midpoint, e.center)
        assert(e == e)
        if eold is not None:
            assert(e != eold)
        if e.is_infinite():
            assert(np.isclose(elen, -1.0))
            assert(np.isinf(e.center).all())
        else:
            p1 = e.point(0)
            p2 = e.point(1)
            l = np.sqrt(np.sum((p1 - p2)**2.0))
            assert(np.isclose(elen, l))
        eold = e


def test_cell():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    cold = None
    for c in T.all_cells:
        print(c, c.dimension, c.circumcenter, c.center)
        assert(c == c)
        if cold is not None:
            assert(c != cold)
        v1 = c.vertex(0)
        v2 = c.vertex(1)
        v3 = c.vertex(2)
        e1 = c.edge(0)
        e2 = c.edge(1)
        e3 = c.edge(2)
        assert(c.has_vertex(v1))
        assert(c.has_vertex(v1, return_index=True) == 0)
        assert(c.ind_vertex(v1) == 0)
        del e1, e2, e3

        c.reset_vertices()
        c.set_vertex(0, v1)
        c.set_vertices(v3, v2, v1)

        n1 = c.neighbor(0)
        n2 = c.neighbor(1)
        n3 = c.neighbor(2)
        assert(c.has_neighbor(n1))
        assert(c.has_neighbor(n1, return_index=True) == 0)
        assert(c.ind_neighbor(n1) == 0)

        c.reset_neighbors()
        c.set_neighbor(0, n1)
        c.set_neighbors(n3, n2, n1)

        p = c.circumcenter + 1.5*(c.point(0) - c.circumcenter)

        def rad(point, center):
            return np.sqrt(np.sum((point - center)**2))

        def p_outside(point, center):
            r_out = rad(point, center)
            p_out = center+1.1*np.sqrt(2)*r_out
            return p_out
        for i in range(3):
            print(i)
            print('    Cell: ', c.point(i), c.periodic_point(i),
                  c.periodic_offset(i))
            print('    Vert: ', c.vertex(i).point, c.vertex(i).periodic_point,
                  c.vertex(i).periodic_offset)
        print(p, rad(p, c.circumcenter), c.side_of_circle(p))
        print(80*'-')
        # print(c.side(c.center))
        # print(c.side(c.point(0)), c.side(v1.point), c.side(v2.point))
        # print(c.side(2*c.point(0)-c.center),
        #       c.side(2*v1.point-c.center),
        #       c.side(2*v2.point-c.center),
        #       c.side(p_outside(c.point(0), c.center)))
        # print(c.side_of_circle(c.circumcenter))
        # print(c.side_of_circle(c.point(0)), c.side_of_circle(v1.point))
        print(c.side_of_circle(1.01*c.point(0)-c.circumcenter),
              1.1*c.point(0)-c.circumcenter)
        print(c.side_of_circle(1.01*v1.point-c.circumcenter),
              1.1*v1.point-c.circumcenter)
        print(c.side_of_circle(p_outside(c.point(0), c.circumcenter)),
              p_outside(c.point(0), c.circumcenter))
        if c.is_infinite():
            assert(np.isinf(c.center).all())
            assert(c.side(c.center) == -1)
            assert(c.side(v1.point) == -1)
            assert(c.side(2*v1.point-c.center) == -1)
            assert(np.isinf(c.circumcenter).all())
            assert(c.side_of_circle(c.circumcenter) == -1)
            assert(c.side_of_circle(v1.point) == -1)
            assert(c.side_of_circle(2*v1.point-c.circumcenter) == -1)
        else:
            assert(c.side(c.center) == 1)
            if np.all((v1.point % 1) == 0):
                assert(c.side(v1.point) == 0)
            assert(c.side(p_outside(c.point(0), c.center)) == -1)
            assert(c.side_of_circle(c.circumcenter) == 1)
            if np.all((c.point(0) % 1) == 0):
                assert(c.side_of_circle(c.point(0)) == 0)
            # TODO: Fix this, currently fails
            # assert(c.side_of_circle(
            #            p_outside(c.point(0), c.circumcenter)) == -1)

        c.reorient()
        c.ccw_permute()
        c.cw_permute()

        cold = c


def test_move():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(2, 'float64')
    v = T.move(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(T.num_verts == (nverts - 1))


def test_move_if_no_collision():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(2, 'float64')
    v = T.move_if_no_collision(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move_if_no_collision(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v1.point, pts[1, :]))
    assert(T.num_verts == nverts)


def test_flip():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    for c in T.all_cells:
        out = T.flip(c, 0)
        assert(out)
    print(T.num_edges, nedges)
    assert(T.num_edges == nedges)
    for e in T.all_edges:
        out = e.flip()
        assert(out)
    print(T.num_edges, nedges)


def test_flippable():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    for c in T.all_cells:
        T.flip(c, 0)
    print(T.num_edges, nedges)
    assert(T.num_edges == nedges)
    for e in T.all_edges:
        e.flip()
    print(T.num_edges, nedges)


def test_vert_incident_verts():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_vertices():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 432)


def test_vert_incident_edges():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 432)


def test_vert_incident_cells():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_cells():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count)
    assert(count == 432)


def test_edge_incident_verts():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 432)


def test_edge_incident_edges():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 2340)


def test_edge_incident_cells():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 2340)


def test_cell_incident_verts():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 432)


def test_cell_incident_edges():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 432)


def test_cell_incident_cells():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 432)


def test_nearest_vertex():
    idx_test = 7
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v = T.nearest_vertex(pts[idx_test, :]-0.1)
    assert(v.index == idx_test)


def test_mirror():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    for c in T.all_cells:
        idx = 0
        i2 = T.mirror_index(c, idx)
        assert(c == c.neighbor(idx).neighbor(i2))
        v2 = T.mirror_vertex(c, idx)
        assert(v2 == c.neighbor(idx).vertex(i2))


def test_get_boundary_of_conflicts():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    edges = T.get_boundary_of_conflicts(p, c)
    print(len(edges))


def test_get_conflicts():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    cells = T.get_conflicts(p, c)
    print(len(cells))


def test_get_conflicts_and_boundary():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    cells, edges = T.get_conflicts_and_boundary(p, c)
    print(len(cells), len(edges))


def test_vertices():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v = T.vertices
    assert(v.shape[0] == pts.shape[0])
    assert(v.shape[1] == pts.shape[1])
    assert(np.allclose(pts, v))


def test_edges():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    e = T.edges
    assert(e.shape[0] == T.num_finite_edges)
    assert(e.shape[1] == 2)


def test_boundary_points():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    out = T.boundary_points(left_edge, right_edge, False)
    del out


def test_outgoing_points():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    le = np.array([[-2, -2],
                   [-1, -1],
                   [+0,  0],
                   [+1,  1]], 'float64')
    re = np.array([[-1, -1],
                   [+0,  0],
                   [+1,  1],
                   [+2,  2]], 'float64')
    out = T.outgoing_points(le, re)
    assert(len(out) == le.shape[0])


def test_voronoi_volumes():
    T = Delaunay2(left_edge, right_edge)
    T.insert(pts)
    v = T.voronoi_volumes()
    assert(v.shape[0] == T.num_finite_verts)
    assert(np.all(v > 0))
    # TODO: answer testing
