r"""Testing for the 3D Periodic Delaunay extension.

.. todo:
   * answer test circumcenter
   * answer test ability to flip
   * clarify equality between facets defined using different cells

"""
import numpy as np
import os
from cgal4py.delaunay import PeriodicDelaunay3 as Delaunay3


left_edge = -2*np.ones(3, 'float64')
right_edge = 2*np.ones(3, 'float64')
pts = np.array([[+0,  0,  0],
                [-1, -1, -1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1,  1,  1],
                [+1, -1, -1],
                [+1, -1,  1],
                [+1,  1, -1],
                [+1,  1,  1.0000001]], 'float64')
pts_dup = np.concatenate([pts, np.reshape(pts[0, :], (1, pts.shape[1]))])
nsheets = np.array([3, 3, 3], 'int32')
nverts_fin = pts.shape[0]
nverts_inf = 0
nverts = nverts_fin + nverts_inf
nedges_fin = 66
nedges_inf = 0
nedges = nedges_fin + nedges_inf
nfacets_fin = 114
nfacets_inf = 0
nfacets = nfacets_fin + nfacets_inf
ncells_fin = 57
ncells_inf = 0
ncells = ncells_fin + ncells_inf


def test_create():
    T = Delaunay3(left_edge, right_edge)
    del T


def test_insert():
    # without duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    assert(T.is_valid())
    # with duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts_dup)
    assert(T.is_valid())


def test_equal():
    T1 = Delaunay3(left_edge, right_edge)
    T1.insert(pts)
    T2 = Delaunay3(left_edge, right_edge)
    T2.insert(pts)
    assert(T1.is_equivalent(T2))


def test_num_sheets():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    print(T.num_sheets, T.num_sheets_total)
    assert(np.allclose(T.num_sheets, nsheets))
    assert(T.num_sheets_total == nsheets.prod())


def test_num_verts():
    # without duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    print(T.num_finite_verts, T.num_infinite_verts, T.num_verts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)
    assert(T.num_stored_verts == T.num_sheets_total*nverts)
    # with duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts_dup)
    print(T.num_finite_verts, T.num_infinite_verts, T.num_verts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)
    assert(T.num_stored_verts == T.num_sheets_total*nverts)


def test_num_edges():
    # without duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    print(T.num_finite_edges, T.num_infinite_edges, T.num_edges)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)
    assert(T.num_stored_edges == T.num_sheets_total*nedges)
    # with duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts_dup)
    print(T.num_finite_edges, T.num_infinite_edges, T.num_edges)
    assert(T.num_finite_edges == nedges_fin)
    assert(T.num_infinite_edges == nedges_inf)
    assert(T.num_edges == nedges)
    assert(T.num_stored_edges == T.num_sheets_total*nedges)


def test_num_facets():
    # without duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    print(T.num_finite_facets, T.num_infinite_facets, T.num_facets)
    assert(T.num_finite_facets == nfacets_fin)
    assert(T.num_infinite_facets == nfacets_inf)
    assert(T.num_facets == nfacets)
    assert(T.num_stored_facets == T.num_sheets_total*nfacets)
    # with duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts_dup)
    print(T.num_finite_facets, T.num_infinite_facets, T.num_facets)
    assert(T.num_finite_facets == nfacets_fin)
    assert(T.num_infinite_facets == nfacets_inf)
    assert(T.num_facets == nfacets)
    assert(T.num_stored_facets == T.num_sheets_total*nfacets)


def test_num_cells():
    # without duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    print(T.num_finite_cells, T.num_infinite_cells, T.num_cells)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)
    assert(T.num_stored_cells == T.num_sheets_total*ncells)
    # with duplicates
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts_dup)
    print(T.num_finite_cells, T.num_infinite_cells, T.num_cells)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)
    assert(T.num_stored_cells == T.num_sheets_total*ncells)


def test_all_verts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count_fin = count_inf = 0
    for v in T.all_verts:
        if v.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    assert(count_fin == T.num_stored_verts)
    assert(count_inf == T.num_infinite_verts)
    assert(count == T.num_stored_verts)


def test_finite_verts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.finite_verts:
        assert((not v.is_infinite()))
        count += 1
    assert(count == T.num_stored_verts)


def test_all_edges():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count_fin = count_inf = 0
    for e in T.all_edges:
        if e.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    print(count_fin, count_inf)
    assert(count_fin == T.num_stored_edges)
    assert(count_inf == T.num_infinite_edges)
    assert(count == T.num_stored_edges)


def test_finite_edges():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for e in T.finite_edges:
        assert((not e.is_infinite()))
        count += 1
    print(count)
    assert(count == T.num_stored_edges)


def test_all_cells():
    T = Delaunay3(left_edge, right_edge)
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
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for c in T.finite_cells:
        assert((not c.is_infinite()))
        count += 1
    assert(count == T.num_stored_cells)


def test_get_vertex():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    for i in range(nverts_fin):
        v = T.get_vertex(i)
        assert(np.allclose(v.periodic_point, pts[i, :]))


def test_locate():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    for c in T.finite_cells:
        if c.has_offset:
            continue
        p = c.center
        print(c, p, T.locate(p))
        assert(c == T.locate(p))
        assert(c == T.locate(p, c))
        assert(c.vertex(0) == T.locate(c.vertex(0).point))
        assert(c.vertex(0) == T.locate(c.vertex(0).point, c))
        # TODO: check that is_equivalent is working
        # assert(c.facet(0).is_equivalent(T.locate(c.facet(0).edge(0).center)))
        # assert(c.facet(0).is_equivalent(
        #     T.locate(c.facet(0).edge(0).center, c)))
        # assert(c.facet(0) == T.locate(c.facet(0).center))
        # assert(c.facet(0) == T.locate(c.facet(0).center, c))
        # assert(c.facet(0).edge(0).is_equivalent(
        #     T.locate(c.facet(0).edge(0).center)))
        # assert(c.facet(0).edge(0).is_equivalent(
        #     T.locate(c.facet(0).edge(0).center, c)))
        # assert(c.facet(0).edge(0) == T.locate(c.facet(0).edge(0).center))
        # assert(c.facet(0).edge(0) == T.locate(c.facet(0).edge(0).center, c))
        break


def test_remove():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    T.remove(v)
    assert(T.num_verts == (nverts-1))


def test_clear():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    T.clear()
    print(T.num_finite_verts, T.num_cells)
    assert(T.num_finite_verts == 0)
    assert(T.num_cells == 0)


def test_is_edge():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    assert(T.is_edge(T.get_vertex(0), T.get_vertex(1)))
    assert(not T.is_edge(T.get_vertex(1), T.get_vertex(nverts_fin-1)))


def test_is_facet():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    assert(T.is_facet(T.get_vertex(0), T.get_vertex(1), T.get_vertex(3)))
    assert(not T.is_facet(T.get_vertex(0), T.get_vertex(1),
                          T.get_vertex(nverts_fin-1)))


def test_is_cell():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    assert(T.is_cell(T.get_vertex(0), T.get_vertex(1),
                     T.get_vertex(3), T.get_vertex(5)))
    assert(not T.is_cell(T.get_vertex(0), T.get_vertex(1),
                         T.get_vertex(3), T.get_vertex(nverts_fin-1)))


def test_vert():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    vold = None
    for v in T.all_verts:
        idx = v.index
        pnt = v.point
        per = v.periodic_point
        off = v.periodic_offset
        vol = v.dual_volume
        print(pts[idx, :])
        print(v, idx, vol, pnt, per, off)
        assert(v == v)
        if vold is not None:
            assert(v != vold)
        if v.is_infinite():
            assert(idx == np.iinfo(np.uint32).max)
            assert(np.isinf(per).all())
            assert(np.isclose(vol, -1.0))
        else:
            assert(np.allclose(per, pts[idx, :]))
            if idx == 0:
                assert(np.isclose(vol, 4.5))
            c = v.cell
            v.set_cell(c)
            v.set_point(per)
        vold = v


def test_edge():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    eold = None
    count_wrap = 0
    for e in T.all_edges:
        off1 = e.periodic_offset(0)
        off2 = e.periodic_offset(1)
        zer = np.zeros(off1.shape, off1.dtype)
        # two = 3*np.ones(off1.shape, off1.dtype)
        # one = off1[0]*np.ones(off1.shape, off1.dtype)
        if np.allclose(off1, zer) or np.allclose(off2, zer):
            assert(e.is_unique)
        # print(e.has_offset, e.spans_wrap, off1, off2)
        if e.is_unique:
            count_wrap += 1
            print(e.is_unique, e.has_offset, off1, off2,
                  e.vertex1.periodic_offset, e.vertex2.periodic_offset)
        if not e.has_offset:
            assert(not e.spans_wrap)
            assert(e.is_unique)
        else:
            # print(off1,off2,e.spans_wrap)
            assert(e.spans_wrap == (max(abs(off1 % 3-off2 % 3)) > 1))
        v1 = e.vertex(0)
        v2 = e.vertex(1)
        assert(v1 == e.vertex1)
        assert(v2 == e.vertex2)
        c = e.cell
        i1 = e.ind1
        i2 = e.ind2
        elen = e.length
        inf = e.is_infinite()
        gab = e.is_Gabriel()
        # print(e, v1.index, v2.index, elen, inf, gab)
        assert(e == e)
        assert(e.is_equivalent(e))
        if eold is not None:
            assert(e != eold)
        p1 = e.center
        p2 = v1.point
        if e.is_infinite():
            assert(np.isclose(elen, -1.0))
            assert(np.isinf(e.center).all())
        else:
            p1 = e.point(0)
            p2 = e.point(1)
            l = np.sqrt(np.sum((p1 - p2)**2.0))
            assert(np.isclose(elen, l))
        eold = e
        del c, i1, i2, inf, gab
    print(count_wrap, nedges_fin)
    assert(count_wrap == nedges_fin)


def test_facet():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    fold = None
    for f in T.all_facets:
        v1 = f.vertex(0)
        v2 = f.vertex(1)
        v3 = f.vertex(2)
        e1 = f.edge(0)
        e2 = f.edge(1)
        e3 = f.edge(2)
        c = f.cell
        i = f.ind
        inf = f.is_infinite()
        gab = f.is_Gabriel()
        print(f, v1.index, v2.index, v3.index, i, inf, gab, f.center)
        assert(f == f)
        assert(f.is_equivalent(f))
        if fold is not None:
            assert(f != fold)
        if f.is_infinite():
            assert(np.isinf(f.center).all())
        fold = f
        del e1, e2, e3, c


def test_cell():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    cold = None
    for c in T.all_cells:
        print(c, c.circumcenter, c.center)
        assert(c == c)
        if cold is not None:
            assert(c != cold)

        f1 = c.facet(0)
        f2 = c.facet(1)
        f3 = c.facet(2)
        f4 = c.facet(3)
        del f1, f2, f3, f4

        v1 = c.vertex(0)
        v2 = c.vertex(1)
        v3 = c.vertex(2)
        v4 = c.vertex(3)
        assert(c.has_vertex(v1))
        assert(c.has_vertex(v1, return_index=True) == 0)
        assert(c.ind_vertex(v1) == 0)

        c.reset_vertices()
        c.set_vertex(0, v1)
        c.set_vertices(v4, v3, v2, v1)

        n1 = c.neighbor(0)
        n2 = c.neighbor(1)
        n3 = c.neighbor(2)
        n4 = c.neighbor(3)
        assert(c.has_neighbor(n1))
        assert(c.has_neighbor(n1, return_index=True) == 0)
        assert(c.ind_neighbor(n1) == 0)

        c.reset_neighbors()
        c.set_neighbor(0, n1)
        c.set_neighbors(n4, n3, n2, n1)

        c1 = c.center
        c2 = c.periodic_circumcenter
        p1 = c.point(0)
        # p1 = v1.point
        p2 = 2*p1 - c1
        p3 = 2*p1 - c2
        if c.is_infinite():
            assert(np.isinf(c.center).all())
            assert(c.side(c1) == -1)
            assert(c.side(p1) == -1)
            assert(np.isinf(c2).all())
            assert(c.side_of_sphere(c2) == -1)
            assert(c.side_of_sphere(p1) == -1)
        else:
            print(c.is_unique)
            print(c.side(c1), c1)
            print(c.side(p1), p1)
            print(c.side(p2), p2)
            print(c.side_of_sphere(c2), c2)
            print(c.side_of_sphere(p1), p1)
            print(c.side_of_sphere(p3), p3)
            # TODO: fix this for wrapped cells
            # assert(c.side(c1) == -1)
            # assert(c.side(p1) == 0)
            # assert(c.side(p2) == 1)
            # assert(c.side_of_sphere(c2) == -1)
            # assert(c.side_of_sphere(p1) == 0)
            # assert(c.side_of_sphere(p3) == 1)
        cold = c


def test_move():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(3, 'float64')
    v = T.move(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(T.num_verts == (nverts-1))


def test_move_if_no_collision():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(3, 'float64')
    v = T.move_if_no_collision(v0, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v0.point, new_pos))
    v1 = T.get_vertex(1)
    v = T.move_if_no_collision(v1, new_pos)
    assert(np.allclose(v.point, new_pos))
    assert(np.allclose(v1.point, pts[1, :]))
    assert(T.num_verts == nverts)


def test_io():
    fname = 'test_io2348_3.dat'
    Tout = Delaunay3(left_edge, right_edge)
    Tout.insert(pts)
    Tout.write_to_file(fname)
    Tin = Delaunay3()
    Tin.read_from_file(fname)
    assert(Tout.num_verts == Tin.num_verts)
    assert(Tout.num_cells == Tin.num_cells)
    os.remove(fname)


def test_vert_incident_verts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_vertices():
            c0 += 1
            count += 1
        x = v.incident_vertices()[0]
        print(v.index, c0, x)
    print(count, 2*T.num_stored_edges)
    assert(count == 2*T.num_stored_edges)


def test_vert_incident_edges():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        x = v.incident_edges()[0]
        print(v.index, c0, x)
    print(count, 2*T.num_stored_edges)
    assert(count == 2*T.num_stored_edges)


def test_vert_incident_facets():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        x = v.incident_facets()[0]
        print(v.index, c0, x)
    print(count, 3*T.num_stored_facets)
    assert(count == 3*T.num_stored_facets)


def test_vert_incident_cells():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_cells():
            c0 += 1
            count += 1
        print(v.index, c0)
    print(count, 4*T.num_stored_cells)
    assert(count == 4*T.num_stored_cells)


def test_edge_incident_verts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count, 2*T.num_stored_edges)
    assert(count == 2*T.num_stored_edges)


def test_edge_incident_edges():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 50814)


def test_edge_incident_facets():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        print(c0)
    print(count, 3*T.num_stored_facets)
    assert(count == 3*T.num_stored_facets)


def test_edge_incident_cells():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_edges:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count, 3*T.num_stored_facets)
    assert(count == 3*T.num_stored_facets)


def test_facet_incident_verts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count, 3*T.num_stored_facets)
    assert(count == 3*T.num_stored_facets)


def test_facet_incident_edges():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count, 3*T.num_stored_facets)
    assert(count == 3*T.num_stored_facets)


def test_facet_incident_facets():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 41418)


def test_facet_incident_cells():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_facets:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count, 2*T.num_stored_facets)
    assert(count == 2*T.num_stored_facets)


def test_cell_incident_verts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count, 4*T.num_stored_cells)
    assert(count == 4*T.num_stored_cells)


def test_cell_incident_edges():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count, 6*T.num_stored_cells)
    assert(count == 6*T.num_stored_cells)


def test_cell_incident_facets():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_facets():
            c0 += 1
            count += 1
        print(c0)
    print(count, 4*T.num_stored_cells)
    assert(count == 4*T.num_stored_cells)


def test_cell_incident_cells():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count, 4*T.num_stored_cells)
    assert(count == 4*T.num_stored_cells)


def test_nearest_vertex():
    idx_test = 8
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v = T.nearest_vertex(pts[idx_test, :]-0.1)
    assert(v.index == idx_test)


def test_mirror():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    for e in T.all_facets:
        T.mirror_facet(e)
        break
    for c in T.all_cells:
        idx = 0
        i2 = T.mirror_index(c, idx)
        assert(c == c.neighbor(idx).neighbor(i2))
        v2 = T.mirror_vertex(c, idx)
        assert(v2 == c.neighbor(idx).vertex(i2))


def test_get_boundary_of_conflicts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    edges = T.get_boundary_of_conflicts(p, c)
    print(len(edges))


def test_get_conflicts():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    cells = T.get_conflicts(p, c)
    print(len(cells))


def test_get_conflicts_and_boundary():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    cells, edges = T.get_conflicts_and_boundary(p, c)
    print(len(cells), len(edges))


def test_vertices():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    v = T.vertices
    assert(v.shape[0] == pts.shape[0])
    assert(v.shape[1] == pts.shape[1])
    assert(np.allclose(pts, v))


def test_edges():
    T = Delaunay3(left_edge, right_edge)
    T.insert(pts)
    e = T.edges
    assert(e.shape[0] == T.num_finite_edges)
    assert(e.shape[1] == 2)
