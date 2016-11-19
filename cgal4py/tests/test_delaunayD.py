r"""Tests for nD Delaunay Triangulation.

.. todo:
   * answer test circumcenter
   * answer test ability to flip
   * clarify equality between facets defined using different cells

"""
import numpy as np
import os
import itertools
from cgal4py.delaunay import _get_Delaunay
from nose.tools import nottest

ndim = 4
DelaunayD = _get_Delaunay(ndim, overwrite=False)

left_edge = -2*np.ones(ndim, 'float64')
right_edge = 2*np.ones(ndim, 'float64')
pts = np.array([[0 for _ in range(ndim)]]+
               [i for i in itertools.product([-1, 1], repeat=ndim)],
               'float64')
pts[-1,-1] += 0.0000001
pts_dup = np.concatenate([pts, np.reshape(pts[0, :], (1, pts.shape[1]))])
nverts_fin = pts.shape[0]
nverts_inf = 1
nverts = nverts_fin + nverts_inf
if ndim == 4:
    ncells_fin = 51
    ncells_inf = 51
    cvol = 11.33333333333
elif ndim == 5:
    ncells_fin = 260
    ncells_inf = 260
    cvol = 24.739583201584463
else:
    ncells_fin = 0
    ncells_inf = 0
    cvol = 0.0
ncells = ncells_fin + ncells_inf


@nottest
def count_faces_per_cell(face_dim):
    N = ndim+1
    K = face_dim+1
    return np.math.factorial(N)/(np.math.factorial(N-K)*np.math.factorial(K))


def test_create():
    T = DelaunayD()
    del T


def test_insert():
    # without duplicates
    T = DelaunayD()
    T.insert(pts)
    assert(T.is_valid())
    # with duplicates
    T = DelaunayD()
    T.insert(pts_dup)
    assert(T.is_valid())


def test_equal():
    T1 = DelaunayD()
    T1.insert(pts)
    T2 = DelaunayD()
    T2.insert(pts)
    assert(T1.is_equivalent(T2))


def test_num_verts():
    # without duplicates
    T = DelaunayD()
    T.insert(pts)
    print(T.num_finite_verts, T.num_infinite_verts, T.num_verts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)
    # with duplicates
    T = DelaunayD()
    T.insert(pts_dup)
    print(T.num_finite_verts, T.num_infinite_verts, T.num_verts)
    assert(T.num_finite_verts == nverts_fin)
    assert(T.num_infinite_verts == nverts_inf)
    assert(T.num_verts == nverts)


def test_num_cells():
    # without duplicates
    T = DelaunayD()
    T.insert(pts)
    print(T.num_finite_cells, T.num_infinite_cells, T.num_cells)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)
    # with duplicates
    T = DelaunayD()
    T.insert(pts_dup)
    print(T.num_finite_cells, T.num_infinite_cells, T.num_cells)
    assert(T.num_finite_cells == ncells_fin)
    assert(T.num_infinite_cells == ncells_inf)
    assert(T.num_cells == ncells)


def test_all_verts():
    T = DelaunayD()
    T.insert(pts)
    count_fin = count_inf = 0
    for v in T.all_verts:
        if v.is_infinite():
            count_inf += 1
        else:
            count_fin += 1
    count = count_fin + count_inf
    assert(count_fin == T.num_finite_verts)
    assert(count_inf == T.num_infinite_verts)
    assert(count == T.num_verts)


def test_finite_verts():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.finite_verts:
        assert((not v.is_infinite()))
        count += 1
    assert(count == T.num_finite_verts)


def test_all_cells():
    T = DelaunayD()
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
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for c in T.finite_cells:
        assert((not c.is_infinite()))
        count += 1
    assert(count == T.num_finite_cells)


def test_get_vertex():
    T = DelaunayD()
    T.insert(pts)
    for i in range(nverts_fin):
        v = T.get_vertex(i)
        assert(np.allclose(v.point, pts[i, :]))


def test_locate():
    T = DelaunayD()
    T.insert(pts)
    for c in T.finite_cells:
        r_cell = c
        p_cell = r_cell.center
        print(r_cell, p_cell, T.locate(p_cell))
        assert(r_cell == T.locate(p_cell))
        assert(r_cell == T.locate(p_cell, c))
        r_vert = c.vertex(0)
        p_vert = r_vert.point
        print(r_vert, p_vert, T.locate(p_vert))
        assert(r_vert == T.locate(p_vert))
        assert(r_vert == T.locate(p_vert, c))
        r_facet = c.facet(0)
        p_facet = r_facet.center
        # TODO: non-equivalence of facets with same vertices
        # x = T.locate(p_facet)
        # print("facet")
        # for i in range(r_facet.nverts):
        #     print(r_facet.vertex(i))
        # print("located")
        # for i in range(x.nverts):
        #     print(x.vertex(i))
        # print(r_facet, p_facet, T.locate(p_facet))
        # assert(r_facet.is_equivalent(r_facet))
        # assert(r_facet.is_equivalent(T.locate(p_facet)))
        # assert(r_facet.is_equivalent(T.locate(p_facet, c)))
        # assert(r_facet == T.locate(p_facet))
        # assert(r_facet == T.locate(p_facet, c))
        break


def test_remove():
    T = DelaunayD()
    T.insert(pts)
    v = T.get_vertex(0)
    T.remove(v)
    assert(T.num_verts == (nverts-1))


def test_clear():
    T = DelaunayD()
    T.insert(pts)
    T.clear()
    print(T.num_finite_verts, T.num_cells)
    assert(T.num_finite_verts == 0)
    assert(T.num_cells == 1)


def test_vert():
    T = DelaunayD()
    T.insert(pts)
    vold = None
    for v in T.all_verts:
        idx = v.index
        pnt = v.point
        vol = v.dual_volume
        print(v, idx, pnt, vol)
        assert(v == v)
        if vold is not None:
            assert(v != vold)
        if v.is_infinite():
            assert(idx == np.iinfo(np.uint32).max)
            assert(np.isinf(pnt).all())
            assert(np.isclose(vol, -1.0))
        else:
            assert(np.allclose(pnt, pts[idx, :]))
            if idx == 0:
                assert(np.isclose(vol, cvol))
            else:
                assert(np.isclose(vol, -1.0))
            c = v.cell
            v.set_cell(c)
            v.set_point(pnt)
        vold = v


# def test_edge():
#     T = DelaunayD()
#     T.insert(pts)
#     eold = None
#     for e in T.all_edges:
#         v1 = e.vertex(0)
#         v2 = e.vertex(1)
#         assert(v1 == e.vertex1)
#         assert(v2 == e.vertex2)
#         c = e.cell
#         i1 = e.ind1
#         i2 = e.ind2
#         elen = e.length
#         inf = e.is_infinite()
#         gab = e.is_Gabriel()
#         print(e, v1.index, v2.index, elen, inf, gab)
#         assert(e == e)
#         assert(e.is_equivalent(e))
#         if eold is not None:
#             assert(e != eold)
#         p1 = e.center
#         p2 = v1.point
#         print(e.side(p1), p1)
#         print(e.side(p2), p2)
#         if e.is_infinite():
#             assert(np.isclose(elen, -1.0))
#             assert(np.isinf(e.center).all())
#             assert(e.side(p1) == -1)
#             assert(e.side(p2) == -1)
#         else:
#             l = np.sqrt(np.sum((pts[v1.index, :]-pts[v2.index, :])**2.0))
#             assert(np.isclose(elen, l))
#             # p3 = e.center + 10*elen
#             # print(e.side(p3), p3)
#             # assert(e.side(p1) == -1) # virtually impossible
#             # assert(e.side(p2) == 0)
#             # assert(e.side(p3) == 1)
#         eold = e
#         del(c, i1, i2)


# def test_facet():
#     T = DelaunayD()
#     T.insert(pts)
#     fold = None
#     for f in T.all_facets:
#         v1 = f.vertex(0)
#         v2 = f.vertex(1)
#         v3 = f.vertex(2)
#         e1 = f.edge(0)
#         e2 = f.edge(1)
#         e3 = f.edge(2)
#         c = f.cell
#         i = f.ind
#         inf = f.is_infinite()
#         gab = f.is_Gabriel()
#         print(f, v1.index, v2.index, v3.index, i, inf, gab, f.center)
#         assert(f == f)
#         assert(f.is_equivalent(f))
#         if fold is not None:
#             assert(f != fold)
#         del(e1, e2, e3, c)

#         p1 = f.center
#         p2 = v1.point
#         print(f.side(p1), p1)
#         print(f.side(p2), p2)
#         if f.is_infinite():
#             assert(np.isinf(f.center).all())
#             assert(f.side(p1) == -1)
#             assert(f.side(p2) == -1)
#         # else:
#         #     p3 = 2*v1.point - f.center + np.arange(3)
#         #     print(f.side(p3), p3)
#         #     assert(f.side(p1) == -1)
#         #     assert(f.side(p2) == 0)
#         #     assert(f.side(p3) == 1)

#         # # This segfaults inside CGAL function call
#         # print(f.side_of_circle((v1.point+v2.point+v3.point)/3),
#         #       (v1.point+v2.point+v3.point)/3)
#         # print(f.side_of_circle(v1.point), v1.point)
#         # print(f.side_of_circle((5*v1.point-v2.point-v3.point)/3),
#         #       (5*v1.point-v2.point-v3.point)/3)
#         # if f.is_infinite():
#         #     assert(f.side_of_circle((v1.point+v2.point+v3.point)/3) == -1)
#         #     assert(f.side_of_circle(v1.point) == -1)
#         #     assert(f.side_of_circle((5*v1.point-v2.point-v3.point)/3) == -1)
#         # else:
#         #     # This segfaults...
#         #     assert(f.side_of_circle((v1.point+v2.point+v3.point)/3) == -1)
#         #     assert(f.side_of_circle(v1.point) == 0)
#         #     assert(f.side_of_circle((5*v1.point-v2.point-v3.point)/3) == 1)

#         fold = f


def test_cell():
    T = DelaunayD()
    T.insert(pts)
    cold = None
    for c in T.all_cells:
        print(c, c.circumcenter, c.center)
        assert(c == c)
        if cold is not None:
            assert(c != cold)

        f1 = c.facet(0)
        del(f1)

        v1 = c.vertex(0)
        assert(c.has_vertex(v1))
        assert(c.has_vertex(v1, return_index=True) == 0)
        assert(c.ind_vertex(v1) == 0)

        c.set_vertex(0, v1)

        n1 = c.neighbor(0)
        assert(c.has_neighbor(n1))
        assert(c.has_neighbor(n1, return_index=True) == 0)
        assert(c.ind_neighbor(n1) == 0)

        c.set_neighbor(0, n1)

        cold = c


# def test_io():
#     fname = 'test_io2348_3.dat'
#     Tout = DelaunayD()
#     Tout.insert(pts)
#     Tout.write_to_file(fname)
#     Tin = DelaunayD()
#     Tin.read_from_file(fname)
#     assert(Tout.num_verts == Tin.num_verts)
#     assert(Tout.num_cells == Tin.num_cells)
#     os.remove(fname)


def test_vert_incident_verts():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_vertices():
            c0 += 1
            count += 1
        x = v.incident_vertices()[0]
        print(v.index, c0, x)
    # print(count, 2*T.num_edges)
    # assert(count == 2*T.num_edges)


def test_vert_incident_edges():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_faces(1):
            c0 += 1
            count += 1
        x = v.incident_faces(1)[0]
        print(v.index, c0, x)
    # print(count, 2*T.num_edges)
    # assert(count == 2*T.num_edges)


def test_vert_incident_facets():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for e in v.incident_faces(ndim-1):
            c0 += 1
            count += 1
        x = v.incident_faces(ndim-1)[0]
        print(v.index, c0, x)
    # print(count, (ndim)*T.num_facets)
    # assert(count == (ndim)*T.num_facets)


def test_vert_incident_cells():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_verts:
        c0 = 0
        for c in v.incident_cells():
            c0 += 1
            count += 1
        print(v.index, c0)
    expected = count_faces_per_cell(0)*T.num_cells
    print(count, expected)
    assert(count == expected)


# def test_edge_incident_verts():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_edges:
#         c0 = 0
#         for e in v.incident_vertices():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count, 2*T.num_edges)
#     assert(count == 2*T.num_edges)  # 68


# def test_edge_incident_edges():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_edges:
#         c0 = 0
#         for e in v.incident_edges():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count)
#     assert(count == 404)


# def test_edge_incident_facets():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_edges:
#         c0 = 0
#         for e in v.incident_facets():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count, 3*T.num_facets)
#     assert(count == 3*T.num_facets)  # 144


# def test_edge_incident_cells():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_edges:
#         c0 = 0
#         for e in v.incident_cells():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count, 3*T.num_facets)
#     assert(count == 3*T.num_facets)  # 144


# def test_facet_incident_verts():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_facets:
#         c0 = 0
#         for e in v.incident_vertices():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count, 3*T.num_facets)
#     assert(count == 3*T.num_facets)  # 144


# def test_facet_incident_edges():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_facets:
#         c0 = 0
#         for e in v.incident_edges():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count, 3*T.num_facets)
#     assert(count == 3*T.num_facets)  # 144


# def test_facet_incident_facets():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_facets:
#         c0 = 0
#         for e in v.incident_facets():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count)
#     assert(count == 480)


# def test_facet_incident_cells():
#     T = DelaunayD()
#     T.insert(pts)
#     count = 0
#     for v in T.all_facets:
#         c0 = 0
#         for e in v.incident_cells():
#             c0 += 1
#             count += 1
#         print(c0)
#     print(count, 2*T.num_facets)
#     assert(count == 2*T.num_facets)  # 96


def test_cell_incident_verts():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    expected = count_faces_per_cell(0)*T.num_cells
    print(count, expected)
    assert(count == expected)


def test_cell_incident_edges():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_faces(1):
            c0 += 1
            count += 1
        print(c0)
    expected = count_faces_per_cell(1)*T.num_cells
    print(count, expected)
    assert(count == expected)


def test_cell_incident_facets():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_faces(ndim-1):
            c0 += 1
            count += 1
        print(c0)
    expected = count_faces_per_cell(ndim-1)*T.num_cells
    print(count, expected)
    assert(count == expected)

def test_cell_incident_cells():
    T = DelaunayD()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count, (ndim+1)*T.num_cells)
    assert(count == (ndim+1)*T.num_cells)


def test_mirror():
    T = DelaunayD()
    T.insert(pts)
    for c in T.all_cells:
        idx = 0
        i2 = T.mirror_index(c, idx)
        assert(c == c.neighbor(idx).neighbor(i2))
        v2 = T.mirror_vertex(c, idx)
        assert(v2 == c.neighbor(idx).vertex(i2))


def test_vertices():
    T = DelaunayD()
    T.insert(pts)
    v = T.vertices
    assert(v.shape[0] == pts.shape[0])
    assert(v.shape[1] == pts.shape[1])
    assert(np.allclose(pts, v))
