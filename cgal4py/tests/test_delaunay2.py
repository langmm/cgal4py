r"""Tests for 2D Delaunay triangulation.

.. todo::
   * value test circumcenter
   * test includes_edge
   * answer testing for voronoi volumes
"""
import numpy as np
import os
from cgal4py.delaunay import Delaunay2
from cgal4py.tests.test_cgal4py import MyTestCase, make_points


pts, left_edge, right_edge = make_points(0, 2)
pts_dup = np.concatenate([pts, np.reshape(pts[0, :], (1, pts.shape[1]))])
nverts_fin = pts.shape[0]
nverts_inf = 1
nverts = nverts_fin + nverts_inf
nedges_fin = 17
nedges_inf = 4
nedges = nedges_fin + nedges_inf
ncells_fin = 10
ncells_inf = 4
ncells = ncells_fin + ncells_inf


class TestDelaunay2(MyTestCase):

    def setup_param(self):
        self._func = Delaunay2
        self.param_runs = [
            ((pts,), {}),
            ((pts_dup,), {}),
            ]
        self.T = Delaunay2()
        self.T.insert(pts)
        self.Tdup = Delaunay2()
        self.Tdup.insert(pts_dup)
        self.pts = pts
        self.pts_dup = pts_dup

    def check_runs(self, args, kwargs):
        out = self.func()
        out.insert(*args, **kwargs)
        assert(out.is_valid())

    def new_T(self):
        T = Delaunay2()
        T.insert(pts)
        return T

    def test_equal(self):
        assert(self.T.is_equivalent(self.T))

    def check_num_verts(self, T):
        print(T.num_finite_verts, T.num_infinite_verts, T.num_verts)
        assert(T.num_finite_verts == nverts_fin)
        assert(T.num_infinite_verts == nverts_inf)
        assert(T.num_verts == nverts)

    def check_num_edges(self, T):
        print(T.num_finite_edges, T.num_infinite_edges, T.num_edges)
        assert(T.num_finite_edges == nedges_fin)
        assert(T.num_infinite_edges == nedges_inf)
        assert(T.num_edges == nedges)

    def check_num_cells(self, T):
        print(T.num_finite_cells, T.num_infinite_cells, T.num_cells)
        assert(T.num_finite_cells == ncells_fin)
        assert(T.num_infinite_cells == ncells_inf)
        assert(T.num_cells == ncells)

    def test_counts_generate(self):
        for T in [self.T, self.Tdup]:
            yield self.check_num_verts, T
            yield self.check_num_edges, T
            yield self.check_num_cells, T

    def test_all_verts(self):
        T = self.T
        count_fin = count_inf = 0
        for v in T.all_verts:
            print(v.index, v.point)
            if v.is_infinite():
                count_inf += 1
            else:
                count_fin += 1
        count = count_fin + count_inf
        assert(count_fin == T.num_finite_verts)
        assert(count_inf == T.num_infinite_verts)
        assert(count == T.num_verts)

    def test_finite_verts(self):
        T = self.T
        count = 0
        for v in T.finite_verts:
            assert((not v.is_infinite()))
            count += 1
        assert(count == T.num_finite_verts)

    def test_all_edges(self):
        T = self.T
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

    def test_finite_edges(self):
        T = self.T
        count = 0
        for e in T.finite_edges:
            assert((not e.is_infinite()))
            count += 1
        print(count)
        assert(count == T.num_finite_edges)

    def test_all_cells(self):
        T = self.T
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

    def test_finite_cells(self):
        T = self.T
        count = 0
        for c in T.finite_cells:
            assert((not c.is_infinite()))
            count += 1
        assert(count == T.num_finite_cells)

    def test_clear(self):
        T = self.new_T()
        T.clear()
        print(T.num_finite_verts, T.num_cells)
        assert(T.num_finite_verts == 0)
        assert(T.num_cells == 0)

    def test_io(self):
        fname = 'test_io2348_2.dat'
        Tout = self.T
        Tout.write_to_file(fname)
        Tin = Delaunay2.from_file(fname)
        assert(Tout.num_verts == Tin.num_verts)
        assert(Tout.num_cells == Tin.num_cells)
        os.remove(fname)

    def test_serialize(self):
        Tout = self.T
        f, n, i = Tout.serialize()
        Tin = Delaunay2.from_serial(self.pts, f, n, i)
        assert(Tout.num_verts == Tin.num_verts)
        assert(Tout.num_cells == Tin.num_cells)

    def test_locate(self):
        T = self.T
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

    def test_get_vertex(self):
        T = self.T
        for i in range(nverts_fin):
            v = T.get_vertex(i)
            assert(np.allclose(v.point, self.pts[i, :]))

    def test_remove(self):
        T = self.new_T()
        v = T.get_vertex(0)
        T.remove(v)
        assert(T.num_verts == (nverts-1))

    def test_is_edge(self):
        T = self.T
        assert(T.is_edge(T.get_vertex(0), T.get_vertex(1)))
        assert(not T.is_edge(T.get_vertex(0), T.get_vertex(nverts_fin-1)))

    def test_is_cell(self):
        T = self.T
        assert(T.is_cell(T.get_vertex(0), T.get_vertex(1), T.get_vertex(3)))
        assert(not T.is_cell(T.get_vertex(0), T.get_vertex(1),
                             T.get_vertex(nverts_fin-1)))

    def test_vert(self):
        T = self.new_T()
        vold = None
        for v in T.all_verts:
            pnt = v.point
            idx = v.index
            vol = v.dual_volume
            print(v, idx, pnt, vol)
            assert(v == v)
            if vold is not None:
                assert(v != vold)
            if v.is_infinite():
                assert(np.isinf(pnt).all())
                assert(idx == np.iinfo(np.uint32).max)
                assert(np.isclose(vol, -1))
            else:
                assert(np.allclose(pnt, pts[idx, :]))
                if idx >= 4:
                    assert(np.isclose(vol, -1))
                c = v.cell
                v.set_cell(c)
                v.set_point(pnt)
            vold = v


# class TestVertex2(MyTestCase):

#     def setup_param(self):
#         self._func = None
#         self.pts = make_points(0, 2)
#         self.T = Delaunay2()
#         self.T.insert(self.pts)
#         self.v = self.T.finite_verts

#     def check_point(self, v):
#         pnt = v.point
#         idx = v.index
#         if v.isinfinite():
#             assert(np.isinf(pnt).all())
#             assert(idx == np.iinfo(np.uint32).max)
#         else:
#             assert(np.allclose(pnt, self.pts[idx, :]))


def test_edge():
    T = Delaunay2()
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
            l = np.sqrt(np.sum((pts[v1.index, :]-pts[v2.index, :])**2.0))
            assert(np.isclose(elen, l))
        eold = e


def test_cell():
    T = Delaunay2()
    T.insert(pts)
    cold = None
    for c in T.all_cells:
        print(c, c.dimension, c.circumcenter, c.center, c.min_angle)
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
        del(e1, e2, e3)

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

        print(c.side(c.center))
        print(c.side(v1.point))
        print(c.side(2*v1.point-c.center))
        print(c.side_of_circle(c.circumcenter))
        print(c.side_of_circle(v1.point))
        print(c.side_of_circle(2*v1.point-c.circumcenter),
              2*v1.point-c.circumcenter)
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
            assert(c.side(c.center) == -1)
            assert(c.side(v1.point) == 0)
            assert(c.side(2*v1.point-c.center) == 1)
            assert(c.side_of_circle(c.circumcenter) == -1)
            assert(c.side_of_circle(v1.point) == 0)
            assert(c.side_of_circle(2*v1.point-c.circumcenter) == 1)

        c.reorient()
        c.ccw_permute()
        c.cw_permute()

        cold = c


def test_move():
    T = Delaunay2()
    T.insert(pts)
    v0 = T.get_vertex(0)
    new_pos = np.zeros(2, 'float64')
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
    T = Delaunay2()
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
    T = Delaunay2()
    T.insert(pts)
    for c in T.all_cells:
        T.flip(c, 0)
    print(T.num_edges, nedges)
    assert(T.num_edges == nedges)
    for e in T.all_edges:
        e.flip()
    print(T.num_edges, nedges)


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


def test_cell_incident_verts():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_vertices():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 42)


def test_cell_incident_edges():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_edges():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 42)


def test_cell_incident_cells():
    T = Delaunay2()
    T.insert(pts)
    count = 0
    for v in T.all_cells:
        c0 = 0
        for e in v.incident_cells():
            c0 += 1
            count += 1
        print(c0)
    print(count)
    assert(count == 42)


def test_nearest_vertex():
    idx_test = 7
    T = Delaunay2()
    T.insert(pts)
    v = T.nearest_vertex(pts[idx_test, :]-0.1)
    assert(v.index == idx_test)


def test_mirror():
    T = Delaunay2()
    T.insert(pts)
    for e in T.all_edges:
        e2 = T.mirror_edge(e)
        del(e2)
        break
    for c in T.all_cells:
        idx = 0
        i2 = T.mirror_index(c, idx)
        assert(c == c.neighbor(idx).neighbor(i2))
        v2 = T.mirror_vertex(c, idx)
        assert(v2 == c.neighbor(idx).vertex(i2))


def test_get_boundary_of_conflicts():
    T = Delaunay2()
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    edges = T.get_boundary_of_conflicts(p, c)
    print(len(edges))


def test_get_conflicts():
    T = Delaunay2()
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    cells = T.get_conflicts(p, c)
    print(len(cells))


def test_get_conflicts_and_boundary():
    T = Delaunay2()
    T.insert(pts)
    v = T.get_vertex(0)
    c = v.incident_cells()[0]
    p = c.circumcenter
    cells, edges = T.get_conflicts_and_boundary(p, c)
    print(len(cells), len(edges))


def test_line_walk():
    T = Delaunay2()
    T.insert(pts)
    p1 = np.array([-1, -1], 'float64')
    p2 = np.array([+1, +1], 'float64')
    x = T.line_walk(p1, p2)
    print(len(x))
    assert(len(x) == 6)


def test_vertices():
    T = Delaunay2()
    T.insert(pts)
    v = T.vertices
    assert(v.shape[0] == pts.shape[0])
    assert(v.shape[1] == pts.shape[1])
    assert(np.allclose(pts, v))


def test_edges():
    T = Delaunay2()
    T.insert(pts)
    e = T.edges
    assert(e.shape[0] == T.num_finite_edges)
    assert(e.shape[1] == 2)


def test_boundary_points():
    T = Delaunay2()
    T.insert(pts)
    out = T.boundary_points(left_edge, right_edge, False)
    del(out)


def test_outgoing_points():
    T = Delaunay2()
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
    T = Delaunay2()
    T.insert(pts)
    v = T.voronoi_volumes()
    assert(v.shape[0] == T.num_finite_verts)

def test_minimum_angles():
    T = Delaunay2()
    T.insert(pts)
    v = T.minimum_angles()
    assert(v.shape[0] < T.num_finite_cells)
