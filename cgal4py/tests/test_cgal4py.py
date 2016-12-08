r"""Tests for package level methods."""
import numpy as np
from datetime import datetime
import cProfile
import pstats
import time
import signal
import nose.tools as nt
from cgal4py import triangulate, voronoi_volumes, domain_decomp


@nt.nottest
def signal_print_traceback(signo, frame):
    print(traceback.print_stack(frame))


@nt.nottest
def make_points(npts, ndim, distrib='uniform', seed=0):
    npts = int(npts)
    if npts <= 0:
        if ndim == 2:
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
        elif ndim == 3:
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
        else:
            raise ValueError("Invalid 'ndim': {}".format(ndim))
        npts = pts.shape[0]
    else:
        np.random.seed(seed)
        LE = 0.0
        RE = 1.0
        left_edge = LE*np.ones(ndim, 'float64')
        right_edge = RE*np.ones(ndim, 'float64')
        if distrib == 'uniform':
            pts = np.random.uniform(low=LE, high=RE, size=(npts, ndim))
        elif distrib in ('gaussian', 'normal'):
            pts = np.random.normal(loc=(LE+RE)/2.0, scale=(RE-LE)/4.0,
                                   size=(npts, ndim))
            np.clip(pts, LE, RE)
        elif distrib in (2, '2'):
            pts = make_points(0, 2)[0]
        elif distrib in (3, '3'):
            pts = make_points(0, 3)[0]
        else:
            raise ValueError("Invalid 'distrib': {}".format(distrib))
    return pts, left_edge, right_edge


@nt.nottest
def make_test(npts, ndim, distrib='uniform', periodic=False,
              leafsize=None, nleaves=0):
    # Points
    pts, left_edge, right_edge = make_points(npts, ndim, distrib=distrib)
    npts = pts.shape[0]
    # Tree
    if leafsize is None:
        leafsize = npts/2 + 2
    tree = domain_decomp.tree("kdtree", pts, left_edge, right_edge,
                              periodic=periodic, leafsize=leafsize,
                              nleaves=nleaves)
    return pts, tree


@nt.nottest
def run_test(npts, ndim, nproc=0, func_name='tess', distrib='uniform',
             profile=False, use_mpi=False, use_python=False, use_buffer=False,
             limit_mem=False, suppress_final_output=False, **kwargs):
    r"""Run a rountine with a designated number of points & dimensions on a
    selected number of processors.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int): Number of particles.
        nproc (int): Number of processors.
        ndim (int): Number of dimensions.
        periodic (bool, optional): If True, the domain is assumed to be
            periodic. Defaults to False.
        use_mpi (bool, optional): If True, the MPI parallelized version is used
            instead of the version using the multiprocessing package. Defaults
            to False.
        use_python (bool, optional): If True and `use_mpi == True`, then
            communications are done in python using mpi4py. Defaults to False.
        use_buffer (bool, optional): If True and `use_mpi == True`, then buffer
            versions of MPI communications will be used rather than indirect
            communications of python objects via pickling. Defaults to False.
        limit_mem (bool, optional): If True, memory usage is limited by
            writing things to file at a cost to performance. Defaults to
            False.
        suppress_final_output (bool, optional): If True, the final output
            from spawned MPI processes is suppressed. This is mainly for
            timing purposes. Defaults to False.

    Raises:
        ValueError: If `func` is not one of the supported values.

    """
    unique_str = datetime.today().strftime("%Y%j%H%M%S")
    pts, left_edge, right_edge = make_points(npts, ndim, distrib=distrib)
    if func_name.lower() in ['tess', 'delaunay', 'triangulate']:
        func = triangulate
    elif func_name.lower() in ['vols', 'volumes', 'voronoivolumes']:
        func = voronoi_volumes
    else:
        raise ValueError("Unsupported function: {}".format(func_name))
    # Set keywords for multiprocessing version
    if nproc > 1:
        kwargs['use_mpi'] = use_mpi
        kwargs['limit_mem'] = limit_mem
        if use_mpi:
            kwargs['use_python'] = use_python
            kwargs['use_buffer'] = use_buffer
            kwargs['suppress_final_output'] = suppress_final_output
            if profile:
                kwargs['profile'] = '{}_mpi_profile.dat'.format(unique_str)
    # Run
    if profile:
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
    out = func(pts, left_edge=left_edge, right_edge=right_edge,
               nproc=nproc, **kwargs)
    if profile:
        pr.disable()
        t1 = time.time()
        ps = pstats.Stats(pr)
        if kwargs.get('use_mpi', False):
            ps.add(kwargs['profile'])
        if isinstance(profile, str):
            ps.dump_stats(profile)
            print("Stats saved to {}".format(profile))
        else:
            sort_key = 'tottime'
            ps.sort_stats(sort_key).print_stats(25)
            # ps.sort_stats(sort_key).print_callers(5)
            print("{} s according to 'time'".format(t1-t0))
        return ps


class MyTestCase(object):

    def __init__(self):
        self._func = None
        self.param_runs = []
        self.param_returns = []
        self.param_raises = []
        self.setup_param()

    def setup_param(self):
        pass

    @property
    def func(self):
        if self._func is None:
            raise AttributeError("_func must be set.")
        else:
            return self._func

    def check_runs(self, args, kwargs):
        self.func(*args, **kwargs)

    def check_returns(self, result, args, kwargs):
        nt.eq_(result, self.func(*args, **kwargs))

    def check_raises(self, excpt, args, kwargs):
        nt.assert_raises(excpt, self.func, *args, **kwargs)

    def test_runs_generator(self):
        for args, kwargs in self.param_runs:
            yield self.check_runs, args, kwargs

    def test_returns_generator(self):
        for res, args, kwargs in self.param_returns:
            yield self.check_returns, res, args, kwargs

    def test_raises_generator(self):
        for err, args, kwargs in self.param_raises:
            yield self.check_raises, err, args, kwargs



class TestTriangulate(MyTestCase):

    def setup_param(self):
        self._func = triangulate
        self.param_runs = []
        for ndim in [2, 3]:  # , 4]:
            pts, le, re = make_points(0, ndim)
            self.param_runs += [
                ((pts,), {}),
                ((pts,), {'periodic': True,
                          'left_edge': le, 'right_edge': re}),
                ]
        pts, le, re = make_points(0, 2)
        self.param_raises = [
            (ValueError, (np.zeros((3, 3, 3)),), {}),
            (ValueError, (pts,), {'left_edge': np.zeros(3)}),
            (ValueError, (pts,), {'right_edge': np.zeros(3)}),
            (ValueError, (pts,), {'left_edge': np.zeros((2, 2, 2))}),
            (ValueError, (pts,), {'right_edge': np.zeros((2, 2, 2))}),
            ]


class TestVoronoiVolumes(MyTestCase):

    def setup_param(self):
        self._func = voronoi_volumes
        self.param_runs = []
        for ndim in [2, 3]:  # , 4]:
            pts, le, re = make_points(0, ndim)
            self.param_runs += [
                ((pts,), {}),
                ((pts,), {'periodic': True,
                          'left_edge': le, 'right_edge': re}),
                ]
        pts, le, re = make_points(0, 2)
        self.param_raises = [
            (ValueError, (np.zeros((3, 3, 3)),), {}),
            (ValueError, (pts,), {'left_edge': np.zeros(3)}),
            (ValueError, (pts,), {'right_edge': np.zeros(3)}),
            (ValueError, (pts,), {'left_edge': np.zeros((2, 2, 2))}),
            (ValueError, (pts,), {'right_edge': np.zeros((2, 2, 2))}),
            ]
