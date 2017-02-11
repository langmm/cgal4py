r"""Routines for running triangulations in paralle.

.. todo::
   * parallelism through treading

"""
from cgal4py import PY_MAJOR_VERSION, _use_multiprocessing
from cgal4py.delaunay import Delaunay, tools, _get_Delaunay
from cgal4py import domain_decomp
from cgal4py.domain_decomp import GenericTree
import numpy as np
import os
import sys
import time
import copy
import struct
import pstats
if PY_MAJOR_VERSION == 2:
    import cPickle as pickle
else:
    import pickle
if _use_multiprocessing:
    import multiprocessing as mp
    from multiprocessing import Process as mp_Process
else:
    mp = object
    mp_Process = object
import warnings
from datetime import datetime
try:
    from mpi4py import MPI
    mpi_loaded = True
except:
    mpi_loaded = False
    warnings.warn("mpi4py could not be imported.")
import ctypes


def _get_mpi_type(np_type):
    r"""Get the correpsonding MPI data type for a given numpy data type.

    Args:
        np_type (str, type): String identifying a numpy data type or a numpy
            data type.

    Returns:
        int: MPI data type.

    Raises:
        ValueError: If `np_type` is not supported.

    """
    if not mpi_loaded:
        raise Exception("mpi4py could not be imported.")
    if np_type in ('i', 'int32', np.int32):
        mpi_type = MPI.INT
    elif np_type in ('l', 'int64', np.int64):
        mpi_type = MPI.LONG
    elif np_type in ('f', 'float32', np.float32):
        mpi_type = MPI.FLOAT
    elif np_type in ('d', 'float64', np.float64):
        mpi_type = MPI.DOUBLE
    else:
        raise ValueError("Unrecognized type: {}".format(np_type))
    return mpi_type


def _generate_filename(name, unique_str=None, ext='.dat'):
    fname = '{}{}'.format(name, ext)
    if isinstance(unique_str, str):
        fname = '{}_{}'.format(unique_str, fname)
    return fname


def _prof_filename(unique_str=None):
    return _generate_filename('prof', unique_str=unique_str, ext='.dat')


def _tess_filename(unique_str=None):
    return _generate_filename('tess', unique_str=unique_str, ext='.dat')


def _vols_filename(unique_str=None):
    return _generate_filename('vols', unique_str=unique_str, ext='.npy')


def _leaf_tess_filename(leaf_id, unique_str=None):
    return _generate_filename('leaf{}'.format(leaf_id),
                              unique_str=unique_str, ext='.dat')


def _final_leaf_tess_filename(leaf_id, unique_str=None):
    return _generate_filename('finaleleaf{}'.format(leaf_id),
                              unique_str=unique_str, ext='.dat')


def write_mpi_script(fname, read_func, taskname, unique_str=None,
                     use_double=False, use_python=False, use_buffer=False,
                     overwrite=False, profile=False, limit_mem=False,
                     suppress_final_output=False):
    r"""Write an MPI script for calling MPI parallelized triangulation.

    Args:
        fname (str): Full path to file where MPI script will be saved.
        read_func (func): Function for reading in points. The function should
            return a dictionary with 'pts' key at a minimum corresponding to
            the 2D array of points that should be triangulated. Additional
            optional keys include:

            * periodic (bool): True if the domain is periodic.
            * left_edge (np.ndarray of float64): Left edges of the domain.
            * right_edge (np.ndarray of float64): Right edges of the domain.

            A list of lines resulting in the above mentioned dictionary is also
            accepted.
        taskname (str): Name of task to be passed to 
            :class:`cgal4py.parallel.DelaunayProcessMPI`.
        unique_str (str, optional): Unique string identifying the domain
            decomposition that is passed to `cgal4py.parallel.ParallelLeaf` for
            file naming. Defaults to None.
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        use_python (bool, optional): If True, communications are done in python
            using mpi4py. Otherwise, communications are done in C++ using MPI.
            Defaults to False.
        use_buffer (bool, optional): If True, communications are done by way of
            buffers rather than pickling python objects. Defaults to False.
        overwrite (bool): If True, any existing script with the same name is
            overwritten. Defaults to False.
        profile (bool, optional): If True, cProfile is used to profile the code
            and output is printed to the screen. This can also be a string
            specifying the full path to the file where the output should be
            saved. Defaults to False.
        limit_mem (bool, optional): If False, the triangulation results from
            each process are moved to local memory using `multiprocessing`
            pipes. If True, each process writes out tessellation info to
            files which are then incrementally loaded as consolidation occurs.
            Defaults to False.
        suppress_final_output (bool, optional): If True, output of the result
            to file is suppressed. This is mainly for testing purposes.
            Defaults to False.

    """
    if not mpi_loaded:
        raise Exception("mpi4py could not be imported.")
    if os.path.isfile(fname):
        if overwrite:
            os.remove(fname)
        else:
            return
    readcmds = isinstance(read_func, list)
    # Import lines
    lines = [
        "import numpy as np",
        "from mpi4py import MPI"]
    if profile:
        lines += [
            "import cProfile",
            "import pstats"]
    lines += ["from cgal4py import parallel"]
    if not readcmds:
        lines.append(
            "from {} import {} as load_func".format(read_func.__module__,
                                                    read_func.__name__))
    # Lines establishing variables
    lines += [
        "",
        "comm = MPI.COMM_WORLD",
        "size = comm.Get_size()",
        "rank = comm.Get_rank()",
        "",
        "unique_str = '{}'".format(unique_str),
        "use_double = {}".format(use_double),
        "limit_mem = {}".format(limit_mem),
        "use_python = {}".format(use_python),
        "use_buffer = {}".format(use_buffer),
        "suppress_final_output = {}".format(suppress_final_output),
        ""]
    # Commands to read in data
    lines += [
        "if rank == 0:"]
    if readcmds:
        lines += ["    "+l for l in read_func]
    else:
        lines.append(
            "    load_dict = load_func()")
    lines += [
        "    pts = load_dict['pts']",
        "    left_edge = load_dict.get('left_edge', np.min(pts, axis=0))",
        "    right_edge = load_dict.get('right_edge', np.max(pts, axis=0))",
        "    periodic = load_dict.get('periodic', False)",
        "    tree = load_dict.get('tree', None)",
        "else:",
        "    pts = None",
        "    left_edge = None",
        "    right_edge = None",
        "    periodic = None",
        "    tree = None"]
    # Start profiler if desired
    if profile:
        lines += [
            "if (rank == 0):",
            "    pr = cProfile.Profile()",
            "    pr.enable()",
            ""]
    # Run
    lines += [
        "p = parallel.DelaunayProcessMPI('{}',".format(taskname),
        "    pts, tree, left_edge=left_edge, right_edge=right_edge,",
        "    periodic=periodic, use_double=use_double, unique_str=unique_str,",
        "    limit_mem=limit_mem, use_python=use_python,",
        "    use_buffer=use_buffer,",
        "    suppress_final_output=suppress_final_output)",
        "p.run()"]
    if profile:
        lines += [
            "",
            "if (rank == 0):",
            "    pr.disable()"]
        if isinstance(profile, str):
            lines.append(
                "    pr.dump_stats('{}')".format(profile))
        else:
            lines.append(
                "    pstats.Stats(pr).sort_stats('time').print_stats(25)")
    with open(fname, 'w') as f:
        f.write("\n".join(lines))


def ParallelDelaunay(pts, tree, nproc, use_mpi=True, **kwargs):
    r"""Return a triangulation that is constructed in parallel.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional
            coordinates.
        tree (object): Domain decomposition tree for splitting points among the
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        nproc (int): Number of processors that should be used.
        use_mpi (bool, optional): If True, `mpi4py` is used for communications.
            Otherwise `multiprocessing` is used. Defaults to True.
        \*\*kwargs: Additional keywords arguments are passed to the correct
            parallel implementation of the triangulation.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`: consolidated 2D or 3D
            triangulation object.

    """
    if use_mpi:
        unique_str = datetime.today().strftime("%Y%j%H%M%S")
        fpick = _generate_filename("dict", unique_str=unique_str)
        out = dict(pts=pts, tree=GenericTree.from_tree(tree))
        if PY_MAJOR_VERSION == 2:
            with open(fpick, 'wb') as fd:
                pickle.dump(out, fd, pickle.HIGHEST_PROTOCOL)
            assert(os.path.isfile(fpick))
            read_lines = ["import cPickle",
                          "with open('{}', 'rb') as fd:".format(fpick),
                          "    load_dict = cPickle.load(fd)"]
        else:
            with open(fpick, 'wb') as fd:
                pickle.dump(out, fd)
            assert(os.path.isfile(fpick))
            read_lines = ["import pickle",
                          "with open('{}', 'rb') as fd:".format(fpick),
                          "    load_dict = pickle.load(fd)"]
        ndim = tree.ndim
        out = ParallelDelaunayMPI(read_lines, ndim, nproc, **kwargs)
        os.remove(fpick)
    else:
        if _use_multiprocessing:
            out = ParallelDelaunayMulti(pts, tree, nproc, **kwargs)
        else:
            raise RuntimeError("The multiprocessing version of parallelism " +
                               "is currently disabled. To enable it, set " +
                               "_use_multiprocessing to True in " +
                               "cgal4py/__init__.py.")
    return out


def ParallelVoronoiVolumes(pts, tree, nproc, use_mpi=True, **kwargs):
    r"""Return a triangulation that is constructed in parallel.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional
            coordinates.
        tree (object): Domain decomposition tree for splitting points among the
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        nproc (int): Number of processors that should be used.
        use_mpi (bool, optional): If True, `mpi4py` is used for communications.
            Otherwise `multiprocessing` is used. Defaults to True.
        \*\*kwargs: Additional keywords arguments are passed to the correct
            parallel implementation of the triangulation.

    Returns:
        np.ndarray of float64: (n,) array of n voronoi volumes for the provided
            points.

    """
    if use_mpi:
        unique_str = datetime.today().strftime("%Y%j%H%M%S")
        fpick = _generate_filename("dict", unique_str=unique_str)
        out = dict(pts=pts, tree=GenericTree.from_tree(tree))
        if PY_MAJOR_VERSION == 2:
            with open(fpick, 'wb') as fd:
                pickle.dump(out, fd, pickle.HIGHEST_PROTOCOL)
            assert(os.path.isfile(fpick))
            read_lines = ["import cPickle",
                          "with open('{}', 'rb') as fd:".format(fpick),
                          "    load_dict = cPickle.load(fd)"]
        else:
            with open(fpick, 'wb') as fd:
                pickle.dump(out, fd)
            assert(os.path.isfile(fpick))
            read_lines = ["import pickle",
                          "with open('{}', 'rb') as fd:".format(fpick),
                          "    load_dict = pickle.load(fd)"]
        ndim = tree.ndim
        out = ParallelVoronoiVolumesMPI(read_lines, ndim, nproc, **kwargs)
        os.remove(fpick)
    else:
        if _use_multiprocessing:
            out = ParallelVoronoiVolumesMulti(pts, tree, nproc, **kwargs)
        else:
            raise RuntimeError("The multiprocessing version of parallelism " +
                               "is currently disabled. To enable it, set " +
                               "_use_multiprocessing to True in " +
                               "cgal4py/__init__.py.")
    return out


def ParallelDelaunayMPI(*args, **kwargs):
    r"""Return a triangulation that is constructed in parallel using MPI.
    See :func:`cgal4py.parallel.ParallelMPI` for information on arguments.

    Returns:
        A Delaunay triangulation class like :class:`cgal4py.delaunay.Delaunay2`
            (but for the appropriate number of dimensions) will be returned.


    """
    return ParallelMPI('triangulate', *args, **kwargs)
    

def ParallelVoronoiVolumesMPI(*args, **kwargs):
    r"""Return the voronoi cell volumes after constructing triangulation in
    parallel using MPI. See :func:`cgal4py.parallel.ParallelMPI` for
    information on arguments.

    Returns:
        np.ndarray of float64: (n,) array of n voronoi volumes for the provided
            points.

    """
    return ParallelMPI('volumes', *args, **kwargs)


def ParallelMPI(task, read_func, ndim, nproc, use_double=False,
                limit_mem=False, use_python=False, use_buffer=False,
                profile=False, suppress_final_output=False):
    r"""Return results form a triangulation that is constructed in parallel
    using MPI.

    Args:
        task (str): Task for which results should be returned. 
            Values include:

            * 'triangulate': Return the Delaunay triangulation class.
            * 'volumes': Return the volumes of the Voronoi cells associated with
              each point.

        read_func (func): Function for reading in points. The function should
            return a dictionary with 'pts' key at a minimum corresponding to
            the 2D array of points that should be triangulated. Additional
            optional keys include:

            * periodic (bool): True if the domain is periodic.
            * left_edge (np.ndarray of float64): Left edges of the domain.
            * right_edge (np.ndarray of float64): Right edges of the domain.

            A list of lines resulting in the above mentioned dictionary is also
            accepted.
        ndim (int): Number of dimension in the domain.
        nproc (int): Number of processors that should be used.
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        limit_mem (bool, optional): If False, the triangulation results from
            each process are moved to local memory using `multiprocessing`
            pipes. If True, each process writes out tessellation info to
            files which are then incrementally loaded as consolidation occurs.
            Defaults to False.
        use_python (bool, optional): If True, communications are done in python
            using mpi4py. Otherwise, communications are done in C++ using MPI.
            Defaults to False.
        use_buffer (bool, optional): If True, communications are done by way of
            buffers rather than pickling python objects. Defaults to False.
        profile (bool, optional): If True, cProfile is used to profile the code
            and output is printed to the screen. This can also be a string
            specifying the full path to the file where the output should be
            saved. Defaults to False.
        suppress_final_output (bool, optional): If True, output of the result
            to file is suppressed. This is mainly for testing purposes.
            Defaults to False.

    Returns:
        Dependent on task. For 'triangulate', a Delaunay triangulation class
            like :class:`cgal4py.delaunay.Delaunay2` (but for the appropriate
            number of dimensions will be returned. For 'volumes', a numpy
            array of floating point volumes will be returned where values
            less than zero indicate infinite volumes.

    Raises:
        ValueError: If the task is not one of the accepted values listed above.
        RuntimeError: If the MPI script does not result in a file containing
            the triangulation.

    """
    if task not in ['triangulate', 'volumes']:
        raise ValueError("Unsupported task: {}".format(task))
    unique_str = datetime.today().strftime("%Y%j%H%M%S")
    fscript = '{}_mpi.py'.format(unique_str)
    write_mpi_script(fscript, read_func, task, limit_mem=limit_mem,
                     unique_str=unique_str, use_double=use_double,
                     use_python=use_python, use_buffer=use_buffer,
                     profile=profile,
                     suppress_final_output=suppress_final_output)
    cmd = 'mpiexec -np {} python {}'.format(nproc, fscript)
    os.system(cmd)
    os.remove(fscript)
    if suppress_final_output:
        return
    if task == 'triangulate':
        fres = _tess_filename(unique_str=unique_str)
    elif task == 'volumes':
        fres = _vols_filename(unique_str=unique_str)
    if os.path.isfile(fres):
        with open(fres, 'rb') as fd:
            if task == 'triangulate':
                out = _get_Delaunay(ndim=ndim).from_serial_buffer(fd)
            elif task == 'volumes':
                out = np.frombuffer(bytearray(fd.read()), dtype='d')
        os.remove(fres)
        return out
    else:
        raise RuntimeError("The tessellation file does not exist. " +
                           "There must have been an error while running the " +
                           "parallel script.")
    

if _use_multiprocessing:
    def ParallelDelaunayMulti(*args, **kwargs):
        r"""Return a triangulation that is constructed in parallel using the
        `multiprocessing` package. See :func:`cgal4py.parallel.ParallelMulti`
        for information on arguments.

        Returns:
            A Delaunay triangulation class like :class:`cgal4py.delaunay.Delaunay2`
                (but for the appropriate number of dimensions) will be returned.

        """
        return ParallelMulti('triangulate', *args, **kwargs)


    def ParallelVoronoiVolumesMulti(*args, **kwargs):
        r"""Return the voronoi cell volumes after constructing triangulation in
        parallel. See :func:`cgal4py.parallel.ParallelMulti` for information on
        arguments.

        Returns:
            np.ndarray of float64: (n,) array of n voronoi volumes for the provided
                points.

        """
        return ParallelMulti('volumes', *args, **kwargs)


    def ParallelMulti(task, pts, tree, nproc, use_double=False, limit_mem=False):
        r"""Return results from a triangulation that is constructed in parallel
        using the `multiprocessing` package.

        Args:
            task (str): Task for which results should be returned. Values
                include:

                * 'triangulate': Return the Delaunay triangulation class.
                * 'volumes': Return the volumes of the Voronoi cells associated with
                  each point.

            pts (np.ndarray of float64): (n,m) array of n m-dimensional
                coordinates.
            tree (object): Domain decomposition tree for splitting points among the
                processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
            nproc (int): Number of processors that should be used.
            use_double (bool, optional): If True, the triangulation is forced to
                use 64bit integers reguardless of if there are too many points for
                32bit. Otherwise 32bit integers are used so long as the number of
                points is <=4294967295. Defaults to False.
            limit_mem (bool, optional): If False, the triangulation results from
                each process are moved to local memory using `multiprocessing`
                pipes. If True, each process writes out tessellation info to
                files which are then incrementally loaded as consolidation occurs.
                Defaults to False.

        Returns:
            Dependent on task. For 'triangulate', a Delaunay triangulation class
                like :class:`cgal4py.delaunay.Delaunay2` (but for the appropriate
                number of dimensions) will be returned. For 'volumes', a numpy
                array of floating point volumes will be returned where values
                less than zero indicate infinite volumes.

        """
        idxArray = mp.RawArray(ctypes.c_ulonglong, tree.idx.size)
        ptsArray = mp.RawArray('d', pts.size)
        memoryview(idxArray)[:] = tree.idx
        memoryview(ptsArray)[:] = pts
        # Split leaves
        task2leaves = [[] for _ in range(nproc)]
        for leaf in tree.leaves:
            proc = leaf.id % nproc
            task2leaves[proc].append(leaf)
        left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
        right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        # Create & execute processes
        count = [mp.Value('i', 0), mp.Value('i', 0), mp.Value('i', 0)]
        lock = mp.Condition()
        queues = [mp.Queue() for _ in range(nproc+1)]
        in_pipes = [None for _ in range(nproc)]
        out_pipes = [None for _ in range(nproc)]
        for i in range(nproc):
            out_pipes[i], in_pipes[i] = mp.Pipe(True)
        unique_str = datetime.today().strftime("%Y%j%H%M%S")
        processes = [DelaunayProcessMulti(
            task, _, task2leaves[_], ptsArray, idxArray,
            left_edges, right_edges, queues, lock, count, in_pipes[_],
            unique_str=unique_str, limit_mem=limit_mem) for _ in range(nproc)]
        for p in processes:
            p.start()
        # Synchronize to ensure rapid receipt of output info from leaves
        lock.acquire()
        lock.wait()
        lock.release()
        # Setup methods for recieving leaf info
        if task == 'triangulate':
            serial = [None for _ in range(tree.num_leaves)]

            def recv_leaf(p):
                iid, s = processes[p].receive_result(out_pipes[p])
                assert(tree.leaves[iid].id == iid)
                serial[iid] = s

        elif task == 'volumes':
            vol = np.empty(pts.shape[0], pts.dtype)

            def recv_leaf(p):
                iid, ivol = processes[p].receive_result(out_pipes[p])
                assert(tree.leaves[iid].id == iid)
                vol[tree.idx[tree.leaves[iid].slice]] = ivol

        # Recieve output from processes
        proc_list = range(nproc)
        # Version that takes whatever is available
        total_count = 0
        max_total_count = tree.num_leaves
        while total_count != max_total_count:
            for i in proc_list:
                while out_pipes[i].poll():
                    recv_leaf(i)
                    total_count += 1
        # Version that does processors in order
        # for i in proc_list:
        #     for _ in range(len(task2leaves[i])):
        #         recv_leaf(i)
        # Consolidate tessellation
        if task == 'triangulate':
            out = consolidate_tess(tree, serial, pts, use_double=use_double,
                                   unique_str=unique_str, limit_mem=limit_mem)
        elif task == 'volumes':
            out = vol
        # Close queues and processes
        for p in processes:
            p.join()
        return out


# @profile
def consolidate_tess(tree, leaf_output, pts, use_double=False,
                     unique_str=None, limit_mem=False):
    r"""Creates a single triangulation from the triangulations of leaves.

    Args:
        tree (object): Domain decomposition tree for splitting points among the
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        leaf_output (object): Output from each parallel leaf.
        pts (np.ndarray of float64): (n,m) Array of n mD points.
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        unique_str (str, optional): Unique identifier for files in a run. If
            `limit_mem == True` those files will be loaded and used to create
            the consolidated tessellation. Defaults to None. If None, there is
            a risk that multiple runs could be sharing files of the same name.
        limit_mem (bool, optional): If False, the triangulation is consolidated
            from partial triangulations on each leaf that already exist in
            memory. If True, partial triangulations are loaded from files for
            each leaf. Defaults to `False`.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`: consolidated 2D or 3D
            triangulation object.

    """
    npts = pts.shape[0]
    ndim = pts.shape[1]
    uint32_max = np.iinfo('uint32').max
    if npts >= uint32_max:
        use_double = True
    if use_double:
        idx_inf = np.uint64(np.iinfo('uint64').max)
    else:
        idx_inf = np.uint32(uint32_max)
    # Loop over leaves adding them
    if not limit_mem:
        ncells_tot = 0
        for s in leaf_output:
            ncells_tot += np.int64(s[5])
        if use_double:
            cons = tools.ConsolidatedLeaves64(ndim, idx_inf, ncells_tot)
        else:
            cons = tools.ConsolidatedLeaves32(ndim, idx_inf, ncells_tot)
        for i, leaf in enumerate(tree.leaves):
            leaf_dtype = leaf_output[i][0].dtype
            if leaf_dtype == np.uint64:
                sleaf = tools.SerializedLeaf64(
                    leaf.id, ndim, leaf_output[i][0].shape[0],
                    leaf_output[i][2], leaf_output[i][0], leaf_output[i][1],
                    leaf_output[i][3], leaf_output[i][4],
                    leaf.start_idx, leaf.stop_idx)
            elif leaf_dtype == np.uint32:
                sleaf = tools.SerializedLeaf32(
                    leaf.id, ndim, leaf_output[i][0].shape[0],
                    leaf_output[i][2], leaf_output[i][0], leaf_output[i][1],
                    leaf_output[i][3], leaf_output[i][4],
                    leaf.start_idx, leaf.stop_idx)
            else:
                raise TypeError("Unsupported leaf type: {}".format(leaf_dtype))
            cons.add_leaf(sleaf)
    else:
        ncells_tot = sum(leaf_output)
        if use_double:
            cons = tools.ConsolidatedLeaves64(ndim, idx_inf, ncells_tot)
        else:
            cons = tools.ConsolidatedLeaves32(ndim, idx_inf, ncells_tot)
        for i, leaf in enumerate(tree.leaves):
            fname = _final_leaf_tess_filename(leaf.id, unique_str=unique_str)
            cons.add_leaf_fromfile(fname)
            os.remove(fname)
    cons.finalize()
    cells = cons.verts
    neigh = cons.neigh
    # if np.sum(neigh == idx_inf) != 0:
    #     for i in range(ncells):
    #         print(i, cells[i, :], neigh[i, :])
    # assert(np.sum(neigh == idx_inf) == 0)
    # Do tessellation
    T = Delaunay(np.zeros([0, ndim]), use_double=use_double)
    T.deserialize_with_info(pts, tree.idx.astype(cells.dtype),
                            cells, neigh, idx_inf)
    return T


def DelaunayProcessMPI(taskname, pts, tree=None,
                       left_edge=None, right_edge=None,
                       periodic=False, unique_str=None, use_double=False,
                       use_python=False, use_buffer=False, limit_mem=False,
                       suppress_final_output=False):
    r"""Get object for coordinating MPI operations.

    Args:
        See :class:`cgal4py.parallel.DelaunayProcessMPI_Python` and
            :class:`cgal4py.parallel.DelaunayProcessMPI_C` for information on
            arguments.

    Raises:
        ValueError: if `task` is not one of the accepted values listed above.

    Returns:
        :class:`cgal4py.parallel.DelaunayProcessMPI_Python` if
            `use_python == True`, :class:`cgal4py.parallel.DelaunayProcessMPI_C`
            otherwise.

    """
    if use_python:
        out = DelaunayProcessMPI_Python(
            taskname, pts, tree=tree, left_edge=left_edge,
            right_edge=right_edge, periodic=periodic, unique_str=unique_str,
            use_double=use_double, use_buffer=use_buffer,
            limit_mem=limit_mem, suppress_final_output=suppress_final_output)
    else:
        out = DelaunayProcessMPI_C(
            taskname, pts, left_edge=left_edge,
            right_edge=right_edge, periodic=periodic, unique_str=unique_str,
            use_double=use_double, limit_mem=limit_mem,
            suppress_final_output=suppress_final_output)
    return out


class DelaunayProcessMPI_C(object):
    r"""Class for coordinating MPI operations in C. This serves as a wrapper
    for :class:`cagl4py.delaunay.ParallelDelaunayD` to function the same as
    :class:`cgal4py.parallel.DelaunayProcessMPI_Python`.

    Args:
        taskname (str): Key for the task that should be parallelized. 
            Options:

            * 'triangulate': Perform triangulation and put serialized info in
              the output queue.
            * 'volumes': Perform triangulation and put volumes in output queue.

        pts (np.ndarray of float64): Array of coordinates to triangulate.
        left_edge (np.ndarray of float64, optional): Array of domain mins in
            each dimension. If not provided, they are determined from the
            points. Defaults to None.
        right_edge (np.ndarray of float64, optional): Array of domain maxes in
            each dimension. If not provided, they are determined from the
            points. Defaults to None.
        periodic (bool, optional): If True, the domain is assumed to be
            periodic at its left/right edges in each dimension. Defaults to
            False.
        unique_str (str, optional): Unique string identifying the domain
            decomposition that is passed to `cgal4py.parallel.ParallelLeaf` for
            file naming. Defaults to None.
        use_double (bool, optional): If True, 64 bit integers will be used for
            the triangulation. Defaults to False.
        limit_mem (bool, optional): If True, additional leaves are used and
            as each process cycles through its subset, leaves are written to/
            read from a file. Otherwise, all leaves are kept in memory at
            all times. Defaults to False.
        suppress_final_output (bool, optional): If True, output of the result
            to file is suppressed. This is mainly for testing purposes.
            Defaults to False.

    Raises:
        ValueError: if `task` is not one of the accepted values listed above.

    """
    def __init__(self, taskname, pts, left_edge=None, right_edge=None,
                 periodic=False, unique_str=None, use_double=False,
                 limit_mem=False, suppress_final_output=False):
        if not mpi_loaded:
            raise Exception("mpi4py could not be imported.")
        task_list = ['triangulate', 'volumes']
        if taskname not in task_list:
            raise ValueError('{} is not a valid task.'.format(taskname))
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        ndim = None
        if rank == 0:
            ndim = pts.shape[1]
            if left_edge is None:
                left_edge = pts.min(axis=0)
            if right_edge is None:
                right_edge = pts.max(axis=0)
        ndim = comm.bcast(ndim, root=0)
        Delaunay = _get_Delaunay(ndim, parallel=True, bit64=use_double)
        self.PT = Delaunay(left_edge, right_edge, periodic=periodic,
                           limit_mem=limit_mem)
        self.size = size
        self.rank = rank
        self.comm = comm
        self.pts = pts
        self.taskname = taskname
        self.unique_str = unique_str
        self.suppress_final_output = suppress_final_output

    def output_filename(self):
        if self.taskname == 'triangulate':
            fname = _tess_filename(unique_str=self.unique_str)
        elif self.taskname == 'volumes':
            fname = _vols_filename(unique_str=self.unique_str)
        return fname

    def run(self):
        r"""Perform necessary steps to complete the supplied task."""
        self.PT.insert(self.pts)
        if self.taskname == 'triangulate':
            T = self.PT.consolidate_tess()
            if (self.rank == 0):
                if not self.suppress_final_output:
                    ftess = self.output_filename()
                    with open(ftess, 'wb') as fd:
                         T.serialize_to_buffer(fd, pts)
        elif self.taskname == 'volumes':
            vols = self.PT.consolidate_vols()
            if (self.rank == 0):
                if not self.suppress_final_output:
                    fvols = self.output_filename()
                    with open(fvols, 'wb') as fd:
                        fd.write(vols.tobytes())


class DelaunayProcessMPI_Python(object):
    r"""Class for coordinating MPI operations in Python.

    Args:
        taskname (str): Key for the task that should be parallelized.
            Options:

            * 'triangulate': Perform triangulation and put serialized info in
              the output queue.
            * 'volumes': Perform triangulation and put volumes in output queue.

        pts (np.ndarray of float64): Array of coordinates to triangulate.
        tree (Tree, optional): Decomain decomposition tree. If not provided,
            :func:`cgal4py.domain_decomp.tree` is used to construct one.
            Defaults to None.
        left_edge (np.ndarray of float64, optional): Array of domain mins in
            each dimension. If not provided, they are determined from the
            points. Defaults to None. This is not required if `tree` is
            provided.
        right_edge (np.ndarray of float64, optional): Array of domain maxes in
            each dimension. If not provided, they are determined from the
            points. Defaults to None. This is not required if `tree` is
            provided.
        periodic (bool, optional): If True, the domain is assumed to be
            periodic at its left/right edges in each dimension. Defaults to
            False. This is not required if `tree` is provided.
        unique_str (str, optional): Unique string identifying the domain
            decomposition that is passed to `cgal4py.parallel.ParallelLeaf` for
            file naming. Defaults to None.
        use_double (bool, optional): If True, 64 bit integers will be used for
            the triangulation. Defaults to False.
        use_buffer (bool, optional): If True, communications are done by way of
            buffers rather than pickling python objects. Defaults to False.
        limit_mem (bool, optional): If True, additional leaves are used and
            as each process cycles through its subset, leaves are written to/
            read from a file. Otherwise, all leaves are kept in memory at
            all times. Defaults to False.
        suppress_final_output (bool, optional): If True, output of the result
            to file is suppressed. This is mainly for testing purposes.
            Defaults to False.

    Raises:
        ValueError: if `task` is not one of the accepted values listed above.

    """
    def __init__(self, taskname, pts, tree=None,
                 left_edge=None, right_edge=None,
                 periodic=False, unique_str=None, use_double=False,
                 use_buffer=False, limit_mem=False,
                 suppress_final_output=False):
        if not mpi_loaded:
            raise Exception("mpi4py could not be imported.")
        task_list = ['triangulate', 'volumes']
        if taskname not in task_list:
            raise ValueError('{} is not a valid task.'.format(taskname))
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        # Domain decomp
        task2leaves = None
        left_edges = None
        right_edges = None
        if rank == 0:
            if tree is None:
                tree = domain_decomp.tree('kdtree', pts, left_edge, right_edge,
                                          periodic=periodic, nleaves=size)
            if not isinstance(tree, GenericTree):
                tree = GenericTree.from_tree(tree)
            task2leaves = [[] for _ in range(size)]
            for leaf in tree.leaves:
                leaf.pts = pts[tree.idx[leaf.start_idx:leaf.stop_idx]]
                task = leaf.id % size
                task2leaves[task].append(leaf)
            left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
            right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        # Communicate points
        # TODO: Serialize & allow for use of buffer
        leaves = comm.scatter(task2leaves, root=0)
        pkg = (left_edges, right_edges, unique_str)
        left_edges, right_edges, unique_str = comm.bcast(pkg, root=0)
        nleaves = len(leaves) 
        # Set attributes
        self._task = taskname
        self._pts = pts
        self._tree = tree
        self._unique_str = unique_str
        self._limit_mem = limit_mem
        self._use_double = use_double
        self._use_buffer = use_buffer
        self._suppress_final_output = suppress_final_output
        self._comm = comm
        self._num_proc = size
        self._proc_idx = rank
        self._leaves = [ParallelLeaf(leaf, left_edges, right_edges,
                                     unique_str=unique_str,
                                     limit_mem=limit_mem) for leaf in leaves]
        self._leafid2idx = {leaf.id:i for i,leaf in enumerate(leaves)}
        ndim = left_edges.shape[1]
        self._ndim = ndim
        self._local_leaves = len(leaves)
        self._total_leaves = 0
        if self._local_leaves != 0:
            self._total_leaves = leaves[0].num_leaves
        self._done = False
        self._task2leaf = {i:[] for i in range(size)}
        for i in range(self._total_leaves):
            task = i % size
            self._task2leaf[task].append(i)
                
    def output_filename(self):
        if self._task == 'triangulate':
            fname = _tess_filename(unique_str=self._unique_str)
        elif self._task == 'volumes':
            fname = _vols_filename(unique_str=self._unique_str)
        return fname

    def get_leaf(self, leafid):
        r"""Return the leaf object associated wth a given leaf id.

        Args:
            leafid (int): Leaf ID.

        """
        return self._leaves[self._leafid2idx[leafid]]

    def tessellate_leaves(self):
        r"""Performs the tessellation for each leaf on this process."""
        for leaf in self._leaves:
            leaf.tessellate()

    def gather_leaf_arrays(self, local_arr, root=0):
        r"""Gather arrays for all leaves to a single process.

        Args:
            local_arr (dict): Arrays to be gathered for each leaf ID.
            root (int, optional): Process to which arrays should be gathered.
                Defaults to 0.

        Returns:
            dict: Arrays for each leaf ID.
        
        """
        total_arr = {}
        if self._use_buffer:  # pragma: no cover
            leaf_ids = list(local_arr.keys())
            np_dtype = local_arr[leaf_ids[0]].dtype
            mpi_dtype = _get_mpi_type(np_dtype)
            # Prepare things to send
            scnt = np.array([len(leaf_ids)], 'int64')
            sids = np.empty(2*len(leaf_ids), 'int64')
            for i, k in enumerate(leaf_ids):
                sids[2*i] = k
                sids[2*i+1] = local_arr[k].size
            sarr = np.concatenate([local_arr[k] for k in leaf_ids])
            if sarr.dtype != np_dtype:
                sarr = sarr.astype(np_dtype)
            # Send the number of leaves on each processor
            if self._proc_idx == root:
                rcnt = np.empty(self._num_proc, scnt.dtype)
            else:
                rcnt = np.array([], scnt.dtype)
            self._comm.Gather((scnt, _get_mpi_type(scnt.dtype)),
                              (rcnt, _get_mpi_type(rcnt.dtype)), root)
            tot_nleaves = rcnt.sum()
            # Send the ids and sizes of leaves
            rids = np.empty(2*tot_nleaves, sids.dtype)
            recv_buf = None
            if self._proc_idx == root:
                recv_buf = (rids, 2*rcnt, _get_mpi_type(rids.dtype))
            self._comm.Gatherv((sids, _get_mpi_type(sids.dtype)),
                               recv_buf, root)
            # Count number on each processor
            arr_counts = np.zeros(self._num_proc, 'int')
            if self._proc_idx == root:
                j = 1
                for i in range(self._num_proc):
                    for _ in range(rcnt[i]):
                        arr_counts[i] += rids[j]
                        j += 2
            # Send the arrays for each leaf
            rarr = np.empty(arr_counts.sum(), sarr.dtype)
            recv_buf = None
            if self._proc_idx == root:
                recv_buf = (rarr, arr_counts, _get_mpi_type(rarr.dtype))
            self._comm.Gatherv((sarr, _get_mpi_type(sarr.dtype)),
                               recv_buf, root)
            # Parse out info for each leaf
            if self._proc_idx == root:
                curr = 0
                for i in range(tot_nleaves):
                    j = 2*i
                    k = rids[j]
                    siz = rids[j + 1]
                    total_arr[k] = rarr[curr:(curr+siz)]
                    curr += siz
                assert(curr == rarr.size)
        else:
            data = self._comm.gather(local_arr, root=root)
            if self._proc_idx == root:
                for x in data:
                    total_arr.update(x)
        return total_arr

    def alltoall_leaf_arrays(self, local_arr, dtype=None, return_counts=False,
                             leaf_counts=None, array_counts=None):
        r"""Exchange arrays between leaves.

        Args:
            local_arr (dict): Arrays to be exchanged with other leaves. Keys
                are 2-element tuples. The first element is the leaf the array
                came from, the second element is the leaf the array should be 
                send to.

        Returns:
            dict: Incoming arrays where the keys are 2-element tuples as
                described above.
        
        """
        total_arr = {}
        if self._use_buffer:  # pragma: no cover
            local_leaf_ids = list(local_arr.keys())
            leaf_ids = [[] for i in range(self._num_proc)]
            local_array_count = 0
            for k in local_leaf_ids:
                task = k[1] % self._num_proc
                leaf_ids[task].append(k)
                local_array_count += local_arr[k].size
            # Get data type
            if len(local_leaf_ids) == 0:
                if dtype is None:
                    raise Exception("Nothing being sent from this process. " +
                                    "Cannot determine type to be recieved.")
                np_dtype = dtype
            else:
                np_dtype = local_arr[local_leaf_ids[0]].dtype
            mpi_dtype = _get_mpi_type(np_dtype)
            # Compute things to send
            scnt = np.array([len(x) for x in leaf_ids], 'int64')
            nleaves_send = scnt.sum()
            array_counts_send = np.zeros(self._num_proc, 'int64')
            sids = np.empty(3*nleaves_send, 'int64')
            sarr = np.empty(local_array_count, np_dtype)
            pl = pa = 0  # Entry for previous leaf/array entry
            for i, x in enumerate(leaf_ids):
                for k in x:
                    array_counts_send[i] += local_arr[k].size
                    sids[pl:(pl+2)] = k
                    sids[pl+2] = local_arr[k].size
                    sarr[pa:(pa+local_arr[k].size)] = local_arr[k]
                    pa += local_arr[k].size
                    pl += 3
            # Send the number of leaves on each processor
            if leaf_counts is None:
                rcnt = np.empty(self._num_proc, scnt.dtype)
                self._comm.Alltoall(
                    (scnt, _get_mpi_type(scnt.dtype)),
                    (rcnt, 1, _get_mpi_type(rcnt.dtype)))
            else:
                assert(len(leaf_counts) == self._num_proc)
                rcnt = leaf_counts
            nleaves_recv = rcnt.sum()
            # Send the ids of leaves for each leaf source/destination
            rids = np.empty(3*nleaves_recv, sids.dtype)
            send_buf = (sids, 3*scnt,
                        _get_mpi_type(sids.dtype))
            recv_buf = (rids, 3*rcnt,
                        _get_mpi_type(rids.dtype))
            self._comm.Alltoallv(send_buf, recv_buf)
            # Send the size of arrays for each leaf
            array_counts_recv = np.zeros(self._num_proc, 'int64')
            j = 0
            for i in range(self._num_proc):
                for _ in range(rcnt[i]):
                    array_counts_recv[i] += rids[3*j + 2]
                    j += 1
            # Send the arrays for each leaf
            rarr = np.empty(array_counts_recv.sum(), sarr.dtype)
            send_buf = (sarr, array_counts_send, _get_mpi_type(sarr.dtype))
            recv_buf = (rarr, array_counts_recv, _get_mpi_type(rarr.dtype))
            self._comm.Alltoallv(send_buf, recv_buf)
            # Parse out info for each leaf
            curr = 0
            for i in range(nleaves_recv):
                src = rids[3*i]
                dst = rids[3*i+1]
                siz = rids[3*i+2]
                total_arr[(src, dst)] = rarr[curr:(curr+siz)]
                curr += siz
            assert(curr == rarr.size)
        else:
            return_counts = False
            local_leaf_ids = list(local_arr.keys())
            send_data = [{} for i in range(self._num_proc)]
            for k in local_leaf_ids:
                task = k[1] % self._num_proc
                send_data[task][k] = local_arr[k]
            recv_data = self._comm.alltoall(send_data)
            for x in recv_data:
                total_arr.update(**x)
        if return_counts:
            return total_arr, rcnt, array_counts_recv
        return total_arr

    def outgoing_points(self):
        r"""Enqueues points at edges of each leaf's boundaries."""
        if self._use_buffer:  # pragma: no cover
            send_int = {}
            send_flt = {}
            for leaf in self._leaves:
                src = leaf.id
                hvall, n, le, re, ptall = leaf.outgoing_points(return_pts=True)
                for dst in range(self._total_leaves):
                    task = dst % self._num_proc
                    if hvall[dst] is not None:
                        k = (src, dst)
                        send_int[k] = np.concatenate(
                            [np.array([hvall[dst].shape[0]], 'int64'),
                             hvall[dst], n]).astype('int64')
                        send_flt[k] = np.concatenate(
                            [ptall[dst].flatten(),
                             le.flatten(), re.flatten()]).astype('float64')
            out_int = self.alltoall_leaf_arrays(send_int, dtype='int64',
                                                return_counts=True)
            recv_int, leaf_counts, array_counts_int = out_int
            recv_flt = self.alltoall_leaf_arrays(send_flt, dtype='float64',
                                                 return_counts=False,
                                                 leaf_counts=leaf_counts)
            tot_recv = {}
            for k in recv_int.keys():
                npts = int(recv_int[k][0])
                ndim = self._ndim
                nn = recv_int[k].size - (npts+1)
                tot_recv[k] = {
                    'idx': recv_int[k][1:(npts+1)],
                    'n': recv_int[k][(npts+1):],
                    'pts': recv_flt[k][:(ndim*npts)].reshape(npts, ndim),
                    'le': recv_flt[k][
                        (ndim*npts):(ndim*(npts+nn))].reshape(nn, ndim),
                    're': recv_flt[k][
                        (ndim*(npts+nn)):(ndim*(npts+2*nn))].reshape(nn, ndim)}
            self._tot_recv = tot_recv
        else:
            tot_send = [{k:{} for k in self._task2leaf[i]} for
                        i in range(self._num_proc)]
            for leaf in self._leaves:
                hvall, n, le, re, ptall = leaf.outgoing_points(return_pts=True)
                for i in range(self._total_leaves):
                    task = i % self._num_proc
                    if hvall[i] is None:
                        tot_send[task][i][leaf.id] = None
                    else:
                        tot_send[task][i][leaf.id] = (hvall[i], n, le, re,
                                                      ptall[i])
            self._tot_recv = self._comm.alltoall(sendobj=tot_send)
            del tot_send

    def incoming_points(self):
        r"""Takes points from the queue and adds them to the triangulation."""
        if self._use_buffer:  # pragma: no cover
            nrecv = 0
            for k, v in self._tot_recv.items():
                src, dst = k
                leaf = self.get_leaf(dst)
                leaf.incoming_points(src, v['idx'], v['n'], 
                                     v['le'], v['re'], v['pts'])
                nrecv += v['idx'].size
            del self._tot_recv
        else:
            nrecv = 0
            for leaf in self._leaves:
                for k in range(self._total_leaves):
                    task = k % self._num_proc
                    if k not in self._tot_recv[task][leaf.id]:
                        continue
                    if self._tot_recv[task][leaf.id][k] is None:
                        continue
                    leaf.incoming_points(k, *self._tot_recv[task][leaf.id][k])
                    nrecv += self._tot_recv[task][leaf.id][k][0].size
            del self._tot_recv
        return nrecv

    def enqueue_triangulation(self):
        r"""Enqueue resulting tessellation."""
        if self._use_buffer:  # pragma: no cover
            send_arr = {}
            for leaf in self._leaves:
                send_arr[leaf.id] = leaf.serialize(as_single_array=True)
            recv_arr = self.gather_leaf_arrays(send_arr)
            if self._proc_idx == 0:
                serial = [None for i in range(self._total_leaves)]
                ndim = self._ndim
                for k, v in recv_arr.items():
                    ncell = int(v[0])
                    ncell_tot = int(v[1])
                    idx_inf = int(v[2])
                    # Cells
                    beg = 3
                    end = beg + ncell*(ndim+1)
                    cells = v[beg:end].reshape(ncell, ndim+1).astype('uint64')
                    # Neighbors
                    beg = end
                    end = beg + ncell*(ndim+1)
                    neigh = v[beg:end].reshape(ncell, ndim+1).astype('uint64')
                    # Sort index for verts
                    beg = end
                    end = beg + ncell*(ndim+1)
                    idx_verts = v[beg:end].reshape(ncell, ndim+1).astype(
                        'uint32')
                    # Sort index for cells
                    beg = end
                    end = beg + ncell
                    idx_cells = v[beg:end].astype('uint64')
                    serial[k] = (cells, neigh, idx_inf, idx_verts, idx_cells,
                                 ncell_tot)
                    assert(end == v.size)
                serial1 = serial
        else:
            out = [(leaf.id, leaf.serialize()) for leaf in self._leaves]
            out = self._comm.gather(out, root=0)
            if self._proc_idx == 0:
                serial = [None for i in range(self._total_leaves)]
                for i in range(self._num_proc):
                    for iid, s in out[i]:
                        serial[iid] = s
                serial2 = serial
        if self._proc_idx == 0:
            T = consolidate_tess(self._tree, serial, self._pts,
                                 use_double=self._use_double,
                                 unique_str=self._unique_str)
                                 # limit_mem=limit_mem)
            if not self._suppress_final_output:
                ftess = _tess_filename(unique_str=self._unique_str)
                # T.write_to_file(ftess)
                with open(ftess, 'wb') as fd:
                    T.serialize_to_buffer(fd, self._pts)

    def enqueue_volumes(self):
        r"""Enqueue resulting voronoi volumes."""
        local = {leaf.id: leaf.voronoi_volumes() for leaf in self._leaves}
        if self._use_buffer:  # pragma: no cover
            total = self.gather_leaf_arrays(local)
            if self._proc_idx == 0:
                # Preallocate
                tot = 0
                dtype = 'float64'
                for k in total.values():
                    tot += k.size
                vol = np.empty(tot, dtype)
                # Transfer values
                tree = self._tree
                for k in total.keys():
                    leaf = tree.leaves[k]
                    vol[tree.idx[leaf.start_idx:leaf.stop_idx]] = total[k]
        else:
            out = self._comm.gather(local, root=0)
            if self._proc_idx == 0:
                # Preallocate
                tot = 0
                dtype = None
                for x in out:
                    for k in x.values():
                        tot += k.size
                        dtype = k.dtype
                vol = np.empty(tot, dtype)
                # Transfer values
                for x in out:
                    for k in x.keys():
                        leaf = self._tree.leaves[k]
                        vol[self._tree.idx[leaf.start_idx:leaf.stop_idx]] = x[k]
        if self._proc_idx == 0:
            if not self._suppress_final_output:
                # Save volumes
                fvols = _vols_filename(unique_str=self._unique_str)
                # np.save(fvols, vol)
                with open(fvols, 'wb') as fd:
                    fd.write(vol.tobytes())

    def run(self):
        r"""Performs tessellation and communication for each leaf on this
        process."""
        np_dtype = 'int64'
        mpi_dtype = _get_mpi_type(np_dtype)
        if self._task in ['triangulate', 'volumes']:
            self.tessellate_leaves()
            self._comm.Barrier()
            # Continue exchanges until there are not any particles that need to
            # be exchanged.
            nrecv = -1
            while nrecv != 0:
                self.outgoing_points()
                nrecv0 = self.incoming_points()
                if self._use_buffer:  # pragma: no cover
                    nrecv_local = np.array([nrecv0], np_dtype)
                    nrecv_total = None
                    if self._proc_idx == 0:
                        nrecv_total = np.empty(self._num_proc, np_dtype)
                    self._comm.Gather((nrecv_local, mpi_dtype),
                                      (nrecv_total, mpi_dtype), root=0)
                    nrecv_tot = np.empty(1, np_dtype)
                    if self._proc_idx == 0:
                        nrecv_tot[0] = sum(nrecv_total)
                    self._comm.Bcast(nrecv_tot, root=0)
                    nrecv = nrecv_tot[0]
                else:
                    nrecv_total = self._comm.gather(nrecv0, root=0)
                    nrecv_tot = None
                    if self._proc_idx == 0:
                        nrecv_tot = sum(nrecv_total)
                    nrecv = self._comm.bcast(nrecv_tot, root=0)
            if self._task == 'triangulate':
                self.enqueue_triangulation()
            elif self._task == 'volumes':
                self.enqueue_volumes()
        self._done = True
        # Clean up leaves
        if self._limit_mem:
            for leaf in self._leaves:
                leaf.remove_tess()


if _use_multiprocessing:
    class DelaunayProcessMulti(mp_Process):
        r"""`multiprocessing.Process` subclass for coordinating operations on a
        single process during a parallel Delaunay triangulation.

        Args:
            task (str): Key for the task that should be parallelized. Options
                are:

                * 'triangulate': Perform triangulation and put serialized info in
                  the output queue.
                * 'volumes': Perform triangulation and put volumes in output queue.

            proc_idx (int): Index of this process.
            leaves (list of leaf objects): Leaves that should be triangulated on
                this process. The leaves are created by
                :meth:`cgal4py.domain_decomp.tree`.
            pts (np.ndarray of float64): (n,m) array of n m-dimensional
                coordinates. Each leaf has a set of indices identifying coordinates
                within `pts` that belong to that leaf.
            left_edges (np.ndarray float64): Array of mins for all leaves in the
                domain decomposition.
            right_edges (np.ndarray float64): Array of maxes for all leaves in the
                domain decomposition.
            queues (list of `multiprocessing.Queue`): List of queues for every
                process being used in the triangulation plus one for the main
                process.
            lock (multiprocessing.Lock): Lock for processes.
            count (multiprocessing.Value): Shared integer for tracking exchanged
                points.
            pipe (multiprocessing.Pipe): Input end of pipe that is connected to the
                master process.
            unique_str (str, optional): Unique string identifying the domain
                decomposition that is passed to `cgal4py.parallel.ParallelLeaf` for
                file naming. Defaults to None.
            limit_mem (bool, optional): If False, the triangulation results from
                each process are moved to local memory using `multiprocessing`
                pipes. If True, each process writes out tessellation info to
                files which are then incrementally loaded as consolidation occurs.
                Defaults to False.
            **kwargs: Variable keyword arguments are passed to
                `multiprocessing.Process`.

        Raises:
            ValueError: if `task` is not one of the accepted values listed above.

        """
        def __init__(self, task, proc_idx, leaves, pts, idx,
                     left_edges, right_edges, queues, lock, count, pipe,
                     unique_str=None, limit_mem=False, **kwargs):
            task_list = ['triangulate', 'volumes', 'output']
            if task not in task_list:
                raise ValueError('{} is not a valid task.'.format(task))
            super(DelaunayProcessMulti, self).__init__(**kwargs)
            self._task = task
            self._leaves = [ParallelLeaf(leaf, left_edges, right_edges,
                                         unique_str=unique_str,
                                         limit_mem=limit_mem) for leaf in leaves]
            self._leafid2idx = {leaf.id:i for i,leaf in enumerate(leaves)}
            self._idx = np.frombuffer(idx, dtype='uint64')
            self._ptsFlt = np.frombuffer(pts, dtype='float64')
            ndim = left_edges.shape[1]
            npts = len(self._ptsFlt)/ndim
            self._ndim = ndim
            self._pts = self._ptsFlt.reshape(npts, ndim)
            self._queues = queues
            self._lock = lock
            self._count = count
            self._pipe = pipe
            self._unique_str = unique_str
            self._limit_mem = limit_mem
            self._num_proc = len(queues)-1
            self._local_leaves = len(leaves)
            self._total_leaves = 0
            if self._local_leaves != 0:
                self._total_leaves = leaves[0].num_leaves
            self._proc_idx = proc_idx
            self._done = False

        def get_leaf(self, leafid):
            r"""Return the leaf object associated wth a given leaf id.

            Args:
                leafid (int): Leaf ID.

            """
            return self._leaves[self._leafid2idx[leafid]]

        def tessellate_leaves(self):
            r"""Performs the tessellation for each leaf on this process."""
            for leaf in self._leaves:
                leaf.tessellate(self._pts, idx=self._idx)

        def outgoing_points(self):
            r"""Enqueues points at edges of each leaf's boundaries."""
            for leaf in self._leaves:
                hvall, n, le, re = leaf.outgoing_points()
                for i in range(self._total_leaves):
                    task = i % self._num_proc
                    if hvall[i] is None:
                        self._queues[task].put(None)
                    else:
                        self._queues[task].put((i, leaf.id, hvall[i], n, le, re))
                    time.sleep(0.01)

        def incoming_points(self):
            r"""Takes points from the queue and adds them to the triangulation."""
            queue = self._queues[self._proc_idx]
            count = 0
            nrecv = 0
            while count < (self._local_leaves*self._total_leaves):
                count += 1
                time.sleep(0.01)
                out = queue.get()
                if out is None:
                    continue
                i, j, arr, n, le, re = out
                if (arr is not None) and (arr.shape[0] != 0):
                    # Find leaf this should go to
                    for leaf in self._leaves:
                        if leaf.id == i:
                            break
                    # Add points to leaves
                    new_pts = np.copy(self._pts[self._idx[arr], :])
                    leaf.incoming_points(j, arr, n, le, re, new_pts)
                    nrecv += arr.shape[0]
            with self._count[1].get_lock():
                self._count[1].value += nrecv

        def enqueue_result(self):
            r"""Enqueue the appropriate result for the given task."""
            if self._task == 'triangulate':
                if self._limit_mem:
                    self.enqueue_number_of_cells()
                else:
                    self.enqueue_triangulation()
            elif self._task == 'volumes':
                self.enqueue_volumes()

        def receive_result(self, pipe):
            r"""Return serialized info from a pipe that was placed there by
            `enqueue_result`.

            Args:
                pipe (`multiprocessing.Connection`): Receiving end of pipe.

            """
            if self._task == 'triangulate':
                if self._limit_mem:
                    out = self.receive_number_of_cells(pipe)
                else:
                    out = self.receive_triangulation(pipe)
            elif self._task == 'volumes':
                out = self.receive_volumes(pipe)
            return out

        def enqueue_triangulation(self):
            r"""Enqueue resulting tessellation."""
            for leaf in self._leaves:
                s = leaf.serialize()
                if s[0].dtype == np.uint32:
                    dt = 0
                elif s[0].dtype == np.uint64:
                    dt = 1
                # elif s[0].dtype == np.int32:
                #     dt = 2
                # elif s[0].dtype == np.int64:
                #     dt = 3
                else:
                    raise Exception("No type found for {}".format(s[0].dtype))
                self._pipe.send_bytes(
                    struct.pack('QQQQQ', leaf.id, s[0].shape[0], dt, s[2], s[5]))
                for _ in range(2) + range(3, 5):
                    self._pipe.send_bytes(s[_])

        def receive_triangulation(self, pipe):
            r"""Return serialized info from a pipe that was placed there by
            `enqueue_triangulation`.

            Args:
                pipe (`multiprocessing.Connection`): Receiving end of pipe.

            """
            dt2dtype = {0: np.uint32, 1: np.uint64, 2: np.int32, 3: np.int64}
            dummy_head = np.empty(5, 'uint64')
            pipe.recv_bytes_into(dummy_head)
            iid, ncell, dt, idx_inf, ncell_tot = dummy_head
            dtype = dt2dtype[dt]
            s = [np.empty((ncell, self._ndim+1), dtype),
                 np.empty((ncell, self._ndim+1), dtype),
                 dtype(idx_inf),
                 np.empty((ncell, self._ndim+1), 'uint32'),
                 np.empty(ncell, 'uint64'),
                 ncell_tot]
            for _ in range(2) + range(3, 5):
                pipe.recv_bytes_into(s[_])
            s = tuple(s)
            return iid, s

        def enqueue_volumes(self):
            r"""Enqueue resulting voronoi volumes."""
            for leaf in self._leaves:
                self._pipe.send_bytes(struct.pack('Q', leaf.id))
                self._pipe.send_bytes(leaf.voronoi_volumes())

        def receive_volumes(self, pipe):
            r"""Return serialized info from a pipe that was placed there by
            `enqueue_volumes`.

            Args:
                pipe (`multiprocessing.Connection`): Receiving end of pipe.

            """
            dummy_head = np.empty(1, 'uint64')
            pipe.recv_bytes_into(dummy_head)
            iid = dummy_head[0]
            ivol = np.empty(self.get_leaf(iid).npts, 'float64')
            pipe.recv_bytes_into(ivol)
            return iid, ivol

        def enqueue_number_of_cells(self):
            r"""Enqueue resulting number of cells."""
            for leaf in self._leaves:
                ncells = leaf.write_tess_to_file()
                self._pipe.send_bytes(struct.pack('QQ', leaf.id, ncells))

        def receive_number_of_cells(self, pipe):
            r"""Return serialized info from a pipe that was placed there by
            `enqueue_number_of_cells`.

            Args:
                pipe (`multiprocessing.Connection`): Receiving end of pipe.

            """
            dummy_head = np.empty(2, 'uint64')
            pipe.recv_bytes_into(dummy_head)
            return dummy_head[0], dummy_head[1]

        def output_tess(self):
            r"""Write serialized tessellation info to file for each leaf."""
            for leaf in self._leaves:
                ncells = leaf.write_tess_to_file()

        def run(self, test_in_serial=False):
            r"""Performs tessellation and communication for each leaf on this
            process."""
            if self._task in ['triangulate', 'volumes']:
                self.tessellate_leaves()
                # Continue exchanges until there are not any particles that need to
                # be exchanged.
                while True:
                    with self._count[2].get_lock():
                        if self._count[2].value == 1:
                            break
                    # print('Begin', self._proc_idx, self._count[0].value,
                    #       self._count[1].value, self._count[2].value)
                    self.outgoing_points()
                    self.incoming_points()
                    self._lock.acquire()
                    with self._count[0].get_lock():
                        self._count[0].value += 1
                    # print('Lock acquired: {}/{}'.format(self._count[0].value,
                    #                                     self._num_proc),
                    #       self._count[1].value)
                    if self._count[0].value < self._num_proc:
                        self._lock.wait()
                    else:
                        with self._count[1].get_lock():
                            if self._count[1].value > 0:
                                with self._count[0].get_lock():
                                    self._count[0].value = 0
                                self._count[1].value = 0
                            else:
                                with self._count[0].get_lock():
                                    self._count[0].value = 0
                                self._count[1].value = 0
                                with self._count[2].get_lock():
                                    self._count[2].value = 1
                        self._lock.notify_all()
                    self._lock.release()
                    # print 'Lock released', self._proc_idx,self._count[2].value
                    if test_in_serial:
                        with self._count[2].get_lock():
                            self._count[2].value = 1
                self.enqueue_result()
            # Clean up leaves
            for leaf in self._leaves:
                leaf.remove_tess()
            # Synchronize to ensure rapid receipt
            if test_in_serial:
                with self._count[0].get_lock():
                    self._count[0].value = self._num_proc - 1
            self._lock.acquire()
            with self._count[0].get_lock():
                self._count[0].value += 1
            if self._count[0].value < self._num_proc:
                self._lock.wait()
            else:
                self._lock.notify_all()
            self._lock.release()
            self._done = True


class ParallelLeaf(object):
    r"""Wraps triangulation of a single leaf in a domain decomposition.

    Args:
        leaf (object): Leaf object from a tree returned by
            :meth:`cgal4py.domain_decomp.tree`.
        left_edges (np.ndarray): Minimums of each leaf in the domain
            decomposition.
        right_edges (np.ndarray): Maximums of each leaf in the domain
            decomposition.
        unique_str (str, optional): Unique string identifying the domain
            decomposition that will be used to construct an output file name.
            Default to None.
        limit_mem (bool, optional): If True, triangulations for this leaf are
            only held in memory so long as they are needed and then are output
            to a file. Otherwise, the triangualtions are kept in local memory.

    Attributes:
        norig (int): The number of points originally located on this leaf.
        T (:class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`:): 2D or 3D triangulation
            object.
        idx (np.ndarray of uint64): Indices of points on this leaf in the
            domain sorted position array (including those points transfered
            from other leaves).
        all_neighbors (set): Indices of all leaves that have been considered
            during particle exchanges.
        neighbors (list): Neighboring leaves that will be considered during the
            next particle exchange.
        left_neighbors (list): Neighboring leaves to the left of this leaf in
            each dimension.
        right_neighbors (list): Neighboring leaves to the right of this leaf in
            each dimension.
        left_edges (np.ndarray): Minimums of the domains in each dimension for
            leaves in `self.neighbors`.
        right_edges (np.ndarray): Maximums of the domains in each dimension for
            leaves in `self.neighbors`.
        unique_str (str): Unique string identifying the domain decomposition
            that will be used to construct an output file name.
        limit_mem (bool): If True, triangulations for this leaf are only held
            in memory so long as they are needed and then are output to a file.
            Otherwise, the triangualtions are kept in local memory.
        All attributes of `leaf`'s class also apply.

    """

    def __init__(self, leaf, left_edges, right_edges, unique_str=None,
                 limit_mem=False):
        self._leaf = leaf
        self.norig = leaf.npts
        self.T = None
        dtype = 'uint64'
        if 10*leaf.stop_idx < np.iinfo('uint32').max:
            dtype = 'uint32'
        self.idx = np.arange(leaf.start_idx,
                             leaf.stop_idx).astype(dtype)
        self.all_neighbors = set([])
        self.neighbors = copy.deepcopy(leaf.neighbors)
        keep = False
        for i in range(self.ndim):
            if leaf.id in leaf.left_neighbors[i]:
                keep = True
                break
            if leaf.id in leaf.right_neighbors[i]:
                keep = True
                break
        if not keep:
            self.neighbors.remove(leaf.id)
        self.left_neighbors = copy.deepcopy(leaf.left_neighbors)
        self.right_neighbors = copy.deepcopy(leaf.right_neighbors)
        le = copy.deepcopy(left_edges)
        re = copy.deepcopy(right_edges)
        for i in range(self.ndim):
            if self.periodic_left[i]:
                for k in leaf.left_neighbors[i]:
                    le[k, i] -= self.domain_width[i]
                    re[k, i] -= self.domain_width[i]
            if self.periodic_right[i]:
                for k in leaf.right_neighbors[i]:
                    le[k, i] += self.domain_width[i]
                    re[k, i] += self.domain_width[i]
        self.left_edges = le[self.neighbors, :]
        self.right_edges = re[self.neighbors, :]
        self.unique_str = unique_str
        self.limit_mem = limit_mem

    def __getattr__(self, name):
        if name in dir(self._leaf):
            return getattr(self._leaf, name)
        else:
            raise AttributeError

    @property
    def tess_filename(self):
        r"""The default filename that should be used for tessellation
        output."""
        return _leaf_tess_filename(self.id, unique_str=self.unique_str)

    @property
    def tess_output_filename(self):
        r"""The default filename that should be used for tessellation
        output."""
        return _final_leaf_tess_filename(self.id, unique_str=self.unique_str)

    def save_tess(self, fname=None):
        r"""Save the tessellation data for this leaf to a file and then remove
        it from memory.

        Args:
            fname (str, optional): Full path to file where tessellation data
                should be saved. If not provided `self.tess_filename` is used.
                Defaults to None.

        """
        if fname is None:
            fname = self.tess_filename
        self.T.write_to_file(fname)
        self.T = None

    def load_tess(self, fname=None):
        r"""Load the tessellation data for this leaf from a file.

        Args:
            fname (str, optional): Full path to file where tessellation data
                should be loaded from. If not provided `self.tess_filename` is
                used. Defaults to None.

        """
        if fname is None:
            fname = self.tess_filename
        self.T = self.Tclass.from_file(fname)

    def remove_tess(self, fname=None):
        r"""Remove the tessellation data for this leaf that is saved to a file.

        Args:
            fname (str, optional): Full path to file where tessellation data
                is stored. If not provided `self.tess_filename` is used.
                Defaults to None.

        """
        if fname is None:
            fname = self.tess_filename
        if os.path.isfile(fname):
            os.remove(fname)

    def tessellate(self, pts=None, idx=None):
        r"""Perform tessellation on leaf.

        Args:
            pts (np.ndarray of float64): (n,m) array of n m-dimensional
                coordinates.

        """
        if pts is None:
            new_pts = self.pts
        else:
            if idx is None:
                new_pts = np.copy(pts[self.slice, :])
            else:
                new_pts = np.copy(pts[idx[self.slice], :])
        self.T = Delaunay(new_pts)
        self.Tclass = self.T.__class__
        if self.limit_mem:
            self.save_tess()

    def outgoing_points(self, return_pts=False):
        r"""Get indices of points that should be sent to each neighbor.

        Args:
            return_pts (bool, optional): If True, the associated positions of
                the points are also returned. Defaults to False.
        
        Returns:
            tuple: Containing

                * hvall (list): List of tree indices that should be sent to each
                  process.
                * n (list): Indices of neighbor leaves.
                * left_edges (np.ndarray of float64): Left edges of neighbor
                  leaves.
                * right_edges (np.ndarray of float64): Right edges of neighbor
                  leaves.

        """
        if self.limit_mem:
            self.load_tess()
        n = self.neighbors
        le = self.left_edges
        re = self.right_edges
        idx_enq = self.T.outgoing_points(le, re)
        # Remove points that are not local
        for i in range(len(n)):
            ridx = (idx_enq[i] < self.norig)
            idx_enq[i] = idx_enq[i][ridx]
        # Translate and add entries for non-neighbors
        hvall = [None for k in range(self.num_leaves)]
        for i, k in enumerate(n):
            hvall[k] = self.idx[idx_enq[i]]
        # Reset neighbors for incoming
        self.all_neighbors.update(self.neighbors)
        self.neighbors = []
        self.left_edges = np.zeros((0, self.ndim), 'float64')
        self.right_edges = np.zeros((0, self.ndim), 'float64')
        # Return correct structure
        if return_pts:
            ptall = [None for k in range(self.num_leaves)]
            for i, k in enumerate(n):
                ptall[k] = self.pts[idx_enq[i]]
            out = (hvall, n, le, re, ptall)
        else:
            out = (hvall, n, le, re)
        if self.limit_mem:
            self.save_tess()
        return out

    # def outgoing_points_boundary(self):
    #     r"""Get indices of points that should be sent to each neighbor."""
    #     # TODO: Check that iind does not matter. iind contains points in tets
    #     # that are infinite. For non-periodic boundary conditions, these points
    #     # may need to be sent to distant leaves for an accurate convex hull.
    #     # Currently points on an edge without a bordering leaf are sent to all
    #     # leaves, but it is possible this could miss a few points...
    #     if self.limit_mem:
    #         self.load_tess()
    #     lind, rind, iind = self.T.boundary_points(self.left_edge,
    #                                               self.right_edge,
    #                                               True)
    #     # Remove points that are not local
    #     for i in range(self.ndim):
    #         ridx = (rind[i] < self.norig)
    #         lidx = (lind[i] < self.norig)
    #         rind[i] = rind[i][ridx]
    #         lind[i] = lind[i][lidx]
    #     # Count for preallocation
    #     all_leaves = range(0, self.id) + range(self.id + 1, self.num_leaves)
    #     Nind = np.zeros(self.num_leaves, 'uint32')
    #     for i in range(self.ndim):
    #         l_neighbors = self.left_neighbors[i]
    #         r_neighbors = self.right_neighbors[i]
    #         if len(l_neighbors) == 0:
    #             l_neighbors = all_leaves
    #         if len(r_neighbors) == 0:
    #             r_neighbors = all_leaves
    #         Nind[np.array(l_neighbors, 'uint32')] += len(lind[i])
    #         Nind[np.array(r_neighbors, 'uint32')] += len(rind[i])
    #     # Add points
    #     ln_out = [[[] for _ in range(self.ndim)] for
    #               k in range(self.num_leaves)]
    #     rn_out = [[[] for _ in range(self.ndim)] for
    #               k in range(self.num_leaves)]
    #     hvall = [np.zeros(Nind[k], rind[0].dtype) for
    #              k in range(self.num_leaves)]
    #     Cind = np.zeros(self.num_leaves, 'uint32')
    #     for i in range(self.ndim):
    #         l_neighbors = self.left_neighbors[i]
    #         r_neighbors = self.right_neighbors[i]
    #         if len(l_neighbors) == 0:
    #             l_neighbors = all_leaves
    #         if len(r_neighbors) == 0:
    #             r_neighbors = all_leaves
    #         ilN = len(lind[i])
    #         irN = len(rind[i])
    #         for k in l_neighbors:
    #             hvall[k][Cind[k]:(Cind[k]+ilN)] = lind[i]
    #             Cind[k] += ilN
    #             for j in range(0, i) + range(i + 1, self.ndim):
    #                 rn_out[k][i] += self._leaf.left_neighbors[j]
    #             for j in range(self.ndim):
    #                 rn_out[k][i] += self._leaf.right_neighbors[j]
    #         for k in r_neighbors:
    #             hvall[k][Cind[k]:(Cind[k]+irN)] = rind[i]
    #             Cind[k] += irN
    #             for j in range(0, i) + range(i + 1, self.ndim):
    #                 ln_out[k][i] += self._leaf.right_neighbors[j]
    #             for j in range(self.ndim):
    #                 ln_out[k][i] += self._leaf.left_neighbors[j]
    #     # Ensure unique values (overlap can happen if a point is at a corner)
    #     for k in range(self.num_leaves):
    #         hvall[k] = self.idx[np.unique(hvall[k])]
    #         for i in range(self.ndim):
    #             ln_out[k][i] = list(set(ln_out[k][i]))
    #             rn_out[k][i] = list(set(rn_out[k][i]))
    #     if self.limit_mem:
    #         self.save_tess()
    #     return hvall, ln_out, rn_out

    def incoming_points(self, leafid, idx, n, le, re, pos):
        r"""Add incoming points from other leaves.

        Args:
            leafid (int): ID for the leaf that points came from.
            idx (np.ndarray of int): Indices of points being recieved.
            n (list of int): Indices of new neighbor leaves to add.
            le (np.ndarray of float64): Mins of new neighbor leaves in each
                dimension.
            re (np.ndarray of float64): Maxes of new neighbor leaves in each
                dimension.
            pos (np.ndarray of float): Positions of points being recieved.

        """
        if idx is None or idx.shape[0] == 0:
            return
        if self.limit_mem:
            self.load_tess()
        # Wrap points
        if self.id == leafid:
            for i in range(self.ndim):
                if self.periodic_left[i] and self.periodic_right[i]:
                    idx_left = ((pos[:, i] - self.left_edge[i]) <
                                (self.right_edge[i] - pos[:, i]))
                    idx_right = ((self.right_edge[i] - pos[:, i]) <
                                 (pos[:, i] - self.left_edge[i]))
                    pos[idx_left, i] += self.domain_width[i]
                    pos[idx_right, i] -= self.domain_width[i]
        else:
            for i in range(self.ndim):
                if self.periodic_right[i] and leafid in self.right_neighbors[i]:
                    idx_left = ((pos[:, i] + self.domain_width[i] -
                                 self.right_edge[i]) <
                                (self.left_edge[i] - pos[:, i]))
                    pos[idx_left, i] += self.domain_width[i]
                if self.periodic_left[i] and leafid in self.left_neighbors[i]:
                    idx_right = ((self.left_edge[i] - pos[:, i] +
                                  self.domain_width[i]) <
                                 (pos[:, i] - self.right_edge[i]))
                    pos[idx_right, i] -= self.domain_width[i]
        # Concatenate arrays
        self.idx = np.concatenate([self.idx, idx.astype(self.idx.dtype)])
        # Insert points
        self.T.insert(pos)
        # Add neighbors
        for i in range(len(n)):
            if (n[i] != self.id) and (n[i] not in self.all_neighbors):
                self.neighbors.append(n[i])
                self.left_edges = np.vstack([self.left_edges, le[i, :]])
                self.right_edges = np.vstack([self.right_edges, re[i, :]])
        if self.limit_mem:
            self.save_tess()

    # def incoming_points_boundary(self, leafid, idx, ln, rn, pos):
    #     r"""Add incoming points from other leaves.

    #     Args:
    #         leafid (int): ID for the leaf that points came from.
    #         idx (np.ndarray of int): Indices of points being recieved.
    #         rn (list of lists): Right neighbors that should be added in each
    #             dimension.
    #         ln (list of lists): Left neighbors that should be added in each
    #             dimension.
    #         pos (np.ndarray of float): Positions of points being recieved.

    #     """
    #     if idx.shape[0] == 0:
    #         return
    #     if self.limit_mem:
    #         self.load_tess()
    #     # Wrap points
    #     if self.id == leafid:
    #         for i in range(self.ndim):
    #             if self.periodic_left[i] and self.periodic_right[i]:
    #                 idx_left = ((pos[:, i] - self.left_edge[i]) <
    #                             (self.right_edge[i] - pos[:, i]))
    #                 idx_right = ((self.right_edge[i] - pos[:, i]) <
    #                              (pos[:, i] - self.left_edge[i]))
    #                 pos[idx_left, i] += self.domain_width[i]
    #                 pos[idx_right, i] -= self.domain_width[i]
    #     else:
    #         for i in range(self.ndim):
    #             if self.periodic_right[i] and leafid in self.right_neighbors:
    #                 idx_left = ((pos[:, i] + self.domain_width[i] -
    #                              self.right_edge[i]) <
    #                             (self.left_edge[i] - pos[:, i]))
    #                 pos[idx_left, i] += self.domain_width[i]
    #             if self.periodic_left[i] and leafid in self.left_neighbors:
    #                 idx_right = ((self.left_edge[i] - pos[:, i] +
    #                               self.domain_width[i]) <
    #                              (pos[:, i] - self.right_edge[i]))
    #                 pos[idx_right, i] -= self.domain_width[i]
    #     # Concatenate arrays
    #     self.idx = np.concatenate([self.idx, idx])
    #     # Insert points
    #     self.T.insert(pos)
    #     # Add neighbors
    #     for i in range(self.ndim):
    #         if self.id in ln[i]:
    #             ln[i].remove(self.id)
    #         if self.id in rn[i]:
    #             rn[i].remove(self.id)
    #         self.left_neighbors[i] = ln[i]
    #         self.right_neighbors[i] = rn[i]
    #     if self.limit_mem:
    #         self.save_tess()

    # def consolidate(self, ncells, idx_inf, all_verts, all_cells,
    #                 leaf_start, leaf_stop, split_map, inf_map):
    #     r"""Add local tessellation to global one.

    #     Args:
    #         ncells: Total number of cells currently in the global
    #             tessellation.
    #         split_map: Tuple containing necessary arrays to reconstruct the
    #             map containing information for cells split between leaves.
    #         inf_map: Tuple containing necessary arrays to reconstruct the map
    #             containing information for cells that are infinite.

    #     Returns:
    #         ncells: Total number of cells in the global tessellation after
    #             adding this leaf.
    #         split_map: Tuple containing necessary arrays to reconstruct the
    #             map containing information for cells split between leaves,
    #             updated after adding this leaf.
    #         inf_map: Tuple containing necessary arrays to reconstruct the map
    #             containing information for cells that are infinite, updated
    #             after adding this leaf.

    #     """
    #     ncells, split_map, inf_map = tools.add_leaf(
    #         self.ndim, ncells, idx_inf, all_verts, all_cells,
    #         leaf_start, leaf_stop,
    #         split_map[0], split_map[1], inf_map[0], inf_map[1],
    #         leaf.id, leaf.idx_inf, leaf.verts, leaf.neigh,
    #         leaf.sort_verts, leaf.sort_cells)
    #     return ncells, split_map, inf_map

    def serialize(self, store=False, as_single_array=False):
        r"""Get the serialized tessellation for this leaf.

        Args:
            store (bool, optional): If True, values are stored as attributes
                and not returned. Defaults to False.
            as_single_array (bool, optional): If True, a single array is
                returned that contains all serialized information.

        Returns:
            tuple: Vertices and neighbors for cells in the triangulation.

        """
        if self.limit_mem:
            self.load_tess()
        cells, neigh, idx_inf = self.T.serialize_info2idx(self.norig, self.idx)
        idx_verts, idx_cells = tools.py_arg_sortSerializedTess(cells)
        if store:
            self.idx_inf = idx_inf
            self.verts = cells
            self.neigh = neigh
            self.sort_verts = idx_verts
            self.sort_cells = idx_cells
            out = None
        else:
            ncell_tot = self.T.num_cells
            if as_single_array:
                out = np.concatenate([
                    np.array([cells.shape[0], ncell_tot, idx_inf]),
                    cells.flatten(), neigh.flatten(),
                    idx_verts.flatten(), idx_cells.flatten()]).astype('int64')
            else:
                out = (cells, neigh, idx_inf, idx_verts, idx_cells, ncell_tot)
        if self.limit_mem:
            self.save_tess()
        return out

    def write_tess_to_file(self, fname=None):
        r"""Write out serialized information about the tessellation on this
        leaf.

        Args:
            fname (str, optional): Full path to file where tessellation info
                should be written. Defaults to None. If None, it is set to
                :py:meth:`cgal4py.parallel.ParallelLeaf.tess_output_filename`.

        Returns:
            int: The maximum number of cells that will be contributed by this
                leaf. This is based on the number of cells found to be on this
                leaf in the local tessellation and includes cells that are not
                output to file (e.g. infinite cells).

        """
        if fname is None:
            fname = self.tess_output_filename
        out = self.serialize()
        cells, neigh, idx_inf, idx_verts, idx_cells, ncell_tot = out
        tools.output_leaf(fname, self.id, idx_inf, cells, neigh,
                          idx_verts, idx_cells, self.start_idx, self.stop_idx)
        return ncell_tot

    def voronoi_volumes(self):
        r"""Get the voronoi cell volumes for the original vertices on this
        leaf.

        Returns:
            np.ndarray of float64: Voronoi cell volumes. -1 indicates an
                infinite cell.

        """
        if self.limit_mem:
            self.load_tess()
        out = self.T.voronoi_volumes()[:self.norig]
        if self.limit_mem:
            self.save_tess()
        return out
