r"""Routines for tracking the scaling of the triangulation routines."""
import numpy as np
import time
import os
import cProfile
import pstats
from cgal4py import parallel, delaunay
from test_cgal4py import run_test
import matplotlib.pyplot as plt
np.random.seed(10)


def stats_run(func, npart, nproc, ndim, periodic=False, use_mpi=False,
              use_python=False, use_buffer=False, overwrite=False,
              display=False, suppress_final_output=False):
    r"""Get timing stats using :package:`cProfile`.

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
        use_python (bool, optional): If True and `use_mpi == True`, then MPI
            communications are done in python using `mpi4py`. Defaults to
            False.
        use_buffer (bool, optional): If True and `use_mpi == True`, then buffer
            versions of MPI communications will be used rather than indirect
            communications of python objects via pickling. Defaults to False.
        suppress_final_output (bool, optional): If True, the final output 
            from spawned MPI processes is suppressed. This is mainly for
            timing purposes. Defaults to False.
        overwrite (bool, optional): If True, the existing file for this
            set of input parameters if overwritten. Defaults to False.
        display (bool, optional): If True, display the profile results.
            Defaults to False.

    """
    perstr = ""
    if periodic:
        perstr = "_periodic"
    mpistr = ""
    bufstr = ""
    pycstr = ""
    outstr = ""
    if use_mpi:
        mpistr = "_mpi"
        if suppress_final_output:
            outstr = "_noout"
        if use_python:
            pycstr = "_python"
            if use_buffer:
                bufstr = "_buffer"
    fname_stat = 'stat_{}_{}part_{}proc_{}dim{}{}{}{}{}.txt'.format(
        func, npart, nproc, ndim, perstr, mpistr, pycstr, bufstr, outstr)
    if overwrite or not os.path.isfile(fname_stat):
        cProfile.run(
            "from cgal4py.tests.test_cgal4py import run_test; "+
            "run_test({}, {}, nproc={}, func_name='{}',".format(
                npart, ndim, nproc, func) +
            "periodic={},use_mpi={},use_python={},use_buffer={},".format(
                periodic, use_mpi, use_python, use_buffer) +
            "suppress_final_output={})".format(suppress_final_output),
            fname_stat)
    if display:
        p = pstats.Stats(fname_stat)
        p.sort_stats('time').print_stats(10)
        return p
    return fname_stat


def time_run(func, npart, nproc, ndim, nrep=1, periodic=False, use_mpi=False,
             use_python=False, use_buffer=False, suppress_final_output=False):
    r"""Get runing times using :package:`time`.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int): Number of particles.
        nproc (int): Number of processors.
        ndim (int): Number of dimensions.
        nrep (int, optional): Number of times the run should be performed to
            get an average. Defaults to 1.
        periodic (bool, optional): If True, the domain is assumed to be
            periodic. Defaults to False.
        use_mpi (bool, optional): If True, the MPI parallelized version is used
            instead of the version using the multiprocessing package. Defaults
            to False.
        use_python (bool, optional): If True and `use_mpi == True`, then MPI
            communications are done in python using `mpi4py`. Defaults to
            False.
        use_buffer (bool, optional): If True and `use_mpi == True`, then buffer
            versions of MPI communications will be used rather than indirect
            communications of python objects via pickling. Defaults to False.
        suppress_final_output (bool, optional): If True, the final output 
            from spawned MPI processes is suppressed. This is mainly for
            timing purposes. Defaults to False.

    """
    times = np.empty(nrep, 'float')
    for i in range(nrep):
        t1 = time.time()
        run_test(npart, ndim, nproc=nproc, func_name=func,
                 periodic=periodic, use_mpi=use_mpi,
                 use_python=use_python, use_buffer=use_buffer,
                 suppress_final_output=suppress_final_output)
        t2 = time.time()
        times[i] = t2 - t1
    return np.mean(times), np.std(times)


def strong_scaling(func, npart=1e6, nrep=1, periodic=False, use_mpi=False,
                   use_python=False, use_buffer=False, overwrite=True,
                   suppress_final_output=False):
    r"""Plot the scaling with number of processors for a particular function.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int, optional): Number of particles. Defaults to 1e6.
        nrep (int, optional): Number of times the run should be performed to
            get an average. Defaults to 1.
        periodic (bool, optional): If True, the domain is assumed to be
            periodic. Defaults to False.
        use_mpi (bool, optional): If True, the MPI parallelized version is used
            instead of the version using the multiprocessing package. Defaults
            to False.
        use_python (bool, optional): If True and `use_mpi == True`, then MPI
            communications are done in python using `mpi4py`. Defaults to
            False.
        use_buffer (bool, optional): If True and `use_mpi == True`, then buffer
            versions of MPI communications will be used rather than indirect
            communications of python objects via pickling. Defaults to False.
        suppress_final_output (bool, optional): If True, the final output 
            from spawned MPI processes is suppressed. This is mainly for
            timing purposes. Defaults to False.
        overwrite (bool, optional): If True, the existing file for this
            set of input parameters if overwritten. Defaults to False.

    """
    npart = int(npart)
    perstr = ""
    if periodic:
        perstr = "_periodic"
    mpistr = ""
    pycstr = ""
    bufstr = ""
    outstr = ""
    if use_mpi:
        mpistr = "_mpi"
        if suppress_final_output:
            outstr = "_noout"
        if use_python:
            pycstr = "_python"
            if use_buffer:
                bufstr = "_buffer"
    fname_plot = 'plot_strong_scaling_{}_nproc_{}part{}{}{}{}{}.png'.format(
        func, npart, perstr, mpistr, pycstr, bufstr, outstr)
    nproc_list = [1, 2, 4, 8, 16]
    ndim_list = [2, 3, 4]
    clr_list = ['b', 'r', 'g', 'm']
    times = np.empty((len(nproc_list), len(ndim_list), 2), 'float')
    for j, nproc in enumerate(nproc_list):
        for i, ndim in enumerate(ndim_list):
            times[j, i, 0], times[j, i, 1] = time_run(
                func, npart, nproc, ndim, nrep=nrep,
                periodic=periodic, use_mpi=use_mpi,
                use_python=use_python, use_buffer=use_buffer,
                suppress_final_output=suppress_final_output)
            print("Finished {}D on {}.".format(ndim, nproc))
    fig, axs = plt.subplots(1, 1)
    for i in range(len(ndim_list)):
        ndim = ndim_list[i]
        clr = clr_list[i]
        axs.errorbar(nproc_list, times[:, i, 0], yerr=times[:, i, 1],
                     fmt=clr, label='ndim = {}'.format(ndim))
    axs.set_xlabel("# of Processors")
    axs.set_ylabel("Time (s)")
    axs.legend()
    fig.savefig(fname_plot)
    print('    '+fname_plot)


def weak_scaling(func, npart=1e4, nrep=1, periodic=False, use_mpi=False,
                 use_python=False, use_buffer=False, overwrite=True,
                 suppress_final_output=False):
    r"""Plot the scaling with number of processors with a constant number of
    particles per processor for a particular function.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int, optional): Number of particles per processor. Defaults to
            1e4.
        nrep (int, optional): Number of times the run should be performed to
            get an average. Defaults to 1.
        periodic (bool, optional): If True, the domain is assumed to be
            periodic. Defaults to False.
        use_mpi (bool, optional): If True, the MPI parallelized version is used
            instead of the version using the multiprocessing package. Defaults
            to False.
        use_python (bool, optional): If True and `use_mpi == True`, then MPI
            communications are done in python using `mpi4py`. Defaults to
            False.
        use_buffer (bool, optional): If True and `use_mpi == True`, then buffer
            versions of MPI communications will be used rather than indirect
            communications of python objects via pickling. Defaults to False.
        suppress_final_output (bool, optional): If True, the final output 
            from spawned MPI processes is suppressed. This is mainly for
            timing purposes. Defaults to False.
        overwrite (bool, optional): If True, the existing file for this
            set of input parameters if overwritten. Defaults to False.

    """
    npart = int(npart)
    perstr = ""
    if periodic:
        perstr = "_periodic"
    mpistr = ""
    pycstr = ""
    bufstr = ""
    outstr = ""
    if use_mpi:
        mpistr = "_mpi"
        if suppress_final_output:
            outstr = "_noout"
        if use_python:
            pycstr = "_python"
            if use_buffer:
                bufstr = "_buffer"
    fname_plot = 'plot_weak_scaling_{}_nproc_{}part{}{}{}{}{}.png'.format(
        func, npart, perstr, mpistr, pycstr, bufstr, outstr)
    nproc_list = [1, 2, 4, 8, 16]
    ndim_list = [2, 3]
    clr_list = ['b', 'r', 'g', 'm']
    times = np.empty((len(nproc_list), len(ndim_list), 2), 'float')
    for j, nproc in enumerate(nproc_list):
        for i, ndim in enumerate(ndim_list):
            times[j, i, 0], times[j, i, 1] = time_run(
                func, npart*nproc, nproc, ndim, nrep=nrep,
                periodic=periodic, use_mpi=use_mpi,
                use_python=use_python, use_buffer=use_buffer,
                suppress_final_output=suppress_final_output)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(ndim_list)):
        ndim = ndim_list[i]
        clr = clr_list[i]
        axs.errorbar(nproc_list, times[:, i, 0], yerr=times[:, i, 1],
                     fmt=clr, label='ndim = {}'.format(ndim))
    axs.set_xlabel("# of Processors")
    axs.set_ylabel("Time (s)")
    axs.legend()
    fig.savefig(fname_plot)
    print('    '+fname_plot)
