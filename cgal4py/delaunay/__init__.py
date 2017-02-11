r"""Access to generic triangulation extensions.


..todo::
   * check for eigen3 library before building nD extensions

"""
import numpy as np
import warnings
import importlib
from cgal4py.delaunay import tools
import os
import pyximport
import distutils
from cgal4py.delaunay import delaunay2
from cgal4py.delaunay import delaunay3
from cgal4py.delaunay.delaunay2 import Delaunay2
from cgal4py.delaunay.delaunay3 import Delaunay3
from cgal4py.delaunay.periodic_delaunay2 import is_valid as is_valid_P2
from cgal4py.delaunay.periodic_delaunay3 import is_valid as is_valid_P3
from cgal4py.delaunay import parallel_delaunayD
if is_valid_P2():
    from cgal4py.delaunay import periodic_delaunay2
    from cgal4py.delaunay.periodic_delaunay2 import PeriodicDelaunay2
else:
    warnings.warn("Could not import the 2D periodic triangulation package. " +
                  "Update CGAL to at least version 4.9 to enable this " +
                  "feature.")
if is_valid_P3():
    from cgal4py.delaunay import periodic_delaunay3
    from cgal4py.delaunay.periodic_delaunay3 import PeriodicDelaunay3
else:
    warnings.warn("Could not import the 3D periodic triangulation package. " +
                  "Update CGAL to at least version 3.5 to enable this " +
                  "feature.")

_include_dir = distutils.sysconfig.get_config_var('CONFINCLUDEDIR')
_include_dirs = []
_drive = os.path.abspath(os.sep)
if 'eigen3' in os.listdir(_include_dir):
    _include_dirs.append(os.path.join(_include_dir, 'eigen3'))
elif 'eigen3' in os.listdir(os.path.join(_drive, 'usr', 'include')):
    _include_dirs.append(os.path.join(_drive, 'usr', 'include', 'eigen3'))
else:
    raise Exception("eigen3 is not installed.")
if 'boost' in os.listdir(_include_dir):
    _include_dirs.append(os.path.join(_include_dir, 'boost'))
elif 'boost' in os.listdir(os.path.join(_drive, 'usr', 'include')):
    _include_dirs.append(os.path.join(_drive, 'usr', 'include', 'boost'))
else:
    raise Exception("boost is not installed.")

_delaunay_dir = os.path.dirname(os.path.realpath(__file__))


def _create_pyxdep_file(fname, includes=[], overwrite=False):
    r"""Create a .pyxdep files for an extension.

    Args:
        fname (str): Full path to file that will be created.
        includes (list, optional): List of files that should be included
            by the pyxdep file. Defaults to [].
        overwrite (bool, optional): If True, generated extension files are
            re-generated. Defaults to False.

    Returns:
        bool: If True, a new file was created. If False, there was an
            appropriate existing file.

    """
    assert(fname.endswith('.pyxdep'))
    if os.path.isfile(fname):
        if overwrite:
            os.remove(fname)
        else:
            return False
    with open(fname, 'w') as new_file:
        for incl in includes:
            new_file.write('{}\n'.format(incl))
    return True

def _create_pyxbld_file(fname, overwrite=False, **kwargs):
    r"""Create a .pyxbld files for an extension.

    Args:
        fname (str): Full path to file that will be created.
        overwrite (bool, optional): If True, generated extension files are
            re-generated. Defaults to False.
        \*\*kwargs: Additional kwargs are treated as additional input
            arguments to `distutils.extension.Extension`.

    Returns:
        bool: If True, a new file was created. If False, there was an
            appropriate existing file.

    """
    assert(fname.endswith('.pyxbld'))
    if os.path.isfile(fname):
        if overwrite:
            os.remove(fname)
        else:
            return False
    lines = [
        "def make_ext(modname, pyxfilename):",
        "    from distutils.extension import Extension",
        "    import numpy",
        "    return Extension(name=modname,",
        "                     sources=[pyxfilename] + " + \
            "{},".format(kwargs.get('sources', [])),
        "                     include_dirs=[numpy.get_include()] + " + \
            "{},".format(kwargs.get('include_dirs', [])),
        "                     libraries=['gmp','CGAL'] + " + \
            "{},".format(kwargs.get('libraries', [])),
        "                     language='c++',",
        "                     extra_compile_args=['-std=gnu++11'] + " + \
            "{},".format(kwargs.get('extra_compile_args', [])),
        "                     extra_link_args=['-lgmp'] + " + \
            "{},".format(kwargs.get('extra_link_args', [])),
        "                     define_macros=[('CGAL_EIGEN3_ENABLED','1')] + " + \
            "{})".format(kwargs.get('define_macros', []))]
    with open(fname, 'w') as new_file:
        for line in lines:
            new_file.write('{}\n'.format(line))
    return True

def _create_ext_file(fname0, fname1, replace=[], insert=[], header=[],
                     overwrite=False):
    r"""Create an alternate version of an extension by making the appropriate
    replacements and insertions.

    Args:
        fname0 (str): Base extension file on which the new extension should be
            based.
        fname1 (str): Name of extension file that will be created.
        replace (list): List of 2 element tuples where the first element is a
            string that should be replaced and the second element is the string
            it should be replaced with. Defautls to [].
        insert (list): List of 2 element tuples where the first element is a
            string contained in the line after which the second string should
            be inserted. Defaults to [].
        header (list): List of lines to include at the very beginning of the
            file. Defaults to [].
        overwrite (bool, optional): If True, generated extension files are
            re-generated. Defaults to False.

    Returns:
        bool: If True, a new file was created. If False, there was an
            appropriate existing file.

    Raises:
        ValueError: If `fname0` does not have a supported extension.
        IOError: If `fname0` does not exist.

    """
    created_file = False
    ext = os.path.splitext(fname0)[1]
    if ext in [".pxd", ".pyx"]:
        comment = '#'
    elif ext in [".hpp", ".cpp", ".h", ".c"]:
        comment = '//'
    else:
        raise ValueError("Unsupported extension {}".format(ext))
    gen_file_warn = (comment + " WARNING: This file was automatically " +
                     "generated. Do NOT edit it directly.\n")
    if (not os.path.isfile(fname0)):
        raise IOError("Cannot create {} because original ".format(fname1) +
                      "({}). dosn't exist".format(fname0))
    if ((not os.path.isfile(fname1)) or overwrite or
            (os.path.getmtime(fname1) < os.path.getmtime(fname0))):
        created_file = True
        if os.path.isfile(fname1):
            os.remove(fname1)
        with open(fname1,'w') as new_file:
            with open(fname0,'r') as old_file:
                for line in header:
                    new_file.write(line)
                new_file.write(gen_file_warn)
                for line in old_file:
                    # Make replacements
                    match = False
                    for r0,r1 in replace:
                        if r0 in line:
                            new_file.write(line.replace(r0,r1))
                            match = True
                            break
                    if not match:
                        new_file.write(line)
                    # Insert new lines
                    for i0,i1 in insert:
                        if i0 in line:
                            new_file.write(i1)
    return created_file


def _delaunay_filename(ftype, dim, periodic=False, bit64=False,
                       parallel=False):
    r"""Get a filename for a specific Delaunay extension.

    Args:
        ftype (str): Type of file for which the filename should be returned.
            Options include:
                'ext': The path to the extension.
                'pyx': The .pyx file for the extension.
                'pxd': The .pxd file for the extension.
                'cpp': The .cpp file containing the underlying C++ wrapper.
                'hpp': The .hpp file containing the underlying C++ wrapper.
                'import': The import line that must be added to 64bit version
                    extensions.
                'module': The full module name starting at 'cgal4py'.
                'pyclass': The associated triangulation python class name.
                'cppclass': The associated triangulation C++ class name.
                'pyxdep': The .pyxdep pyximport dependency file.
                'pyxbld': The .pyxbld pyximport build file.
        dim (int): The dimensionality of the requested extension.
        periodic (bool, optional): If True, the names for the periodic
            extension are returned. Defaults to False.
        bit64 (bool, optional): If True, the names for the 64bit extension are
            returned. Defaults to False.
        parallel (bool, optional): If True, the names for the parallel version
            are returned. Defaults to False.

    Returns:
        str: File name replative to cgal4py package directory.

    Raises:
        ValueError: If `ftype` is not one of the values listed above.

    """
    relpath = False
    ver = str(dim)
    perstr = '' ; bitstr = ''
    if periodic:
        perstr = 'periodic_'
    if parallel:
        perstr = 'parallel_'
        if dim in [2, 3]:
            ver = 'D'
    if bit64:
        bitstr = '_64bit'
    if ftype == 'ext':
        relpath = True
        fname = "{}delaunay{}{}".format(perstr, ver, bitstr)
    elif ftype == 'pyx':
        relpath = True
        fname = "{}delaunay{}{}.pyx".format(perstr, ver, bitstr)
    elif ftype == 'pxd':
        relpath = True
        fname = "{}delaunay{}{}.pxd".format(perstr, ver, bitstr)
    elif ftype == 'cpp':
        relpath = True
        fname = "c_{}delaunay{}{}.cpp".format(perstr, ver, bitstr)
    elif ftype == 'hpp':
        relpath = True
        fname = "c_{}delaunay{}{}.hpp".format(perstr, ver, bitstr)
    elif ftype == 'import':
        if ver not in ['2','3','D']:
            fname = '\nfrom cgal4py.delaunay.{}delaunay{} '.format(
                perstr, ver) + \
                    'cimport {}Delaunay_with_info_{},VALID,D\n'.format(
                        perstr.title().rstrip('_'), ver)
        else:
            fname = '\nfrom cgal4py.delaunay.{}delaunay{} '.format(
                perstr, ver) + \
                    'cimport {}Delaunay_with_info_{},VALID\n'.format(
                        perstr.title().rstrip('_'), ver)
    elif ftype == 'module':
        fname = 'cgal4py.delaunay.{}delaunay{}{}'.format(perstr, ver, bitstr)
    elif ftype == 'pyclass':
        fname = '{}Delaunay{}{}'.format(
            perstr.title().rstrip('_'), ver, bitstr)
    elif ftype == 'cppclass':
        fname = '{}Delaunay_with_info_{}{}'.format(
            perstr.title().rstrip('_'), ver, bitstr)
    elif ftype == 'pyxdep':
        fname = _delaunay_filename('pyx', dim, periodic=periodic, bit64=bit64,
                                   parallel=parallel)
        fname += 'dep'
    elif ftype == 'pyxbld':
        fname = _delaunay_filename('pyx', dim, periodic=periodic, bit64=bit64,
                                   parallel=parallel)
        fname += 'bld'
    else:
        raise ValueError("Unsupported file type {}.".format(ftype))
    if relpath:
        fname = os.path.join(_delaunay_dir, fname)
    return fname


def _make_ext(dim, periodic=False, bit64=False, parallel=False,
              overwrite=False):
    r"""Create the necessary files for an arbitrary Delaunay extension.

    Args:
        dim (int): Dimensionality of the triangulations supported by the
            extension.
        periodic (bool, optional): If True, the periodic version of the base
            extension is used. This is invalid for dim > 3. Defaults to False.
        bit64 (bool, optional): If True, the 64bit version of the extension is
            created. Defaults to False.
        parallel (bool, optional): If True, the periodic version of the base
            extension is used. Defaults to False.
        overwrite (bool, optional): If True, generated extension files are
            re-generated. Defaults to False.

    Returns:
        bool: True if extension is generated and will be compiled the next
            time it is imported. False otherwise.

    Raises:
        ValueError: If dim < 2.
        NotImplementedError: If the periodic version of a an extension with
            greater than 3 dimensions is requested.

    """
    generated = False
    is_new = False
    # Create base file from nD case for >3 dimensions
    if dim < 2:
        raise ValueError("Triangulations are not supported in " +
                         "{} dimensions".format(dim))
    if parallel:
        periodic = False
    if dim not in [2, 3]:
        generated = True
        if periodic:
            raise NotImplementedError(
                "Periodic nD triangulations not currently supported.")
        # hpp
        fnameD = _delaunay_filename('hpp', 'D')
        fnameN = _delaunay_filename('hpp', str(dim))
        replace = [('Delaunay_with_info_D',
                    'Delaunay_with_info_{}'.format(dim)),
                   ('const int D = 4; // REPLACE',
                    'const int D = {}; // REPLACE'.format(dim))]
        is_new += _create_ext_file(fnameD, fnameN, replace=replace,
                                   overwrite=overwrite)
        # pxd
        fnameD = _delaunay_filename('pxd', 'D')
        fnameN = _delaunay_filename('pxd', str(dim))
        replace = [('c_delaunayD.hpp', 'c_delaunay{}.hpp'.format(dim)),
                   ('Delaunay_with_info_D',
                    'Delaunay_with_info_{}'.format(dim))]
        is_new += _create_ext_file(fnameD, fnameN, replace=replace,
                                   overwrite=overwrite)
        # pyx
        fnameD = _delaunay_filename('pyx', 'D')
        fnameN = _delaunay_filename('pyx', str(dim))
        replace = [('DelaunayD', 'Delaunay{}'.format(dim)),
                   ('Delaunay_with_info_D',
                    'Delaunay_with_info_{}'.format(dim))]
        is_new += _create_ext_file(fnameD, fnameN, replace=replace,
                                   overwrite=overwrite)
        # Parallel files
        if parallel:
            # hpp
            fnameD = _delaunay_filename('hpp', 'D', parallel=parallel)
            fnameN = _delaunay_filename('hpp', str(dim), parallel=parallel)
            replace = [('c_delaunayD.hpp', 'c_delaunay{}.hpp'.format(dim)),
                       ('Delaunay_with_info_D',
                        'Delaunay_with_info_{}'.format(dim))]
            is_new += _create_ext_file(fnameD, fnameN, replace=replace,
                                       overwrite=overwrite)
            # pxd
            fnameD = _delaunay_filename('pxd', 'D', parallel=parallel)
            fnameN = _delaunay_filename('pxd', str(dim), parallel=parallel)
            replace = [('c_parallel_delaunayD.hpp',
                        'c_parallel_delaunay{}.hpp'.format(dim)),
                       ('ParallelDelaunay_with_info_D',
                        'ParallelDelaunay_with_info_{}'.format(dim))]
            is_new += _create_ext_file(fnameD, fnameN, replace=replace,
                                       overwrite=overwrite)
            # pyx
            fnameD = _delaunay_filename('pyx', 'D', parallel=parallel)
            fnameN = _delaunay_filename('pyx', str(dim), parallel=parallel)
            replace = [('ParallelDelaunayD', 'ParallelDelaunay{}'.format(dim)),
                       ('ParallelDelaunay_with_info_D',
                        'ParallelDelaunay_with_info_{}'.format(dim))]
            is_new += _create_ext_file(fnameD, fnameN, replace=replace,
                                       overwrite=overwrite)
    # Create 64bit version if requested
    if bit64:
        generated = True
        fname32 = _delaunay_filename('pyx', dim, periodic=periodic,
                                     parallel=parallel)
        fname64 = _delaunay_filename('pyx', dim, periodic=periodic,
                                     parallel=parallel, bit64=True)
        class32 = _delaunay_filename('pyclass', dim, parallel=parallel)
        class64 = _delaunay_filename('pyclass', dim, parallel=parallel,
                                     bit64=True)
        import_line = _delaunay_filename('import', dim, periodic=periodic,
                                         parallel=parallel)
        replace = [
            ("ctypedef uint32_t info_t","ctypedef uint64_t info_t"),
            ("cdef object np_info = np.uint32",
             "cdef object np_info = np.uint64"),
            ("ctypedef np.uint32_t np_info_t",
             "ctypedef np.uint64_t np_info_t"),
            (class32, class64)]
        insert = [("ctypedef np.uint32_t np_info_t", import_line)]
        is_new += _create_ext_file(fname32, fname64, replace=replace,
                                   insert=insert, overwrite=overwrite)
    # Create pyxdep & pyxbld for generated files
    if generated:
        extra_compile_args = []
        extra_link_args = []
        include_dirs = _include_dirs#[]
        if bit64:
            includes = [
                _delaunay_filename('pyx', dim, periodic=periodic,
                                   parallel=parallel, bit64=bit64),
                _delaunay_filename('pxd', dim, periodic=periodic,
                                   parallel=parallel),
                _delaunay_filename('hpp', dim, periodic=periodic,
                                   parallel=parallel),
                _delaunay_filename('cpp', dim, periodic=periodic,
                                   parallel=parallel)]
            sources = [
                _delaunay_filename('cpp', dim, periodic=periodic,
                                   parallel=parallel)]
        else:
            includes = [
                _delaunay_filename('pyx', dim, periodic=periodic, bit64=bit64,
                                   parallel=parallel),
                _delaunay_filename('pxd', dim, periodic=periodic, bit64=bit64,
                                   parallel=parallel),
                _delaunay_filename('hpp', dim, periodic=periodic, bit64=bit64,
                                   parallel=parallel),
                _delaunay_filename('cpp', dim, periodic=periodic, bit64=bit64,
                                   parallel=parallel)]
            sources = [
                _delaunay_filename('cpp', dim, periodic=periodic, bit64=bit64,
                                   parallel=parallel)]
        include_dirs.append(os.path.dirname(sources[0]))
        # sources.append(_delaunay_filename('hpp', dim, periodic=periodic,
        #                                   parallel=parallel))
        if parallel:
            includes.append(_delaunay_filename('hpp', dim))
            if False:  # OpenMPI
                extra_compile_args += os.popen(
                    "mpic++ --showme:compile").read().strip().split(' ')
                extra_link_args += os.popen(
                    "mpic++ --showme:link").read().strip().split(' ')
            else:  # MPICH
                extra_compile_args += os.popen(
                    "mpic++ -compile_info").read().strip().split(' ')[1:]
                extra_link_args += os.popen(
                    "mpic++ -link_info").read().strip().split(' ')[1:]
            import cykdtree
            cykdtree_dir = os.path.dirname(cykdtree.__file__)
            include_dirs.append(cykdtree_dir)
            sources += [
                _delaunay_filename('cpp', dim),
                os.path.join(cykdtree_dir, "c_parallel_kdtree.cpp"),
                os.path.join(cykdtree_dir, "c_utils.cpp")]
        for cpp_file in sources:
            if not os.path.isfile(cpp_file):
                open(cpp_file,'a').close()
            assert(os.path.isfile(cpp_file))
        dep = _delaunay_filename('pyxdep', dim, periodic=periodic, bit64=bit64,
                                 parallel=parallel)
        bld = _delaunay_filename('pyxbld', dim, periodic=periodic, bit64=bit64,
                                 parallel=parallel)
        is_new += _create_pyxdep_file(dep, includes=includes,
                                      overwrite=overwrite)
        is_new += _create_pyxbld_file(bld, sources=sources,
                                      extra_compile_args=extra_compile_args,
                                      extra_link_args=extra_link_args,
                                      include_dirs=include_dirs,
                                      overwrite=overwrite)
    if is_new:
        modname = _delaunay_filename('module', dim, periodic=periodic,
                                     bit64=bit64, parallel=parallel)
        warnings.warn("Extension {} is not a built in.\n".format(modname) +
                      "It will be automatically generated and compiled " +
                      "at import using pyximport. Please ignore any " +
                      "compilation warnings from numpy.")
    return generated
        

def _get_Delaunay(ndim, periodic=False, parallel=False, bit64=False,
                  overwrite=False, comm=None):
    r"""Dynamically import module for nD Delaunay triangulation and return
    the associated class.

    Args:
        ndim (int): Dimensionality that module should have.
        periodic (bool, optional): If True, the periodic triangulation class is
            returned. Defaults to False.
        parallel (bool, optional): If True, the parallel triangulation class is
            returned. Defaults to False.
        bit64 (bool, optional): If True, the 64bit triangulation class is
            returned. Defaults to False.
        overwrite (bool, optional): If True, generated extension files are
            re-generated. Defaults to False.
        comm (`mpi4py.Comm`, optional): MPI communicator. If provided,
            import of the requested module will be attempted first on the root
            process, then imported on all processes. This prevents pyximport
            compilation being called multiple times and causing a race
            condition. If not provided, import is done on all processes.

    Returns:
        class: Delaunay triangulation class.

    """
    rank = 0
    if comm is not None:
        rank = comm.Get_rank()
    modname = _delaunay_filename('module', ndim, periodic=periodic,
                                 parallel=parallel, bit64=bit64)
    clsname = _delaunay_filename('pyclass', ndim, periodic=periodic,
                                 parallel=parallel, bit64=bit64)
    # Barrier for non-root processes
    if comm is not None:
        if rank > 0:
            comm.Barrier()
    # Create extension
    gen = _make_ext(ndim, periodic=periodic, parallel=parallel,
                    bit64=bit64, overwrite=overwrite)
    # If generated extension, install pyximport
    if gen:
        importers = pyximport.install(setup_args={"include_dirs":np.get_include()},
                                      reload_support=True)
        # Stop obnoxious -Wstrict-prototypes warning with c++
        cfg_vars = distutils.sysconfig.get_config_vars()
        for key, value in cfg_vars.items():
            if type(value) == str:
                cfg_vars[key] = value.replace("-Wstrict-prototypes", "")
    # Import
    out = getattr(importlib.import_module(modname),clsname)
    # If generated extension, uninstall pyximport
    if gen:
        pyximport.uninstall(*importers)
    # Import on all processes once root is successful
    if comm is not None:
        if rank == 0:
            comm.Barrier()
        # out = getattr(importlib.import_module(modname),clsname)
    return out



def Delaunay(pts, use_double=False, periodic=False,
             left_edge=None, right_edge=None):
    r"""Get a triangulation for a set of points with arbitrary dimensionality.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional
            coordinates.
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        periodic (bool optional): If True, the domain is assumed to be
            periodic at its left and right edges. Defaults to False.
        left_edge (np.ndarray of float64, optional): (m,) lower limits on
            the domain. If None, this is set to np.min(pts, axis=0).
            Defaults to None.
        right_edge (np.ndarray of float64, optional): (m,) upper limits on
            the domain. If None, this is set to np.max(pts, axis=0).
            Defaults to None.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`: 2D or 3D triangulation class.

    Raises:
        ValueError: If pts is not a 2D array.
        NotImplementedError: If pts.shape[1] is not 2 or 3.
        RuntimeError: If there are >=4294967295 points or `use_double == True`
            and the 64bit integer triangulation packages could not be imported.
        ValueError: If `left_edge` is not a 1D array with `pts.shape[1]`
            elements.
        ValueError: If `right_edge` is not a 1D array with `pts.shape[1]`
            elements.
        NotImplementedError: If a periodic package could not be imported due
            to an outdated version of CGAL.

    """
    if (pts.ndim != 2):
        raise ValueError("pts must be a 2D array of coordinates")
    npts = pts.shape[0]
    ndim = pts.shape[1]
    # Check if 64bit integers need/can be used
    if npts >= np.iinfo('uint32').max or use_double:
        use_double = True
    # Create arguments
    args = []
    if periodic:
        if left_edge is None:
            left_edge = np.min(pts, axis=0)
        else:
            if (left_edge.ndim != 1) or (len(left_edge) != ndim):
                raise ValueError("left_edge must be a 1D array with " +
                                 "{} elements.".format(ndim))
        if right_edge is None:
            right_edge = np.max(pts, axis=0)
        else:
            if (right_edge.ndim != 1) or (len(right_edge) != ndim):
                raise ValueError("right_edge must be a 1D array with " +
                                 "{} elements.".format(ndim))
        args = [left_edge, right_edge]
    # Initialize correct tessellation
    DelaunayClass = _get_Delaunay(ndim, periodic=periodic,
                                  bit64=use_double)
    T = DelaunayClass(*args)
    # Insert points into tessellation
    if npts > 0:
        T.insert(pts)
    return T


def VoronoiVolumes(pts, *args, **kwargs):
    r"""Get the volumes of the voronoi cells associated with a set of points.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional
            coordinates.
        \*args: Additional arguments are passed to
            :func:`cgal4py.delaunay.Delaunay`.
        \*\*kwargs: Additional keyword arguments are passed to 
            :func:`cgal4py.delaunay.Delaunay`.

    Returns:
        np.ndarray of float64: Volumes of voronoi cells. Negative values
            indicate infinite cells.

    """
    T = Delaunay(pts, *args, **kwargs)
    return T.voronoi_volumes()


__all__ = ["tools", "Delaunay", "VoronoiVolumes",
           "delaunay2", "delaunay3", "Delaunay2", "Delaunay3"]
if is_valid_P2():
    __all__ += ["periodic_delaunay2", "PeriodicDelaunay2"]
if is_valid_P3():
    __all__ += ["periodic_delaunay3", "PeriodicDelaunay3"]
