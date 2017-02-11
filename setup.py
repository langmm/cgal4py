from setuptools import setup
import distutils
from distutils.core import setup
from distutils.extension import Extension
import distutils.sysconfig
import numpy
import os
import copy
import sys
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

release = True

# Check if ReadTheDocs is building extensions
RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')
# RTDFLAG = True

# Stop obnoxious -Wstrict-prototypes warning with c++
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

# Find eigen3 and boost libraries
include_dir = cfg_vars['CONFINCLUDEDIR']
include_dirs = [numpy.get_include(), include_dir]
include_files = os.listdir(include_dir)
if 'eigen3' in include_files:
    include_dirs.append(os.path.join(include_dir, 'eigen3'))
else:
    if os.path.isdir('/usr/include/eigen3'):
        include_dirs.append('/usr/include/eigen3')
    else:
        raise Exception("Install eigen3")
if 'boost' in include_files:
    include_dirs.append(os.path.join(include_dir, 'boost'))
else:
    if os.path.isdir('/usr/include/boost'):
        include_dirs.append('/usr/include/boost')
    else:
        raise Exception("Install boost")

# Needed for line_profiler - disable for production code
if not RTDFLAG and not release and use_cython:
    try:
        from Cython.Compiler.Options import directive_defaults
    except ImportError:
        # Update to cython  
        from Cython.Compiler.Options import get_directive_defaults
        directive_defaults = get_directive_defaults()
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True

# Set generic extension options
ext_options = dict(language="c++",
                   include_dirs=include_dirs,#[numpy.get_include()],
                   libraries=[],
                   extra_link_args=[],
                   extra_compile_args=["-std=gnu++11"],
                   define_macros=[("NPY_NO_DEPRECATED_API", None)])
# CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
if not release:
    ext_options['define_macros'].append(
        ('CYTHON_TRACE', '1'))

cykdtree_cpp = None
cykdtree_parallel_cpp = None
cykdtree_parallel_hpp = None
cykdtree_utils_cpp = None
if RTDFLAG:
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')
    ext_options_cgal = copy.deepcopy(ext_options)
    ext_options_mpicgal = copy.deepcopy(ext_options_cgal)
    compile_parallel = False
else:
    ext_options_cgal = copy.deepcopy(ext_options)
    ext_options_cgal['libraries'] += ['gmp','CGAL']
    ext_options_cgal['extra_link_args'] += ["-lgmp"]
    # Check that there is a version of MPI available
    ext_options_mpicgal = copy.deepcopy(ext_options_cgal)
    compile_parallel = True
    try:
        import mpi4py
        import cykdtree
        cykdtree_cpp = os.path.join(
            os.path.dirname(cykdtree.__file__), "c_kdtree.cpp")
        cykdtree_parallel_cpp = os.path.join(
            os.path.dirname(cykdtree.__file__), "c_parallel_kdtree.cpp")
        cykdtree_parallel_hpp = os.path.join(
            os.path.dirname(cykdtree.__file__), "c_parallel_kdtree.hpp")
        cykdtree_utils_cpp = os.path.join(
            os.path.dirname(cykdtree.__file__), "c_utils.cpp")
        # Attempt to call MPICH first, then OpenMPI
        try: 
            mpi_compile_args = os.popen(
                "mpic++ -compile_info").read().strip().split(' ')[1:]
            mpi_link_args = os.popen(
                "mpic++ -link_info").read().strip().split(' ')[1:]
            if len(mpi_compile_args[0]) == 0:
                raise Exception
        except:
            mpi_compile_args = os.popen(
                "mpic++ --showme:compile").read().strip().split(' ')
            mpi_link_args = os.popen(
                "mpic++ --showme:link").read().strip().split(' ')
            if len(mpi_compile_args[0]) == 0:
                raise Exception
        ext_options_mpicgal['extra_compile_args'] += mpi_compile_args
        ext_options_mpicgal['extra_link_args'] += mpi_link_args
        ext_options_mpicgal['include_dirs'].append(
            os.path.dirname(cykdtree.__file__))
    except:
        compile_parallel = False


def _delaunay_filename(ftype, dim, periodic=False, parallel=False):
    _delaunay_dir = os.path.join('cgal4py', 'delaunay')
    ver = str(dim)
    perstr = ''
    relpath = True
    if periodic:
        perstr = 'periodic_'
    if parallel:
        perstr = 'parallel_'
        if dim in [2, 3]:
            ver = 'D'
    if ftype == 'ext':
        fname = "cgal4py.delaunay.{}delaunay{}".format(perstr, ver)
        relpath = False
    elif ftype == 'pyx':
        fname = "{}delaunay{}.pyx".format(perstr, ver)
    elif ftype == 'pxd':
        fname = "{}delaunay{}.pxd".format(perstr, ver)
    elif ftype == 'cpp':
        fname = "c_{}delaunay{}.cpp".format(perstr, ver)
    elif ftype == 'hpp':
        fname = "c_{}delaunay{}.hpp".format(perstr, ver)
    else:
        raise ValueError("Unsupported file type {}.".format(ftype))
    if relpath:
        fname = os.path.join(_delaunay_dir, fname)
    return fname


# Add Delaunay cython extensions
def add_delaunay(ext_modules, src_include, ver, periodic=False, parallel=False,
                 dont_compile=False):
    ext_name = _delaunay_filename('ext', ver, periodic=periodic,
                                  parallel=parallel)
    pyx_file = _delaunay_filename('pyx', ver, periodic=periodic,
                                  parallel=parallel)
    pxd_file = _delaunay_filename('pxd', ver, periodic=periodic,
                                  parallel=parallel)
    cpp_file = _delaunay_filename('cpp', ver, periodic=periodic,
                                  parallel=parallel)
    hpp_file = _delaunay_filename('hpp', ver, periodic=periodic,
                                  parallel=parallel)
    if not os.path.isfile(pyx_file):
        print("Extension {} ".format(ext_name) +
              "does not exist and will not be compiled")
        return
    if not os.path.isfile(cpp_file):
        open(cpp_file,'a').close()
        assert(os.path.isfile(cpp_file))
    if not dont_compile:
        if parallel:
            ext_modules.append(
                Extension(ext_name, sources=[pyx_file, cpp_file,
                                             # cykdtree_cpp,
                                             cykdtree_parallel_cpp,
                                             cykdtree_utils_cpp],
                          **ext_options_mpicgal))
            src_include += [cykdtree_parallel_hpp]
        else:
            ext_modules.append(
                Extension(ext_name, sources=[pyx_file, cpp_file],
                          **ext_options_cgal))
    src_include += [pyx_file, pxd_file, hpp_file]

# Add extensions
cmdclass = { }
ext_modules = [ ]
src_include = [ ]

# Add delaunay extensions
for ver in [2, 3]:
    add_delaunay(ext_modules, src_include, ver)
    add_delaunay(ext_modules, src_include, ver, periodic=True)
add_delaunay(ext_modules, src_include, 'D', parallel=True,
             dont_compile=(not compile_parallel))

# Add other packages
ext_modules += [
    Extension("cgal4py.delaunay.tools",
              sources=["cgal4py/delaunay/tools.pyx"],
              **ext_options),
    ]
src_include += [
    "cgal4py/delaunay/tools.pyx",
    "cgal4py/delaunay/tools.pxd",
    "cgal4py/delaunay/c_tools.hpp"]


if use_cython:
    ext_modules = cythonize(ext_modules)

with open('README.rst') as file:
    long_description = file.read()

setup(name = 'cgal4py',
      packages = ['cgal4py', 'cgal4py.delaunay',
                  'cgal4py.domain_decomp', 'cgal4py.tests'],
      # package_dir = {'cgal4py':'cgal4py'}, # maybe comment this out
      package_data = {'cgal4py': ['README.md', 'README.rst'],
                      'cgal4py.delaunay': src_include},
      version = '0.1.8',
      description = 'Python interface for CGAL Triangulations',
      long_description = long_description,
      author = 'Meagan Lang',
      author_email = 'langmm.astro@gmail.com',
      url = 'https://langmm@bitbucket.org/langmm/cgal4py',
      keywords = ['delaunay', 'voronoi', 'cgal',
                  'triangulation', 'tessellation'],
      classifiers = ["Programming Language :: Python",
                     "Programming Language :: C++",
                     "Operating System :: OS Independent",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: BSD License",
                     "Natural Language :: English",
                     "Topic :: Scientific/Engineering",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Scientific/Engineering :: Physics",
                     "Development Status :: 3 - Alpha"],
      license = 'BSD',
      zip_safe = False,
      cmdclass = cmdclass,
      ext_modules = ext_modules,
      data_files = [(os.path.join('cgal4py', 'delaunay'), src_include)])


