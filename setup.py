from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
import distutils.sysconfig
from Cython.Build import cythonize
import numpy
import os, copy
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

include_parallel_delaunay = True

# Check if ReadTheDocs is building extensions
RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')
# RTDFLAG = True

# Stop obnoxious -Wstrict-prototypes warning with c++
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

# Needed for line_profiler - disable for production code
# if not RTDFLAG:
#     from Cython.Compiler.Options import directive_defaults
#     directive_defaults['linetrace'] = True
#     directive_defaults['binding'] = True

# Set generic extension options
ext_options = dict(language="c++",
                   include_dirs=[numpy.get_include()],
                   libraries=[],
                   extra_link_args=[],
                   extra_compile_args=["-std=c++14"],# "-std=gnu++11",
                   # CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
                   define_macros=[('CYTHON_TRACE', '1'),
                                  ("NPY_NO_DEPRECATED_API", None)])
if RTDFLAG:
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')
    ext_options_cgal = copy.deepcopy(ext_options)
else:
    ext_options_cgal = copy.deepcopy(ext_options)
    ext_options_cgal['libraries'] += ['gmp','CGAL']
    ext_options_cgal['extra_link_args'] += ["-lgmp"]


def _delaunay_filename(ftype, dim, periodic=False, bit64=False):
    _delaunay_dir = 'cgal4py/delaunay'
    ver = str(dim)
    perstr = '' ; bitstr = ''
    relpath = True
    if periodic:
        perstr = 'periodic_'
    if bit64:
        bitstr = '_64bit'
    if ftype == 'ext':
        fname = "cgal4py.delaunay.{}delaunay{}{}".format(perstr, ver, bitstr)
        relpath = False
    elif ftype == 'pyx':
        fname = "{}delaunay{}{}.pyx".format(perstr, ver, bitstr)
    elif ftype == 'pxd':
        fname = "{}delaunay{}{}.pxd".format(perstr, ver, bitstr)
    elif ftype == 'cpp':
        fname = "c_{}delaunay{}{}.cpp".format(perstr, ver, bitstr)
    elif ftype == 'hpp':
        fname = "c_{}delaunay{}{}.hpp".format(perstr, ver, bitstr)
    else:
        raise ValueError("Unsupported file type {}.".format(ftype))
    if relpath:
        fname = os.path.join(_delaunay_dir, fname)
    return fname


# Add Delaunay cython extensions
def add_delaunay(ext_modules, ver, periodic=False, bit64=False):
    ver = int(ver)
    ext_name = _delaunay_filename('ext', ver, periodic=periodic, bit64=bit64)
    pyx_file = _delaunay_filename('pyx', ver, periodic=periodic, bit64=bit64)
    cpp_file = _delaunay_filename('cpp', ver, periodic=periodic, bit64=bit64)
    if not os.path.isfile(pyx_file):
        print("Extension {} does not exist and will not be compiled".format(ext_name))
        return
    if not os.path.isfile(cpp_file):
        open(cpp_file,'a').close()
        assert(os.path.isfile(cpp_file))
    ext_modules.append(Extension(ext_name, sources=[pyx_file, cpp_file],
                                 **ext_options_cgal))

# Add extensions
cmdclass = { }
ext_modules = [ ]

# Add delaunay extensions
for ver in [2, 3]:
    add_delaunay(ext_modules, ver)
    add_delaunay(ext_modules, ver, periodic=True)

# Add other packages
ext_modules += [
    Extension("cgal4py.delaunay.tools",
              sources=["cgal4py/delaunay/tools.pyx"],
              **ext_options),
    Extension("cgal4py.domain_decomp.kdtree",
              sources=["cgal4py/domain_decomp/kdtree.pyx",
                       "cgal4py/domain_decomp/c_kdtree.cpp",
                       "cgal4py/c_utils.cpp"],
              **ext_options),
    Extension("cgal4py.utils",
              sources=["cgal4py/utils.pyx","cgal4py/c_utils.cpp"],
              **ext_options)
    ]
if include_parallel_delaunay:
    ext_options_mpicgal = copy.deepcopy(ext_options_cgal)
    import cykdtree
    cykdtree_cpp = os.path.join(
        os.path.dirname(cykdtree.__file__), "c_kdtree.cpp")
    cykdtree_utils_cpp = os.path.join(
        os.path.dirname(cykdtree.__file__), "c_utils.cpp")
    # OpenMPI
    if False:
        mpi_compile_args = os.popen(
            "mpic++ --showme:compile").read().strip().split(' ')
        mpi_link_args = os.popen(
            "mpic++ --showme:link").read().strip().split(' ')
    else:
        mpi_compile_args = os.popen(
            "mpic++ -compile_info").read().strip().split(' ')[1:]
        mpi_link_args = os.popen(
            "mpic++ -link_info").read().strip().split(' ')[1:]
    ext_options_mpicgal['extra_compile_args'] += mpi_compile_args
    ext_options_mpicgal['extra_link_args'] += mpi_link_args
    ext_options_mpicgal['include_dirs'].append(
        os.path.dirname(cykdtree.__file__))
    pyx_file = "cgal4py/delaunay/parallel_delaunay.pyx"
    cpp_file = "cgal4py/delaunay/c_parallel_delaunay.cpp"
    if not os.path.isfile(cpp_file):
        open(cpp_file,'a').close()
    assert(os.path.isfile(cpp_file))
    ext_modules.append(
        Extension("cgal4py.delaunay.parallel_delaunay",
                  sources=[pyx_file, cpp_file,
                           # "cgal4py/delaunay/c_tools.cpp",
                           "cgal4py/delaunay/c_delaunay2.cpp",
                           cykdtree_cpp, cykdtree_utils_cpp],
                  **ext_options_mpicgal))


if use_cython:
    ext_modules = cythonize(ext_modules)

setup(name = 'cgal4py',
      version = '0.1',
      description = 'Python interface for CGAL Triangulations',
      url = 'https://langmm@bitbucket.org/langmm/cgal4py',
      author = 'Meagan Lang',
      author_email = 'langmm.astro@gmail.com',
      license = 'GPL',
      packages = ['cgal4py', 'cgal4py.delaunay'],
      zip_safe = False,
      cmdclass = cmdclass,
      ext_modules = ext_modules)


