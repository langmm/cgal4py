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

# Versions of Delaunay triangulation that ahve been wrapped
delaunay_ver = ['2','3']

# Check if ReadTheDocs is building extensions
RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')
# RTDFLAG = True

# Stop obnoxious -Wstrict-prototypes warning with c++
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

# Needed for line_profiler - disable for production code
if not RTDFLAG:
    from Cython.Compiler.Options import directive_defaults
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True

# Set generic extension options
ext_options = dict(language="c++",
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=["-std=c++14"],# "-std=gnu++11",
                   # CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
                   define_macros=[('CYTHON_TRACE', '1'),
                                  ("NPY_NO_DEPRECATED_API", None)])
if RTDFLAG:
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')
    ext_options_cgal = copy.deepcopy(ext_options)
else:
    ext_options_cgal = copy.deepcopy(ext_options)
    ext_options_cgal['libraries'] = ['gmp','CGAL']
    ext_options_cgal['extra_link_args'] = ["-lgmp"]

# Add Delaunay cython extensions
from cgal4py.delaunay import _delaunay_filename
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
    if use_cython:
        ext_modules += cythonize(Extension(ext_name,sources=[pyx_file,cpp_file],
                                           **ext_options_cgal))
    else:
        ext_modules.append(Extension(ext_name,[cpp_file],**ext_options_cgal))
                                     # include_dirs=[numpy.get_include()]))

# Add extensions
cmdclass = { }
ext_modules = [ ]

for ver in delaunay_ver:
    add_delaunay(ext_modules, ver)
    add_delaunay(ext_modules, ver, periodic=True)

# Add other packages
if use_cython:
    ext_modules += cythonize(Extension(
        "cgal4py/delaunay/tools",
        sources=["cgal4py/delaunay/tools.pyx"],
        **ext_options))
    ext_modules += cythonize(Extension(
        "cgal4py/domain_decomp/kdtree",
        sources=["cgal4py/domain_decomp/kdtree.pyx",
                 "cgal4py/domain_decomp/c_kdtree.cpp","cgal4py/c_utils.cpp"],
        **ext_options))
    ext_modules += cythonize(Extension(
        "cgal4py/utils",
        sources=["cgal4py/utils.pyx","cgal4py/c_utils.cpp"],
        **ext_options))
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("cgal4py.delaunay.tools",
                  ["cgal4py/delaunay/c_tools.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.domain_decomp.kdtree",
                  ["cgal4py/domain_decomp/c_kdtree.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.utils", ["cgal4py/c_utils.cpp"],
                  include_dirs=[numpy.get_include()]),
    ]

setup(name = 'cgal4py',
      version = '0.1',
      description = 'Python interface for CGAL Triangulations',
      url = 'https://langmm@bitbucket.org/langmm/cgal4py',
      author = 'Meagan Lang',
      author_email = 'langmm.astro@gmail.com',
      license = 'GPL',
      packages = ['cgal4py'],
      zip_safe = False,
      cmdclass = cmdclass,
      ext_modules = ext_modules)


