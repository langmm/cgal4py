from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults
import numpy
import os

def make_64bit():
    # Create 64bit version from 32bit version (because cython fusedtypes...)
    fnames = [['cgal4py/delaunay/delaunay2.pyx','cgal4py/delaunay/delaunay2_64bit.pyx',
                   "\nfrom cgal4py.delaunay.delaunay2 cimport Delaunay_with_info_2\n"],
              ['cgal4py/delaunay/delaunay3.pyx','cgal4py/delaunay/delaunay3_64bit.pyx',
                   "\nfrom cgal4py.delaunay.delaunay3 cimport Delaunay_with_info_3\n"]]
    replace = [["ctypedef uint32_t info_t","ctypedef uint64_t info_t"],
               ["cdef object np_info = np.uint32","cdef object np_info = np.uint64"],
               ["ctypedef np.uint32_t np_info_t","ctypedef np.uint64_t np_info_t"]]
    for fname32, fname64, import_line in fnames:
        if os.path.getmtime(fname64) < os.path.getmtime(fname32):
            print("Creating 64bit version of {}...".format(fname32))
            with open(fname64,'w') as new_file:
                with open(fname32,'r') as old_file:
                    for line in old_file:
                        if replace[0][0] in line:
                            new_file.write(line.replace(replace[0][0],replace[0][1]))
                        elif replace[1][0] in line:
                            new_file.write(line.replace(replace[1][0],replace[1][1]))
                        elif replace[2][0] in line:
                            new_file.write(line.replace(replace[2][0],replace[2][1]))
                            new_file.write(import_line)
                        else:
                            new_file.write(line)


make_64bit()


RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')
# RTDFLAG = True

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True


# Needed for line_profiler - disable for production code
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

cmdclass = { }
ext_modules = [ ]

ext_options = dict(language="c++",
                       include_dirs=[numpy.get_include()],
                       libraries=['gmp','CGAL'],
                       extra_link_args=["-lgmp"],
                       extra_compile_args=["-std=c++11"],# "-std=gnu++11",
                       # CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
                       define_macros=[('CYTHON_TRACE', '1')])
if RTDFLAG:
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')

if use_cython:
    ext_modules += cythonize(Extension("cgal4py/delaunay/delaunay2",
                                       sources=["cgal4py/delaunay/delaunay2.pyx","cgal4py/delaunay/c_delaunay2.cpp"],
                                       **ext_options))
    ext_modules += cythonize(Extension("cgal4py/delaunay/delaunay3",
                                       sources=["cgal4py/delaunay/delaunay3.pyx","cgal4py/delaunay/c_delaunay3.cpp"],
                                       **ext_options))
    ext_modules += cythonize(Extension("cgal4py/delaunay/delaunay2_64bit",
                                       sources=["cgal4py/delaunay/delaunay2_64bit.pyx","cgal4py/delaunay/c_delaunay2.cpp"],
                                       **ext_options))
    ext_modules += cythonize(Extension("cgal4py/delaunay/delaunay3_64bit",
                                       sources=["cgal4py/delaunay/delaunay3_64bit.pyx","cgal4py/delaunay/c_delaunay3.cpp"],
                                       **ext_options))
    ext_modules += cythonize(Extension("cgal4py/domain_decomp/kdtree",
                                       sources=["cgal4py/domain_decomp/kdtree.pyx","cgal4py/domain_decomp/c_kdtree.cpp","cgal4py/c_utils.cpp"],
                                       language="c++",
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args=["-std=gnu++11"]))
    ext_modules += cythonize(Extension("cgal4py/utils",
                                       sources=["cgal4py/utils.pyx","cgal4py/c_utils.cpp"],
                                       language="c++",
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args=["-std=gnu++11"]))
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("cgal4py.delaunay.delaunay2", ["cgal4py/delaunay/c_delaunay2.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.delaunay.delaunay3", ["cgal4py/delaunay/c_delaunay3.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.delaunay.delaunay2_64bit", ["cgal4py/delaunay/c_delaunay2.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.delaunay.delaunay3_64bit", ["cgal4py/delaunay/c_delaunay3.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.domain_decomp.kdtree", ["cgal4py/domain_decomp/c_kdtree.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.utils", ["cgal4py/c_utils.cpp"],
                  include_dirs=[numpy.get_include()]),
    ]

setup(name='cgal4py',
      version='0.1',
      description='Python interface for CGAL Triangulations',
      url='https://langmm@bitbucket.org/langmm/cgal4py',
      author='Meagan Lang',
      author_email='langmm.astro@gmail.com',
      license='GPL',
      packages=['cgal4py'],
      zip_safe=False,
      cmdclass = cmdclass,
      ext_modules = ext_modules)


