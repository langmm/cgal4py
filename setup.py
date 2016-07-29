from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults
import numpy
import os


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
                       libraries=['gmp','cgal'],
                       extra_link_args=["-lgmp"],
                       extra_compile_args=["-std=c++11"],# "-std=gnu++11",
                       # CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
                       define_macros=[('CYTHON_TRACE', '1')])
if RTDFLAG:
    # ext_options['language'] = "c"
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')

if use_cython:
    ext_modules += cythonize(Extension("cgal4py/delaunay2",
                                       sources=["cgal4py/delaunay2.pyx","cgal4py/c_delaunay2.cpp"],
                                       **ext_options))
    ext_modules += cythonize(Extension("cgal4py/delaunay3",
                                       sources=["cgal4py/delaunay3.pyx","cgal4py/c_delaunay3.cpp"],
                                       **ext_options))
    # ext_modules += cythonize(Extension("cgal4py/kdtree",
    #                          sources=["cgal4py/kdtree.pyx","cgal4py/c_kdtree.cpp"],
    #                          language="c++",
    #                          include_dirs=[numpy.get_include()],
    #                          extra_compile_args=["-std=gnu++11"]))
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("cgal4py.delaunay2", ["cgal4py/c_delaunay2.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.delaunay3", ["cgal4py/c_delaunay3.cpp"],
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




# setup(
#     ext_modules=[
#         Extension("mlutils", ["mlutils.c"],
#                   include_dirs=[])#umpy.get_include()]),
#     ],
# )

# # from Cython.Build import cythonize
# setup(
#     ext_modules = cythonize("mlutils.pyx")
# )
