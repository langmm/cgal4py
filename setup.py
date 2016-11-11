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

eigen3_exists = True

# Versions of Delaunay triangulation that ahve been wrapped
delaunay_ver = ['2','3',4]
# if eigen3_exists:
#     delaunay_ver.append('D')

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
                   extra_compile_args=["-std=c++11"],# "-std=gnu++11",
                   # CYTHON_TRACE required for coverage and line_profiler.  Remove for release.
                   define_macros=[('CYTHON_TRACE', '1'),
                                  ("NPY_NO_DEPRECATED_API", None)])
if eigen3_exists:
    ext_options['define_macros'].append(("CGAL_EIGEN3_ENABLED",'1'))
    # ("EIGEN3_INC_DIR","/usr/include/eigen3/Eigen/")])

if RTDFLAG:
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')
    ext_options_cgal = copy.deepcopy(ext_options)
else:
    ext_options_cgal = copy.deepcopy(ext_options)
    ext_options_cgal['libraries'] = ['gmp','CGAL'] #,'eigen3']
    ext_options_cgal['extra_link_args'] = ["-lgmp"]

def delaunay_filename(ftype, ver, periodic=False, bit64=False):
    perstr = '' ; bitstr = ''
    if periodic:
        perstr = 'periodic_'
    if bit64:
        bitstr = '_64bit'
    if ftype == 'ext':
        fname = "cgal4py/delaunay/{}delaunay{}{}".format(perstr,ver,bitstr)
    elif ftype == 'pyx':
        fname = "cgal4py/delaunay/{}delaunay{}{}.pyx".format(perstr,ver,bitstr)
    elif ftype == 'pxd':
        fname = "cgal4py/delaunay/{}delaunay{}{}.pxd".format(perstr,ver,bitstr)
    elif ftype == 'cpp':
        fname = "cgal4py/delaunay/c_{}delaunay{}{}.cpp".format(perstr,ver,
                                                               bitstr)
    elif ftype == 'hpp':
        fname = "cgal4py/delaunay/c_{}delaunay{}{}.hpp".format(perstr,ver,
                                                               bitstr)
    elif ftype == 'import':
        fname = '\nfrom cgal4py.delaunay.{}delaunay{} '.format(perstr,ver) + \
                'cimport {}Delaunay_with_info_{},VALID\n'.format(
                    perstr.title().rstrip('_'),ver)
    else:
        raise ValueError("Unsupported file type {}.".format(ftype))
    return fname

def make_alt_ext(fname0, fname1, replace=[], insert=[], comment='#'):
    gen_file_warn = (comment + " WARNING: This file was automatically " +
                     "generated. Do NOT edit it directly.\n")
    if (not os.path.isfile(fname0)):
        print("Cannot create {} because original ".format(fname1) +
              "({}). dosn't exist".format(fname0))
        return
    if (not os.path.isfile(fname1)) or (os.path.getmtime(fname1) < os.path.getmtime(fname0)):
        print("Creating alterate version of {}...".format(fname0))
        if os.path.isfile(fname1):
            os.remove(fname1)
        with open(fname1,'w') as new_file:
            with open(fname0,'r') as old_file:
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

def make_nD(dim):
    # hpp
    fnameD = delaunay_filename('hpp', 'D')
    fnameN = delaunay_filename('hpp', str(dim))
    if os.path.isfile(fnameN): os.remove(fnameN)
    replace = [['Delaunay_with_info_D', 'Delaunay_with_info_{}'.format(dim)],
               ['int D = 4; // REPLACE', 'int D = {}; // REPLACE'.format(dim)]]
    make_alt_ext(fnameD, fnameN, replace=replace)
    # pxd
    fnameD = delaunay_filename('pxd', 'D')
    fnameN = delaunay_filename('pxd', str(dim))
    if os.path.isfile(fnameN): os.remove(fnameN)
    replace = [['c_delaunayD.hpp', 'c_delaunay{}.hpp'.format(dim)],
               ['Delaunay_with_info_D', 'Delaunay_with_info_{}'.format(dim)]]
    make_alt_ext(fnameD, fnameN, replace=replace)
    # pyx
    fnameD = delaunay_filename('pyx', 'D')
    fnameN = delaunay_filename('pyx', str(dim))
    if os.path.isfile(fnameN): os.remove(fnameN)
    replace = [['DelaunayD', 'Delaunay{}'.format(dim)],
               ['Delaunay_with_info_D', 'Delaunay_with_info_{}'.format(dim)]]
    make_alt_ext(fnameD, fnameN, replace=replace)

def make_64bit(ver,periodic=False):
    fname32 = delaunay_filename('pyx', ver, periodic=periodic)
    fname64 = delaunay_filename('pyx', ver, periodic=periodic, bit64=True)
    import_line = delaunay_filename('import', ver, periodic=periodic)
    replace = [
        ["ctypedef uint32_t info_t","ctypedef uint64_t info_t"],
        ["cdef object np_info = np.uint32","cdef object np_info = np.uint64"],
        ["ctypedef np.uint32_t np_info_t","ctypedef np.uint64_t np_info_t"]]
    insert = [["ctypedef np.uint32_t np_info_t", import_line]]
    make_alt_ext(fname32, fname64, replace=replace, insert=insert)
    # if (not os.path.isfile(fname32)):
    #     print("Cannot create 64bit version of {} ".format(fname32) +
    #           "as it dosn't exist.".format(fname32))
    #     return
    # if (not os.path.isfile(fname64)) or (os.path.getmtime(fname64) < os.path.getmtime(fname32)):
    #     print("Creating 64bit version of {}...".format(fname32))
    #     if os.path.isfile(fname64):
    #         os.remove(fname64)
    #     with open(fname64,'w') as new_file:
    #         with open(fname32,'r') as old_file:
    #             for line in old_file:
    #                 if replace[0][0] in line:
    #                     new_file.write(line.replace(replace[0][0],replace[0][1]))
    #                 elif replace[1][0] in line:
    #                     new_file.write(line.replace(replace[1][0],replace[1][1]))
    #                 elif replace[2][0] in line:
    #                     new_file.write(line.replace(replace[2][0],replace[2][1]))
    #                     new_file.write(import_line)
    #                 else:
    #                     new_file.write(line)

# Add Delaunay cython extensions
def add_delaunay(ext_modules, ver, periodic=False, bit64=False):
    ver = int(ver)
    if bit64:
        make_64bit(ver, periodic=periodic)
    if ver > 3:
        make_nD(ver)
    ext_name = delaunay_filename('ext', ver, periodic=periodic, bit64=bit64)
    pyx_file = delaunay_filename('pyx', ver, periodic=periodic, bit64=bit64)
    cpp_file = delaunay_filename('cpp', ver, periodic=periodic, bit64=bit64)
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
    add_delaunay(ext_modules, ver, bit64=True)
    add_delaunay(ext_modules, ver, bit64=True, periodic=True)

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


