#!/usr/bin/env python
"""QuTiP: The Quantum Toolbox in Python

QuTiP is open-source software for simulating the dynamics of closed and open
quantum systems. The QuTiP library depends on the excellent Numpy, Scipy, and
Cython numerical packages. In addition, graphical output is provided by
Matplotlib.  QuTiP aims to provide user-friendly and efficient numerical
simulations of a wide variety of quantum mechanical problems, including those
with Hamiltonians and/or collapse operators with arbitrary time-dependence,
commonly found in a wide range of physics applications. QuTiP is freely
available for use and/or modification on all common platforms. Being free of
any licensing fees, QuTiP is ideal for exploring quantum mechanics in research
as well as in the classroom.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""

import os
import sys

# The following is required to get unit tests up and running.
# If the user doesn't have, then that's OK, we'll just skip unit tests.
try:
    from setuptools import setup, Extension
    EXTRA_KWARGS = {
        'tests_require': ['pytest']
    }
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    EXTRA_KWARGS = {}

try:
    import numpy as np
except ImportError as e:
    raise ImportError("numpy is required at installation") from e

import Cython.Distutils


# all information about QuTiP goes here
MAJOR = 5
MINOR = 0
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%db1' % (MAJOR, MINOR, MICRO)
REQUIRES = ['numpy (>=1.12)', 'scipy (>=1.0)', 'cython (>=0.29.20)']
EXTRAS_REQUIRE = {'graphics': ['matplotlib(>=1.2.1)']}
INSTALL_REQUIRES = ['numpy>=1.12', 'scipy>=1.0', 'cython>=0.29.20']
PACKAGES = ['qutip', 'qutip/ui', 'qutip/qip', 'qutip/qip/device',
            'qutip/qip/operations', 'qutip/qip/compiler',
            'qutip/qip/algorithms', 'qutip/control',
            'qutip/solve', 'qutip/solve/nonmarkov',
            'qutip/_mkl', 'qutip/tests', 'qutip/tests/core',
            'qutip/tests/core/data', 'qutip/tests/solve',
            'qutip/core', 'qutip/core/cy',
            'qutip/core/data/', 'qutip/core/cy/openmp']
PACKAGE_DATA = {
    'qutip': ['configspec.ini'],
    'qutip/tests': ['*.ini'],
    'qutip/core/data': ['*.pxd', '*.pyx'],
    'qutip/core/cy': ['*.pxd', '*.pyx'],
    'qutip/core/cy/src': ['*.hpp', '*.cpp'],
    'qutip/core/cy/openmp': ['*.pxd', '*.pyx'],
    'qutip/core/cy/openmp/src': ['*.hpp', '*.cpp'],
    'qutip/solve': ['*.pxd', '*.pyx'],
    'qutip/solve/nonmarkov': ['*.pxd', '*.pyx'],
    'qutip/tests/qasm_files': ['*.qasm'],
    'qutip/control': ['*.pyx'],
}

INCLUDE_DIRS = [np.get_include()]
NAME = "qutip"
AUTHOR = ("Alexander Pitchford, Paul D. Nation, Robert J. Johansson, "
          "Chris Granade, Arne Grimsmo, Nathan Shammah, Shahnawaz Ahmed, "
          "Neill Lambert, Eric Giguere, Boxi Li, Jake Lishman")
AUTHOR_EMAIL = ("alex.pitchford@gmail.com, nonhermitian@gmail.com, "
                "jrjohansson@gmail.com, cgranade@cgranade.com, "
                "arne.grimsmo@gmail.com, nathan.shammah@gmail.com, "
                "shahnawaz.ahmed95@gmail.com, nwlambert@gmail.com, "
                "eric.giguere@usherbrooke.ca, etamin1201@gmail.com, "
                "jake@binhbar.com")
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "quantum physics dynamics"
URL = "http://qutip.org"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


def git_short_hash():
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except:
        git_str = ""
    else:
        if git_str == '+':  # fixes setuptools PEP issues with versioning
            git_str = ''
    return git_str


FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'+str(MICRO)+git_short_hash()


def write_version_py(filename='qutip/version.py'):
    cnt = """\
# THIS FILE IS GENERATED FROM QUTIP SETUP.PY
short_version = '%(version)s'
version = '%(fullversion)s'
release = %(isrelease)s
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'fullversion':
                FULLVERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()


local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, 'qutip'))  # to retrive _version

# always rewrite _version
if os.path.exists('qutip/version.py'):
    os.remove('qutip/version.py')

write_version_py()

# Cython extensions to be compiled.  The key is the relative package name, the
# value is a list of the Cython modules in that package.
cy_exts = {
    'core.data': [
        'add',
        'adjoint',
        'base',
        'convert',
        'csr',
        'dense',
        'dispatch',
        'expect',
        'inner',
        'kron',
        'matmul',
        'mul',
        'norm',
        'permute',
        'pow',
        'project',
        'properties',
        'ptrace',
        'reshape',
        'tidyup',
        'trace',
    ],
    'core.cy': [
        'coefficient',
        'cqobjevo',
        'inter',
        'interpolate',
        'math',
    ],
    'control': [
        'cy_grape',
    ],
    'solve': [
        '_brtensor',
        '_brtools',
        '_brtools_checks',
        '_mcsolve',
        '_piqs',
        '_steadystate',
        '_stochastic',
    ],
    'solve.nonmarkov': [
        '_heom',
    ],
}

EXT_MODULES = []
MACROS = [
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
]

# Add Cython files from qutip
for package, files in cy_exts.items():
    for file in files:
        _module = 'qutip' + ('.' + package if package else '') + '.' + file
        _file = os.path.join('qutip', *package.split("."), file + '.pyx')
        _sources = [_file, 'qutip/core/data/src/matmul_csr_vector.cpp']
        _ext = Extension(
            _module,
            sources=_sources,
            include_dirs=INCLUDE_DIRS,
            language='c++',
            define_macros=MACROS,
        )
        EXT_MODULES.append(_ext)


# Remove -Wstrict-prototypes from cflags - it's valid for C, but not C++.
# distutils still seems to inject it, though, so we may need to remove it.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")


# TODO: reinstate proper OpenMP handling.
if '--with-openmp' in sys.argv:
    sys.argv.remove('--with-openmp')


def _combine_args(base, extras):
    if base is None:
        base = []
    return base + extras


class BuildExtOverride(Cython.Distutils.build_ext):
    """
    Provide overrides for the default setuptools build_ext command to handle
    platform- and compiler-specific command-line options.  This inherits from
    the Cython version because we want to use their automatic cythonisation and
    parallel building capabilities in `finalize_options()`.
    """
    def build_extensions(self):
        # At this point, setuptools has chosen the compiler, so we don't need
        # to guess based on the environment.
        compiler = self.compiler.compiler_type
        cflags = []
        ldflags = []
        if compiler == 'msvc':
            cflags += ['/Ox']
        else:
            cflags += ['-w', '-O3', '-funroll-loops']
        if 'darwin' in sys.platform:
            # These are needed for compiling on macOS >= 10.14.
            cflags += ['-mmacosx-version-min=10.9']
            ldflags += ['-mmacosx-version-min=10.9']
        for extension in self.extensions:
            extension.extra_compile_args =\
                _combine_args(extension.extra_compile_args, cflags)
            extension.extra_link_args =\
                _combine_args(extension.extra_link_args, ldflags)
        super().build_extensions()


# Setup commands go here
setup(
    name=NAME,
    version=FULLVERSION,
    packages=PACKAGES,
    include_package_data=True,
    include_dirs=INCLUDE_DIRS,
    ext_modules=cythonize(EXT_MODULES),
    cmdclass={'build_ext': build_ext},
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    requires=REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data=PACKAGE_DATA,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    **EXTRA_KWARGS,
)

print("""\
==============================================================================
Installation complete
Please cite QuTiP in your publication.
==============================================================================
For your convenience a bibtex reference can be easily generated using
`qutip.cite()`\
""")
