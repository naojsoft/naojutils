#! /usr/bin/env python
#
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from naoj.version import version
import os

srcdir = os.path.dirname(__file__)

from distutils.command.build_py import build_py

def read(fname):
    buf = open(os.path.join(srcdir, fname), 'rt').read()
    return buf

setup(
    name = "naoj",
    version = version,
    author = "Software Division, Subaru Telescope, National Astronomical Observatory of Japan",
    author_email = "ocs@naoj.org",
    description = ("Misc python modules for working with Subaru Telescope instrument data."),
    long_description = read('README.txt'),
    license = "BSD",
    keywords = "NAOJ, subaru, telescope, instrument, data",
    url = "http://naojsoft.github.com/naojutils",
    packages = ['naoj', 'naoj.cmap', 'naoj.spcam', 'naoj.hsc',
                'naoj.focas', 'naoj.util'],
    package_data = { 'naoj.focas': ['ifu_regions/*.reg'] },
    scripts = ['scripts/focas_ifu_reconstruct_image',
               'scripts/focas_ifu_biassub', 'scripts/focas_ifu_mkbiastemplate',
               'scripts/focas_ifu_mkflat'],
    classifiers = [
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    cmdclass={'build_py': build_py}
)
