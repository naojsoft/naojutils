NAOJUTILS ABOUT
---------------
Miscellaneous Python modules for working with NAOJ instrument data.

COPYRIGHT AND LICENSE
---------------------
Copyright (c) 2014-2017  Software Division, Subaru Telescope, 
  National Astronomical Observatory of Japan.  All rights reserved.

naojutils is distributed under an open-source BSD licence.  Please see the
file LICENSE.txt in the top-level directory for details.

BUILDING AND INSTALLATION
-------------------------
naojutils is a standard pyproject.toml (PEP 621) package, so it installs
with pip:

    $ pip install .

For a development install, where edits take effect without reinstalling:

    $ pip install -e .

To install somewhere specific:

    $ pip install --prefix=/some/path .

To build a wheel and an sdist without installing:

    $ python -m build

The Ginga plugins ship with the package -- they are registered through
the "ginga.rv.plugins" entry point, so Ginga finds them once naojutils
is installed, with nothing further to build.
