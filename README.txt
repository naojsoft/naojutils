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
naojutils uses a standard distutils based install, e.g.

    $ python setup.py build

or

    $ python setup.py install

If you want to install to a specific area, do

    $ python setup.py install --prefix=/some/path

The files will then end up under /some/path

NOTE: If you want to install the optional Ginga plugins, you must also
repeat the "setup.py" operation in the "ginga_plugins" directory, e.g.:

    $ cd ginga_plugins
    $ python setup.py build

or

    $ cd ginga_plugins
    $ python setup.py install

The above steps are not necessary if you don't plan to use those
plugins.
