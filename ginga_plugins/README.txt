ABOUT
-----
This directory contains optional plugins that can be used with the Ginga
science image viewer.

INSTALLATION (via setuptools)
------------
$ python setup.py install

and start ginga with

$ ginga

OR (manual way)
---------------
$ mkdir $HOME/.ginga
$ cp -r plugins $HOME/.ginga/.

and restart ginga with the plugin you want:

$ ginga --plugins=<PLUGIN>

