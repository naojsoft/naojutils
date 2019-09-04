from ginga.misc.Bunch import Bunch

import os.path
# my plugins are available here
p_path = os.path.split(__file__)[0]

def setup_hscplanner():
    spec = Bunch(path=os.path.join(p_path, 'HSCPlanner.py'),
                 module='HSCPlanner', klass='HSCPlanner', tab='HSCPlanner',
                 category='Planning', workspace='dialogs', ptype='local')
    return spec

# END
