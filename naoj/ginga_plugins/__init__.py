import os.path
from ginga.misc.Bunch import Bunch


# my plugins are available here
p_path = os.path.dirname(__file__)

def setup_MOIRCS_Mask_Builder():
    spec = Bunch(path=os.path.join(p_path, 'MOIRCS_Mask_Builder.py'),
                 module='MOIRCS_Mask_Builder', klass='MOIRCS_Mask_Builder',
                 ptype='local', category="Spectroscopy",
                 workspace='Dialogs',
                 menu="MOIRCS Mask Builder", tab='MOIRCS MB',
                 enabled=True, exclusive=True)
    return spec
