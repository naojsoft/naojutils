"""
MOIRCS_Mask_Builder.py -- Ginga plugin to build masks for MOIRCS

Requirements
============

naojsoft packages
-----------------
- ginga
"""
# ginga
from ginga.gw import Widgets
from ginga import GingaPlugin


class MOIRCS_Mask_Builder(GingaPlugin.LocalPlugin):

    def __init__(self, fv, viewer):
        # superclass defines some variables for us, like logger
        super().__init__(fv, viewer)

        # get preferences
        prefs = self.fv.get_preferences()
        self.settings = prefs.create_category('plugin_MOIRCS_Mask_Builder')
        #self.settings.add_defaults()
        self.settings.load(onError='silent')

        self.viewer = viewer

    def build_gui(self, container):
        """Method called to build a GUI in the widget `container`"""
        top = Widgets.VBox()
        top.set_border_width(4)

        # Add a stretchy spacer
        container.add_widget(Widgets.Label(''), stretch=1)

        # Add Close and Help programs
        btns = Widgets.HBox()
        btns.set_border_width(4)
        btns.set_spacing(3)

        btn = Widgets.Button("Close")
        btn.add_callback('activated', lambda w: self.close())
        btns.add_widget(btn, stretch=0)
        btn = Widgets.Button("Help")
        #btn.add_callback('activated', lambda w: self.help())
        btns.add_widget(btn, stretch=0)
        btns.add_widget(Widgets.Label(''), stretch=1)

        top.add_widget(btns, stretch=0)

        container.add_widget(top, stretch=1)
        self.gui_up = True

    def close(self):
        """Called when user closes this plugin normally"""
        self.fv.stop_local_plugin(self.chname, str(self))
        return True

    def start(self):
        """Method called after build_gui() when the plugin is started.
        Do final post_GUI initialization here.
        """
        pass

    def stop(self):
        """Final method called when a plugin is stopped normally or abnormally"""
        self.gui_up = False

    def __str__(self):
        return 'moircs_mask_builder'
