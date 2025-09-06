"""
A plugin to build masks for MOIRCS Instrument

**Plugin Type:** Local

`MOIRCS Mask Builder` is a local plugin, which means it is associated with a channel. An instance can be opened for each channel.

.. note:: This plugin is intended to replace the legacy MDP system. It supports loading and saving of `.mdp` and `.sbr` mask definition files and offers interactive slit and alignment hole editing capabilities on astronomical images with valid WCS.

**Usage**

**1. Load a FITS Image**

* Navigate to `File > Load Image`.
* Select and open the desired FITS file in the viewer.

**2. Launch the Plugin**

* Go to `Plugins > Subaru > Planning > MOIRCS Mask Builder` to activate the plugin panel for the current channel.

**3. Load an MDP File**

* Click the **Browse** button in the plugin.
* Select an existing `.mdp` file containing mask information.
* Click **Load** to populate the slit/hole list and overlay elements on the image.

**4. Toggle Detector Channels (New!)**

* Both channels (Ch1 and Ch2) are shown by default.
* To view channels separately, uncheck the desired channel checkboxes to hide them.

**5. Set the Field of View (FOV) Center**

* Enter the desired center coordinates (X, Y) in the input boxes.
* Press **Update** to reposition the field overlay accordingly.

**6. Display Options**

* Toggle the following options as needed:

  * **Slit/Hole ID** -- Show object IDs in the upper-right corner of each shape.
  * **Comments** -- Display user-entered comments near slits or holes.
  * **Excluded** -- Highlight excluded slits/holes in purple.

**7. View or Manage Slits and Holes**

* Click **Show Slit List** to open the full list of defined slits and holes.
* Items are displayed in ID order.
* You can toggle visibility or mark items for deletion (unchecked items will be commented out when saving).

**8. Auto Detection (New!)**

* The plugin automatically detects:

  * Overlapping slits or holes
  * Out-of-bound placements:

    * Outside circular field boundary
    * Outside central channel gap
    * More than +/- 3 arcsec from the centerline
* Lower-priority items will be auto-marked as **excluded**.
* Enable **Excluded** toggle to view these items in purple on the canvas.

**9. Add Slits or Holes**

* Click **Add**, then choose between **Slit** or **Hole** mode.
* Click on the canvas to place the center of the object.
* Enter an optional comment in the prompt dialog.
* A warning will appear if placed out of bounds (user may proceed regardless).

**10. Edit Existing Objects**

* Click **Edit**.
* Select an existing slit or hole from the dropdown list.
* Modify dimensions, orientation, or ID.
* Click **Apply** to update the object.

**11. Delete Slits or Holes**

* Open **Show Slit List**.
* Uncheck any item to exclude it from future saves (these will be commented out in the `.mdp` file).

**12. Undo Support (New!)**

* Basic undo functionality is now available for recent **Add** and **Edit** actions.
* Revert your last action with one click.

**13. Toggle Spectral Footprint (New!)**

* Use the **Spectra** checkbox to enable or disable overlaid spectra for slits, improving visibility for mask layout.

**14. Tick Mark selection**

* Default is **none**.
* Select a different Tick from the **Tick Marks** dropdown menu.

**15. Grism Selection and Parameters**

* Default grism is **Zj500**.
* Select a different grism from the **Grism** dropdown menu.
* To adjust grism parameters (e.g., tilt, dispersion), enter numeric values in the corresponding fields and press **Update**.

**16. Save to .mdp**

* Click **Save MDP**.
* Enter the desired filename and confirm to save the current layout.

**17. Save to .sbr**

* Click **Save SBR**.
* Confirm filename and FOV center (auto-filled from current settings).
* Header info includes the original `.mdp` file name and current center coordinates.

"""
# stdlib
import os.path
import copy

# 3rd party
import numpy as np
from astropy.units.quantity import Quantity

# ginga
from ginga.gw import Widgets
from ginga import GingaPlugin
from ginga.util.paths import icondir as ginga_icon_dir
from ginga.util import wcs

# local
from naoj.moircs.grism_info import grism_info_map
from naoj.moircs import mdp

# default center pixel (in FITS (1-based) indexing)
default_x_ctr, default_y_ctr = (1084, 1786)


class MOIRCS_Mask_Builder(GingaPlugin.LocalPlugin):
    def __init__(self, fv, fitsimage):
        super().__init__(fv, fitsimage)

        prefs = self.fv.get_preferences()
        self.settings = prefs.create_category('plugin_MOIRCS_Mask_Builder')
        self.settings.add_defaults(display_slitID=True, grism='zJ500',
                                   default_pixscale_arcsec=0.117)
        self.settings.load(onError='silent')

        self.grismtypes = list(grism_info_map.keys())
        default_grism = self.settings.get('grism', 'zJ500')
        self.grism_info = self.get_grism_info(default_grism)

        self.shapes = []  # Unified list for slits and holes
        self._undo_stack = []
        self._updating_grism_params = False
        self.show_excluded = False

        self.dc = fv.get_draw_classes()
        canvas = self.dc.DrawingCanvas()
        canvas.enable_draw(True)
        canvas.enable_edit(True)
        canvas.set_drawtype('rectangle')
        canvas.set_draw_mode('edit')
        canvas.set_surface(self.fitsimage)
        canvas.register_for_cursor_drawing(self.fitsimage)
        canvas.name = 'maskbuilder-canvas'
        self.canvas = canvas

        compass = self.dc.Compass(0.10, 0.10, 0.08,
                                  fontsize=14, coord='percentage',
                                  color='orange')
        self.canvas.add(compass, redraw=False)

        self.pixel_scale = self.settings['default_pixscale_arcsec']
        self.image_pa_deg = 0.0
        # our delta PA
        self.pa_deg = 0.0
        self.beta = 0.29898169
        self.mos_rot_deg = 0.0
        self.mdp_filename = 'UNKNOWN_MDP'
        self.fits_filename = 'UNKNOWN_IMAGE'
        # unit of angstroms
        self.valid_intervals = ['None', '100', '250', '500', '1000']
        self.fov_center = (default_x_ctr, default_y_ctr)
        self.det_fov = [4.0, 4.0]  # detector dimensions in arcminute
        self.gui_up = False

    def build_gui(self, container):
        top = Widgets.VBox()
        top.set_border_width(3)

        vbox, sw, orientation = Widgets.get_oriented_box(container, orientation=self.settings.get('orientation', None))
        vbox.set_border_width(4)
        vbox.set_spacing(2)

        # MOIRCS FOV Controls with Checkboxes
        fr = Widgets.Frame("MOIRCS FOV Controls")
        fov_controls = Widgets.VBox()
        fov_controls.set_spacing(3)

        hbox_fov = Widgets.HBox()
        hbox_fov.set_spacing(4)
        hbox_fov.add_widget(Widgets.Label("Detectors:"), stretch=0)

        self.w.cb_ch1 = Widgets.CheckBox("Show DET 1")
        self.w.cb_ch1.set_state(True)
        self.w.cb_ch1.add_callback('activated',
                                   lambda w, tf: self.update_fov())
        hbox_fov.add_widget(self.w.cb_ch1, stretch=0)

        self.w.cb_ch2 = Widgets.CheckBox("Show DET 2")
        self.w.cb_ch2.set_state(True)
        self.w.cb_ch2.add_callback('activated',
                                   lambda w, tf: self.update_fov())
        hbox_fov.add_widget(self.w.cb_ch2, stretch=0)

        fov_controls.add_widget(hbox_fov, stretch=0)

        hbox_center = Widgets.HBox()
        hbox_center.set_spacing(4)
        hbox_center.add_widget(Widgets.Label("FOV Center X:", halign='right'),
                               stretch=0)
        self.w.fov_center_x = Widgets.SpinBox()
        self.w.fov_center_x.set_limits(0, 7000, 1)
        self.w.fov_center_x.set_value(self.fov_center[0])
        hbox_center.add_widget(self.w.fov_center_x, stretch=0)

        hbox_center.add_widget(Widgets.Label("Y:", halign='right'), stretch=0)
        self.w.fov_center_y = Widgets.SpinBox()
        self.w.fov_center_y.set_limits(0, 7000, 1)
        self.w.fov_center_y.set_value(self.fov_center[1])
        hbox_center.add_widget(self.w.fov_center_y, stretch=0)

        # Update button
        btn_update = Widgets.Button("Update")
        btn_update.add_callback('activated', self.set_fov_center_from_user_input)
        hbox_center.add_widget(btn_update, stretch=0)
        fov_controls.add_widget(hbox_center, stretch=0)

        hbox = Widgets.HBox()
        hbox.add_widget(Widgets.Label("Image PA:", halign='right'), stretch=0)
        pa_lbl = Widgets.Label(f"{self.image_pa_deg:.1f}")
        self.w.pa_lbl = pa_lbl
        hbox.add_widget(pa_lbl, stretch=0)
        hbox.add_widget(Widgets.Label(""), stretch=1)
        hbox.add_widget(Widgets.Label("\u0394 PA (deg):", halign='right'), stretch=0)
        self.w.pa_deg = Widgets.SpinBox(dtype=float)
        self.w.pa_deg.set_limits(-180.0, 180.0, 1.0)
        self.w.pa_deg.set_value(0.0)
        self.w.pa_deg.set_tooltip("Set the delta to Position Angle of the field")
        self.w.pa_deg.add_callback('value-changed', self.set_pa_cb)
        hbox.add_widget(self.w.pa_deg, stretch=0)
        hbox.add_widget(Widgets.Label(''), stretch=1)
        fov_controls.add_widget(hbox, stretch=0)

        fr.set_widget(fov_controls)
        vbox.add_widget(fr, stretch=0)

        # --- Frame for Slit and Hole Controls ---
        fr_slit = Widgets.Frame("Slit and Hole Controls")
        vbox_slit = Widgets.VBox()
        vbox_slit.set_spacing(6)

        # Display Options (Slit/Hole ID, Comments, Show Excluded)
        hbox_sh_display = Widgets.HBox()
        hbox_sh_display.set_spacing(4)

        label_sh_display = Widgets.Label("Display Options:", halign='right')
        hbox_sh_display.add_widget(label_sh_display, stretch=0)

        self.w.display_slit_id = Widgets.CheckBox("Slit/Hole ID")
        self.w.display_slit_id.set_tooltip("Show slit or hole id beside item")
        self.w.display_slit_id.set_state(True)
        self.w.display_slit_id.add_callback('activated',
                                            lambda *args: self.draw_slits())
        hbox_sh_display.add_widget(self.w.display_slit_id, stretch=0)

        self.w.display_comments = Widgets.CheckBox("Comments")
        self.w.display_comments.set_tooltip("Show comments by slits")
        self.w.display_comments.set_state(True)
        self.w.display_comments.add_callback('activated',
                                             lambda *args: self.draw_slits())
        hbox_sh_display.add_widget(self.w.display_comments, stretch=0)

        self.w.show_excluded = Widgets.CheckBox("Excluded")
        self.w.show_excluded.set_tooltip("Show excluded slits or holes")
        self.w.show_excluded.add_callback('activated', lambda w, val: self.toggle_show_excluded(val))
        hbox_sh_display.add_widget(self.w.show_excluded, stretch=0)

        vbox_slit.add_widget(hbox_sh_display, stretch=0)

        # Row 1: Show Slit List + Auto Detection
        hbox_view_auto = Widgets.HBox()
        hbox_view_auto.set_spacing(4)

        btn_view_params = Widgets.Button("Show Slit List")
        btn_view_params.set_tooltip("Show the slit list with enabling checkboxes")
        btn_view_params.add_callback('activated', lambda w: self.show_slit_and_hole_info())

        btn_auto = Widgets.Button("Auto Detection")
        btn_auto.set_tooltip("Detect and exclude holes and slits outside detector area")
        btn_auto.add_callback('activated', lambda w: self.auto_detect_overlaps())

        hbox_view_auto.add_widget(btn_view_params, stretch=0)
        hbox_view_auto.add_widget(btn_auto, stretch=0)
        vbox_slit.add_widget(hbox_view_auto, stretch=0)

        # Row 2: Add + Edit + Undo
        hbox_add_edit = Widgets.HBox()
        hbox_add_edit.set_spacing(4)
        btn_add = Widgets.Button("Add")
        btn_add.set_tooltip("Add a hole or slit")
        btn_add.add_callback('activated', lambda w: self.add_slit_or_hole())
        btn_edit = Widgets.Button("Edit")
        btn_edit.set_tooltip("Edit a hole or slit")
        btn_edit.add_callback('activated', lambda w: self.edit_slit_or_hole())
        btn_undo = Widgets.Button("Undo")
        btn_undo.set_tooltip("Undo the last add/edit")
        btn_undo.add_callback('activated', lambda w: self.undo_last_edit())
        hbox_add_edit.add_widget(btn_add, stretch=0)
        hbox_add_edit.add_widget(btn_edit, stretch=0)
        hbox_add_edit.add_widget(btn_undo, stretch=0)
        vbox_slit.add_widget(hbox_add_edit, stretch=0)

        fr_slit.set_widget(vbox_slit)
        vbox.add_widget(fr_slit, stretch=0)

        # --- Frame for All Controls ---
        fr_controls = Widgets.Frame("Grism and Spectra Controls")
        vbox_controls = Widgets.VBox()
        vbox_controls.set_spacing(6)

        # Display Options (Spectra / Slit ID)
        hbox_display = Widgets.HBox()
        hbox_display.set_spacing(8)
        label_display = Widgets.Label("Display Options:", halign='right')
        hbox_display.add_widget(label_display, stretch=0)

        self.w.display_spectra = Widgets.CheckBox("Spectra")
        self.w.display_spectra.set_tooltip("Show spectral dispersion areas")
        self.w.display_spectra.add_callback('activated', lambda *args: self.draw_spectra())
        hbox_display.add_widget(self.w.display_spectra, stretch=0)

        vbox_controls.add_widget(hbox_display, stretch=0)

        # Spectra Dashed Line Interval Dropdown
        hbox_dashline = Widgets.HBox()
        hbox_dashline.set_spacing(6)
        hbox_dashline.add_widget(Widgets.Label("Tick Marks (\u212B):",
                                               halign='right'), stretch=0)

        self.w.dash_interval = Widgets.ComboBox(editable=True)
        self.w.dash_interval.set_tooltip("Show dashed lines in spectral dispersion boxes")
        for val in self.valid_intervals:
            self.w.dash_interval.append_text(val)
        self.w.dash_interval.set_index(0)
        self.w.dash_interval.add_callback('activated', self.dashline_change_cb)

        hbox_dashline.add_widget(self.w.dash_interval, stretch=0)
        vbox_controls.add_widget(hbox_dashline, stretch=0)

        # Grism selection
        hbox_grism = Widgets.HBox()
        hbox_grism.set_spacing(6)
        hbox_grism.add_widget(Widgets.Label("Grism:", halign='right'),
                              stretch=0)

        self.w.grism = Widgets.ComboBox()
        for name in self.grismtypes:
            self.w.grism.append_text(name)
        self.w.grism.set_index(self.grismtypes.index(self.settings.get('grism')))
        self.w.grism.add_callback('activated', lambda w, idx: self.set_grism())
        hbox_grism.add_widget(self.w.grism, stretch=0)

        vbox_controls.add_widget(hbox_grism, stretch=0)

        # Float parameter input using TextEntries
        param_fields = ["directwave", "wavestart", "waveend", "dispersion1",
                        "dispersion2", "dx1", "dx2", "tilt1", "tilt2"]
        labels = {"directwave": "Direct Wave (\u212B):",
                  "wavestart": "Wave Start (\u212B):",
                  "waveend": "Wave End (\u212B):",
                  "dispersion1": "Dispersion DET 1 (\u212B/px):",
                  "dispersion2": "Dispersion DET 2 (\u212B/px):",
                  "dx1": "DX DET 1:",
                  "dx2": "DX DET 2:",
                  "tilt1": "Tilt DET 1:",
                  "tilt2": "Tilt DET 2:",
                  }

        self.textentries = {}
        grid = Widgets.GridBox()
        grid.set_spacing(4)

        row = 0
        for key in param_fields:
            lbl = Widgets.Label(labels[key])
            entry = Widgets.TextEntry()
            val = self.grism_info.get(key, 0.0)
            entry.set_text(str(val))
            entry.add_callback('activated', lambda w, k=key: self.on_grism_param_changed(k))

            grid.add_widget(lbl, row, 0, stretch=0)
            grid.add_widget(entry, row, 1, stretch=1)
            self.textentries[key] = entry
            row += 1
        vbox_controls.add_widget(grid, stretch=0)

        # Update/Reset Grism Buttons
        btn_box = Widgets.HBox()
        btn_box.set_spacing(4)
        btn_update = Widgets.Button("Update")
        btn_update.add_callback('activated', lambda w: self.update_all_grism_params())
        btn_reset = Widgets.Button("Reset")
        btn_reset.add_callback('activated', lambda w: self.reset_grism_params())
        btn_box.add_widget(btn_update, stretch=0)
        btn_box.add_widget(btn_reset, stretch=0)

        vbox_controls.add_widget(btn_box, stretch=0)

        fr_controls.set_widget(vbox_controls)
        vbox.add_widget(fr_controls, stretch=0)

        # Add Slit or Hole dialog
        dialog = Widgets.Dialog(title="Add Slit or Hole",
                                buttons=[("Cancel", 1), ("OK", 0)],
                                parent=self.fv.w.root)
        content = dialog.get_content_area()
        content.set_border_width(4)
        combo = Widgets.ComboBox()
        for label in ["Slit (Rectangle)", "Hole (Circle)"]:
            combo.append_text(label)
        content.add_widget(Widgets.Label("Select shape to add:"), stretch=0)
        content.add_widget(combo, stretch=1)
        dialog.add_callback('activated', self.add_slit_cb, combo)
        self.w.add_slit_dialog = dialog

        # Slit and Hole Manager dialog
        dialog = Widgets.Dialog(title="Slit and Hole Manager",
                                buttons=[("Close", 0)],
                                parent=self.fv.w.root)
        content = dialog.get_content_area()
        content.set_border_width(4)
        scroll = Widgets.ScrollArea()
        gbox = Widgets.GridBox(columns=1)
        scroll.set_widget(gbox)
        content.add_widget(scroll, stretch=1)
        self.w.slits_gbox = gbox
        self.w.slits_dialog = dialog
        dialog.add_callback('activated', lambda w, val: w.hide())

        # Confirm Center dialog
        dialog = Widgets.Dialog(title="Confirm Center Pixel",
                                buttons=[("Cancel", 1), ("Confirm", 0)],
                                parent=self.fv.w.root)
        content = dialog.get_content_area()
        content.set_border_width(4)
        dialog.add_callback('activated', self.confirm_center_dialog_cb)
        self.w.confirm_center_dialog = dialog

        # --- Bottom Buttons (Load/Save/etc.) ---
        btns = Widgets.HBox()
        btns.set_spacing(4)

        self.w.filepath = Widgets.TextEntry()
        btn_browse = Widgets.Button("Load MDP")
        btn_browse.set_tooltip("Load an MDP file")
        # Load MDP file dialog
        self.w.fbrowser = Widgets.FileDialog(title="Select MDP file",
                                             parent=self.fv.w.root)
        self.w.fbrowser.set_mode('file')
        self.w.fbrowser.add_ext_filter("MDP files", '.mdp')
        self.w.fbrowser.add_callback('activated', self.browse_file_cb)
        btn_browse.add_callback('activated', lambda w: self.w.fbrowser.popup())

        btn_reload = Widgets.Button("Reload")
        btn_reload.set_tooltip("Reload the MDP file")
        btn_reload.add_callback('activated', lambda w: self.load_file(self.w.filepath.get_text()))

        btns.add_widget(btn_browse, stretch=0)
        btns.add_widget(self.w.filepath, stretch=1)
        btns.add_widget(btn_reload, stretch=0)
        vbox.add_widget(btns, stretch=0)

        btns = Widgets.HBox()
        btns.set_spacing(4)

        btn_save_mdp = Widgets.Button("Save MDP")
        btn_save_mdp.set_tooltip("Save MDP file")
        # Save MDP file dialog
        self.w.save_mdp = Widgets.FileDialog(title="Save MDP file",
                                             parent=self.fv.w.root)
        self.w.save_mdp.set_mode('save')
        self.w.save_mdp.add_ext_filter(".mdp files", '.mdp')
        self.w.save_mdp.add_callback('activated', self.save_mdp_file_cb)
        btn_save_mdp.add_callback('activated', lambda w: self.w.save_mdp.popup())


        btn_save_sbr = Widgets.Button("Save SBR")
        btn_save_sbr.set_tooltip("Save SBR file")
        # Save SBR file dialog
        self.w.save_sbr = Widgets.FileDialog(title="Save SBR file",
                                             parent=self.fv.w.root)
        self.w.save_sbr.set_mode('save')
        self.w.save_sbr.add_ext_filter(".sbr files", '.sbr')
        self.w.save_sbr.add_callback('activated', self.save_sbr_file_cb)
        btn_save_sbr.add_callback('activated', self.confirm_center_dialog)

        btns.add_widget(Widgets.Label(''), stretch=1)
        btns.add_widget(btn_save_mdp, stretch=0)
        btns.add_widget(btn_save_sbr, stretch=0)
        vbox.add_widget(btns, stretch=0)

        btns = Widgets.HBox()
        btns.set_spacing(3)

        btn_close = Widgets.Button("Close")
        btn_close.add_callback('activated', lambda w: self.close())
        btns.add_widget(btn_close, stretch=0)

        btn_help = Widgets.Button("Help")
        btn_help.add_callback('activated', lambda w: self.help())
        btns.add_widget(btn_help, stretch=0)

        btns.add_widget(Widgets.Label(''), stretch=1)

        # Add to main container
        top.add_widget(sw, stretch=1)
        top.add_widget(btns, stretch=0)
        container.add_widget(top, stretch=1)

        self.update_fov()

        self.gui_up = True

    def set_entry_value(self, key, val):
        if key in self.textentries:
            self.textentries[key].set_text(str(val))

    def get_entry_value(self, key):
        if key in self.textentries:
            try:
                return float(self.textentries[key].get_text().strip())

            except ValueError:
                return 0.0  # or log a warning
        return 0.0

    def set_fov_center_from_user_input(self, widget):
        x = int(self.w.fov_center_x.get_value())
        y = int(self.w.fov_center_y.get_value())
        self.fov_center = (x, y)
        # NOTE: account for FITS indexing vs. canvas indexing
        self.fitsimage.set_pan(x - 1, y - 1)
        self.update_fov()

    def set_fov_center(self, x, y):
        x, y = int(x), int(y)
        self.fov_center = (x, y)
        self.update_fov()

    def set_fov_center_from_image(self):
        image = self.fitsimage.get_image()
        if image is not None:
            width, height = image.get_size()
            x, y = int(width * 0.5), int(height * 0.5)
            # NOTE: account for FITS indexing vs. canvas indexing
            self.set_fov_center(x_center + 1, y_center + 1)

    def update_fov(self):
        self.draw_fov()

        # redraw everything else
        self.update_slits_spectra()

        self.canvas.redraw(whence=3)

    def update_slits_spectra(self):
        self.draw_slits()
        self.draw_spectra()

    def browse_file_cb(self, w, paths):
        if len(paths) > 0:
            file_path = paths[0]
            self.w.filepath.set_text(file_path)
            self.load_file(file_path)

    def load_file(self, filepath):
        if isinstance(filepath, tuple):
            filepath = filepath[0]
        if filepath and os.path.exists(filepath):
            self.mdp_filename = filepath
            self.load_mdp(filepath)
            self.update_slits_spectra()

    def load_mdp(self, filepath):
        rows = mdp.load_mdp(filepath)
        self.shapes = rows
        self._undo_stack = []

    def show_slit_and_hole_info(self):
        gbox = self.w.slits_gbox
        gbox.remove_all(delete=True)

        for i, shape in enumerate(self.shapes):
            shape_type = 'Slit' if shape['type'].startswith('B') else 'Hole'
            comment = shape.get('comment', '')
            label = f"{shape_type} #{i} | x={shape['x']:.1f}, y={shape['y']:.1f} | {comment}"

            cb = Widgets.CheckBox(label)
            gbox.add_widget(cb, i, 0)
            # Checked = included; Unchecked = either _deleted or _excluded
            cb.set_state(not shape.get('_deleted', False) and not shape.get('_excluded', False))
            cb.add_callback('activated', self.slit_manager_cb, i)

        self.w.slits_dialog.show()

    def slit_manager_cb(self, w, checked, i):
        self.shapes[i]['_deleted'] = not checked
        self.shapes[i]['_excluded'] = not checked
        self.update_slits_spectra()

    def toggle_show_excluded(self, val):
        self.show_excluded = val
        self.update_slits_spectra()

    def auto_detect_overlaps(self):
        if not self.shapes:
            self.message_box('info', "Info", "No shapes to analyze.")
            return

        # Reset exclusions
        for shape in self.shapes:
            shape['_excluded'] = False

        excluded_count = 0

        def get_x_bounds(shape):
            x = shape['x']
            if shape['type'].startswith('B'):
                w = shape.get('width', 100.0)
            else:
                w = shape.get('diameter', 30.0)
            return x - w / 2, x + w / 2

        y_center = self.fov_center[1]
        n = len(self.shapes)

        for i in range(n):
            s1 = self.shapes[i]
            if s1.get('_excluded'):
                continue

            x1_min, x1_max = get_x_bounds(s1)
            ch1 = s1['y'] < y_center

            if (not self.is_within_fov_bounds(s1['x'], s1['y']) or
                    not self.is_within_y_arcsec_limit(s1['y'], min_arcsec_from_center=3)):
                s1['_excluded'] = True
                excluded_count += 1
                continue

            for j in range(i + 1, n):
                s2 = self.shapes[j]
                if s2.get('_excluded'):
                    continue

                ch2 = s2['y'] < y_center
                if ch1 != ch2:
                    continue

                x2_min, x2_max = get_x_bounds(s2)
                if x1_max >= x2_min and x2_max >= x1_min:
                    s2['_excluded'] = True
                    excluded_count += 1

        self.update_slits_spectra()
        self.message_box('info', "Auto Detection", f"Excluded {excluded_count} shape(s).")

    def add_slit_or_hole(self):
        self._undo_stack.append({'shapes': copy.deepcopy(self.shapes)})

        self.w.add_slit_dialog.show()

    def add_slit_cb(self, w, val, combo):
        w.hide()
        if val == 1:
            # cancel
            return
        self.logger.info("adding slit")
        choice = combo.get_text()
        self._add_shape_type = 'slit' if choice.startswith("Slit") else 'hole'
        self.canvas.set_drawtype('point')
        self.canvas.set_draw_mode('draw')
        self.canvas.set_callback('button-press', self._on_click_event)
        self.canvas.ui_set_active(True, viewer=self.fitsimage)

    def is_within_fov_bounds(self, x, y):
        """Check if (x, y) is within MOIRCS rectangle in x, and
        circle radius in y."""
        # Rectangle half-width in pixels
        xr = (self.det_fov[0] * 60) / self.pixel_scale * 0.5
        # Circle radius in pixels
        radius_6 = (6.0 * 60) / self.pixel_scale * 0.5

        x_center, y_center = self.fov_center
        within_x = (x_center - xr) <= x <= (x_center + xr)
        r = np.hypot(x - x_center, y - y_center)
        within_radius = r <= radius_6
        return within_x and within_radius

    def is_within_y_arcsec_limit(self, y, min_arcsec_from_center=10):
        """Ensure the slit/hole is at least +/- min_arcsec_from_center from the centerline."""
        y_center = self.fov_center[1]
        min_pixel_dist = min_arcsec_from_center / 3600.0 / self.pixel_scale
        return np.fabs(y - y_center) >= min_pixel_dist

    def _on_click_event(self, canvas, button, data_x, data_y):
        self.canvas.set_draw_mode(None)
        self.canvas.remove_callback('button-press', self._on_click_event)
        self.canvas.ui_set_active(False, viewer=self.fitsimage)

        # convert canvas coords to FITS coords
        x = data_x + 1
        y = data_y + 1

        out_of_bounds = (not self.is_within_fov_bounds(x, y) or
                         not self.is_within_y_arcsec_limit(y))
        if out_of_bounds:
            self.message_box('warning', "Out of Bounds", "The selected position is outside the allowed FOV, but it will be added.")

        dialog = Widgets.Dialog(title="Confirm New Shape",
                                buttons=[("Confirm", 0), ("Cancel", 1)],
                                parent=self.fv.w.root)
        layout = dialog.get_content_area()
        layout.set_border_width(4)
        layout.add_widget(Widgets.Label(f"Add new {self._add_shape_type} at x={x:.1f}, y={y:.1f}?"), stretch=0)
        comment_field = Widgets.TextEntry()
        layout.add_widget(Widgets.Label("Comment:"), stretch=0)
        layout.add_widget(comment_field, stretch=0)

        def on_confirm(w, val):
            w.hide()
            comment = comment_field.get_text()
            shape = {'x': x, 'y': y, 'comment': comment}
            if out_of_bounds:
                # Initially excluded from auto detection/export
                shape['_excluded'] = True
            if self._add_shape_type == 'slit':
                shape.update({'type': 'B', 'width': 100,
                              'length': 7, 'angle': 0, 'priority': '1'})
            else:
                shape.update({'type': 'C', 'diameter': 30})
            self.shapes.append(shape)
            self.update_slits_spectra()

        dialog.add_callback('activated', on_confirm)
        self.w.confirm_click_dialog = dialog
        dialog.show()

    def edit_slit_or_hole(self):
        dialog = Widgets.Dialog(title="Edit Slit or Hole",
                                buttons=[("Apply", 0), ("Close", 1)])
        layout = dialog.get_content_area()
        layout.set_border_width(4)
        layout.add_widget(Widgets.Label("Select ID to edit:"))
        combo = Widgets.ComboBox()
        id_map = {}
        j = 0
        for i, shape in enumerate(self.shapes):
            if shape.get('_deleted'):
                continue
            prefix = 'B' if shape['type'].startswith('B') else 'C'
            label = f"{prefix}{i}: {shape.get('comment', '')}"
            combo.append_text(label)
            id_map[j] = shape
            j += 1
        layout.add_widget(combo, stretch=0)
        fields_widget = Widgets.VBox()
        layout.add_widget(fields_widget)
        current_fields = {}

        def clear_fields():
            fields_widget.remove_all()

        def add_field(name, initial_value):
            lbl = Widgets.Label(name)
            le = Widgets.TextEntry()
            le.set_text(str(initial_value))
            fields_widget.add_widget(lbl, stretch=0)
            fields_widget.add_widget(le, stretch=0)
            current_fields[name] = le

        def populate_fields(index):
            clear_fields()
            shape = id_map[index]
            add_field("X:", shape.get('x', ''))
            add_field("Y:", shape.get('y', ''))
            if shape['type'].startswith('B'):
                add_field("Width:", shape.get('width', ''))
                add_field("Length:", shape.get('length', ''))
                add_field("Angle:", shape.get('angle', ''))
            else:
                add_field("Diameter:", shape.get('diameter', ''))
            add_field("Comment:", shape.get('comment', ''))

        combo.add_callback('activated', lambda w, idx: populate_fields(idx))
        populate_fields(0)

        def apply_changes(w, val):
            if val == 1:
                w.hide()
                return
            shape = id_map[combo.get_index()]
            try:
                x = float(current_fields["X:"].get_text())
                y = float(current_fields["Y:"].get_text())

                if not self.is_within_fov_bounds(x, y) or not self.is_within_y_arcsec_limit(y):
                    self.message_box('warning', "Out of Bounds", "The specified position is outside the allowed FOV.", parent=dialog)
                    return

                if shape['type'].startswith('B'):
                    width = float(current_fields["Width:"].get_text())
                    length = float(current_fields["Length:"].get_text())
                    angle = float(current_fields["Angle:"].get_text())
                    if width < 35:
                        self.message_box('warning', "Invalid input", "Width must be at least 35.",
                                         parent=dialog)
                        return
                    if length < 6.8:
                        self.message_box('warning', "Invalid input", "Length must be at least 6.8.", parent=dialog)
                        return
                else:
                    diameter = float(current_fields["Diameter:"].get_text())
                    if diameter < 20 or diameter > 30:
                        self.message_box('warning', "Invalid input", "Diameter must be between 20 and 30.", parent=dialog)
                        return

                self._undo_stack.append({'shapes': copy.deepcopy(self.shapes)})
                shape['x'] = x
                shape['y'] = y
                shape['comment'] = current_fields["Comment:"].get_text()
                if shape['type'].startswith('B'):
                    shape['width'] = width
                    shape['length'] = length
                    shape['angle'] = angle
                else:
                    shape['diameter'] = diameter

                self.update_slits_spectra()

            except ValueError:
                self.message_box('warning', "Invalid input", "Please enter valid numeric values.", parent=dialog)

        dialog.add_callback('activated', apply_changes)
        self.w.edit_slit_dialog = dialog
        dialog.show()

    def undo_last_edit(self):
        if not self._undo_stack:
            self.fv.show_error("Nothing to undo!", raisetab=True)
            return
        last_state = self._undo_stack.pop()
        self.shapes = last_state['shapes']
        self.update_slits_spectra()
        self.logger.info("undo!")

    def on_grism_param_changed(self, key):
        if self._updating_grism_params:
            return
        val = self.get_entry_value(key)
        self.grism_info[key] = val

    def set_grism(self):
        grism_name = self.w['grism'].get_text()
        self.settings.set(dict(grism=grism_name))
        self.grism_info = self.get_grism_info(grism_name)

        self._updating_grism_params = True
        for key, val in self.grism_info.items():
            self.set_entry_value(key, val)
        self._updating_grism_params = False

        self.draw_spectra()

    def update_all_grism_params(self):
        for key in self.textentries:
            self.grism_info[key] = self.get_entry_value(key)
        self.draw_spectra()

    def reset_grism_params(self):
        grism_name = self.w['grism'].get_text()
        original_info = self.get_grism_info(grism_name)

        self._updating_grism_params = True
        for key in self.textentries:
            val = original_info.get(key, 0.0)
            self.set_entry_value(key, val)
            self.grism_info[key] = val
        self._updating_grism_params = False
        self.draw_spectra()

    def draw_fov(self):
        # NOTE: account for FITS indexing vs. canvas indexing
        fov_center_g = np.array(self.fov_center) - 1
        xc, yc = fov_center_g
        rot_deg = - self.pa_deg
        self.canvas.delete_object_by_tag('fov', redraw=False)

        # Apply 4 arcsecond leftward (-x) offset
        # Ichi-san says this might only need to happen at the MDP=>SBR time
        #xc -= 4.0 / 3600.0 / self.pixel_scale

        radius_6 = (6.0 * 60) / self.pixel_scale * 0.5
        radius_8 = (8.0 * 60) / self.pixel_scale * 0.5
        c1 = self.dc.Circle(xc, yc, radius_6,
                            linewidth=1, linestyle='solid', color='cyan')
        c2 = self.dc.Circle(xc, yc, radius_8,
                            linewidth=1, linestyle='solid', color='brown')

        # mark the center
        p1 = self.dc.Point(xc, yc, radius=15, style='cross',
                           linewidth=1, linestyle='solid', color='cyan')
        objs = [c1, c2, p1]

        # calc offset from center pixel to upper and lower box centers
        offset = (1.5 * 60) / self.pixel_scale
        r_wd = (self.det_fov[0] * 60) / self.pixel_scale * 0.5
        r_ht = (self.det_fov[1] * 60) / self.pixel_scale * 0.5
        if self.w.cb_ch1.get_state():
            d1 = self.dc.Box(xc, yc - offset, r_wd, r_ht, rot_deg=rot_deg,
                             linewidth=1, linestyle='solid', color='yellow')
            t1 = self.dc.Text(xc + r_wd, yc - offset - r_ht, text="DET 1",
                              color='yellow', bgcolor='black', bgalpha=1.0,
                              rot_deg=rot_deg)
            objs.extend([d1, t1])

        if self.w.cb_ch2.get_state():
            d2 = self.dc.Box(xc, yc + offset, r_wd, r_ht, rot_deg=rot_deg,
                             linewidth=1, linestyle='solid', color='pink')
            t2 = self.dc.Text(xc + r_wd, yc + offset + r_ht, text="DET 2",
                              color='pink', bgcolor='black', bgalpha=1.0,
                              rot_deg=rot_deg)
            objs.extend([d2, t2])
        lc = self.dc.Line(xc - r_wd, yc, xc + r_wd, yc,
                          linewidth=1, linestyle='dash', color='cyan')
        objs.append(lc)
        fov = self.dc.CompoundObject(*objs)
        self.canvas.add(fov, tag='fov', redraw=False)
        fov.rotate_deg([rot_deg], fov_center_g)

        self.canvas.redraw(whence=3)

    def draw_slits(self):
        # Clear all slit, hole, and label objects
        self.canvas.delete_objects_by_tag(['slits'], redraw=False)

        show_ids = self.w.display_slit_id.get_state()
        show_comments = self.w.display_comments.get_state()
        rot_deg = - self.pa_deg

        # NOTE: convert FITS image coords to canvas coords
        fov_center_g = np.array(self.fov_center) - 1
        y_center = fov_center_g[1]

        objects = []
        for i, shape in enumerate(self.shapes):
            if shape.get('_deleted') or (shape.get('_excluded') and not self.show_excluded):
                continue

            x, y = shape['x'], shape['y']
            comment = shape.get('comment', '')

            # Convert FITS image coords to canvas coords
            xcen = x - 1
            ycen = y - 1

            # Assign shape to CH1 (ycen <= y_center) or CH2 (ycen > y_center)
            is_ch1 = ycen <= y_center
            is_ch2 = ycen > y_center

            # Skip if the shape's channel is unchecked
            if is_ch1 and not self.w.cb_ch1.get_state():
                continue
            if is_ch2 and not self.w.cb_ch2.get_state():
                continue

            if shape['type'].startswith('B'):
                # Draw slit (box)
                w = shape['width']
                l = shape['length']
                angle = shape.get('angle', 0.0) - self.pa_deg
                color = 'purple' if shape.get('_excluded') else 'white'
                box = self.dc.Box(xcen, ycen, w * 0.5, l * 0.5,
                                  rot_deg=angle,
                                  color=color, linewidth=1)
                objects.append(box)

                if show_ids:
                    objects.append(self.dc.Text(xcen, ycen + l / 2 + 10,
                                                text=f"{i}", color='white',
                                                fontsize=11, rot_deg=angle))

                if show_comments and comment:
                    comment_text = self.dc.Text(xcen, ycen - l / 2 - 30,
                                                text=comment, color='white',
                                                rot_deg=angle)
                    objects.append(comment_text)

            elif shape['type'].startswith('C'):
                # Draw hole (circle)
                diameter = shape.get('diameter', 30.0)
                radius = diameter / 2
                angle = - self.pa_deg
                color = 'purple' if shape.get('_excluded') else 'yellow'
                objects.append(self.dc.Circle(xcen, ycen, radius,
                                              color=color, linewidth=1))

                if show_ids:
                    objects.append(self.dc.Text(xcen, ycen + radius + 10,
                                                text=f"{i}", color='yellow',
                                                fontsize=11, rot_deg=angle))

                if show_comments and comment:
                    objects.append(self.dc.Text(xcen,
                                                ycen - radius - 30,
                                                text=comment, color='yellow',
                                                rot_deg=angle))

        if len(objects) > 0:
            slits = self.dc.CompoundObject(*objects)
            self.canvas.add(slits, tag='slits', redraw=False)
            slits.rotate_deg([rot_deg], fov_center_g)

        self.canvas.redraw(whence=3)

    def draw_spectra(self):
        # Clean up previously drawn spectra-related objects
        self.canvas.delete_objects_by_tag(['spectra'], redraw=False)

        if not self.w.display_spectra.get_state():
            self.fitsimage.redraw(whence=3)
            return

        g = self.grism_info
        if not g:
            self.fitsimage.redraw(whence=3)
            return

        # NOTE: convert FITS image coords to canvas coords
        fov_center_g = np.array(self.fov_center) - 1

        rot_deg = - self.pa_deg
        y_center = fov_center_g[1]
        direct_wave = g.get('directwave', 0)
        wave_start = g.get('wavestart', 0)
        wave_end = g.get('waveend', 0)
        dispersion1 = g.get('dispersion1', 1)
        dispersion2 = g.get('dispersion2', 1)
        dx1 = g.get('dx1', 0)
        dx2 = g.get('dx2', 0)
        tilt1 = g.get('tilt1', 0.0)
        tilt2 = g.get('tilt2', 0.0)

        if dispersion1 == 0 or dispersion2 == 0:
            self.logger.error("Invalid dispersion: 0")
            self.fitsimage.redraw(whence=3)
            return

        objects = []

        dash_text = self.w.dash_interval.get_text().strip().lower()
        dash_interval = None
        if dash_text != 'none' and len(dash_text) > 0:
            try:
                dash_interval = int(dash_text)
            except ValueError:
                # TODO: show an error
                pass

        # --- Efficient "center-outward" dashed line drawing ---
        def draw_dashes(objects, x, y, width, spec_y1, spec_y2,
                        interval_y, color, tilt_deg):
            # Clamp direction
            ymin, ymax = sorted([spec_y1, spec_y2])
            x_start = x - width * 0.5
            x_end = x + width * 0.5

            # Generate lines from center outwards
            for direction in [-1, 1]:  # up and down
                offset = 0.0
                while True:
                    y_pos = y + direction * offset
                    if y_pos < ymin or y_pos > ymax:
                        break
                    line = self.dc.Line(x_start, y_pos, x_end, y_pos,
                                        color=color, linewidth=1,
                                        linestyle='dash', coord='data')
                    line.crdmap = self.fitsimage.get_coordmap('data')
                    line.rotate_deg([tilt_deg], (x, y))
                    objects.append(line)
                    offset += interval_y

        # --- Draw shapes ---
        for i, shape in enumerate(self.shapes):
            if shape.get('_deleted') or (shape.get('_excluded') and not self.show_excluded):
                continue

            # NOTE: account for FITS indexing vs. canvas indexing
            xcen, ycen = shape['x'] - 1, shape['y'] - 1
            # apply DX offset
            is_det1 = (ycen <= y_center)
            xcen += dx1 if is_det1 else dx2

            # skip drawing if user doesn't want to see this detector
            if (is_det1 and not self.w.cb_ch1.get_state()) or (not is_det1 and not self.w.cb_ch2.get_state()):
                continue

            width = shape.get('width', 100.0) if shape['type'].startswith('B') else shape.get('diameter', 30.0)
            interval_y = 100

            if is_det1:
                # Detector 1
                bottom_length = (wave_start - direct_wave) / dispersion1
                top_length = (direct_wave - wave_end) / dispersion1
                spec_y1 = ycen + top_length
                spec_y2 = ycen - bottom_length
                if dash_interval is not None:
                    interval_y = dash_interval / dispersion2
                color = 'green'
                tilt_deg = tilt1
            else:
                # Detector 2
                bottom_length = (wave_start - direct_wave) / dispersion2
                top_length = (direct_wave - wave_end) / dispersion2
                spec_y1 = ycen - top_length
                spec_y2 = ycen + bottom_length
                if dash_interval is not None:
                    interval_y = dash_interval / dispersion2
                color = 'red'
                tilt_deg = tilt2

            # --- Spectral Rectangle ---
            points = [(xcen - width * 0.5, spec_y1),
                      (xcen + width * 0.5, spec_y1),
                      (xcen + width * 0.5, spec_y2),
                      (xcen - width * 0.5, spec_y2)]
            # TODO: rotate points according to tilt?
            poly = self.dc.Polygon(points, color=color,
                                   linewidth=1, fill=False, coord='data')
            poly.crdmap = self.fitsimage.get_coordmap('data')
            poly.rotate_deg([tilt_deg], (xcen, ycen))
            objects.append(poly)

            if dash_interval is not None:
                draw_dashes(objects, xcen, ycen, width, spec_y1, spec_y2,
                            interval_y, color, tilt_deg)

        if len(objects) > 0:
            spectra = self.dc.CompoundObject(*objects)
            self.canvas.add(spectra, tag="spectra", redraw=False)
            spectra.rotate_deg([rot_deg], fov_center_g)

        self.fitsimage.redraw(whence=3)

    def dashline_change_cb(self, w, idx):
        self.draw_spectra()

    def set_pa_cb(self, w, val):
        self.pa_deg = val
        self.update_fov()

    def save_mdp_file_cb(self, w, paths):
        if len(paths) == 0:
            return
        filename = paths[0]

        with open(filename, 'w') as f:
            for shape in self.shapes:
                x = shape['x']
                y = shape['y']
                comment = shape.get('comment', '')
                if shape['type'].startswith('B'):
                    w = shape['width']
                    l = shape['length']
                    a = shape['angle']
                    line = f"{x:.2f} {y:.2f} {w:.0f} {l:.0f} {a:.0f} 1 B, {comment}\n"
                else:
                    d = shape['diameter']
                    line = f"{x:.2f} {y:.2f} {d:.0f} {d:.0f} 0 0 C, {comment}\n"
                if shape.get('_deleted') or shape.get('_excluded'):
                    f.write(f"# {line}")
                else:
                    f.write(line)

    def confirm_center_dialog(self, w):
        dialog = self.w.confirm_center_dialog
        content = dialog.get_content_area()
        content.remove_all(delete=True)

        fov_x_ctr, fov_y_ctr = self.fov_center
        lbl = f"FOV center X: ({fov_x_ctr:.2f}), Y: ({fov_y_ctr:.2f})"
        content.add_widget(Widgets.Label(lbl))
        content.add_widget(Widgets.Label("Cancel and change up top if needed"))
        dialog.show()

    def confirm_center_dialog_cb(self, w, val):
        w.hide()
        if val == 1:
            # cancel
            return
        # ok-- go ahead and show the Save SBR dialog
        self.w.save_sbr.popup()

    def write_sbr_file(self, path):
        try:
            shapes_filtered = [s for s in self.shapes
                               if not s.get('_deleted')]

            tbl = mdp.rows2table(shapes_filtered)
            buf, warnings = mdp.table2sbr(tbl, self.fov_center,
                                          self.pixel_scale)

            for warn_str in warnings:
                self.message_box('warning', "Warning", warn_str)

            fov_x_ctr, fov_y_ctr = self.fov_center

            with open(path, 'w') as out_f:
                out_f.write(f"# MDP: {self.mdp_filename}\n")
                out_f.write(f"# Image: {self.fits_filename}\n")
                out_f.write(f"# FOV Center: x={fov_x_ctr:.2f}, y={fov_y_ctr:.2f}\n")
                out_f.write(buf)

        except IOError as e:
            self.message_box('critical', "Error", f"Failed to write SBR file: {str(e)}")
        else:
            self.message_box('info', "Status", f"Wrote SBR file: {path}")

    def save_sbr_file_cb(self, w, paths):
        if len(paths) == 0:
            # cancel
            return

        path = paths[0]
        self.write_sbr_file(path)

    def message_box(self, category, title, message, parent=None):
        warn = Widgets.Dialog(title=title, modal=False,
                              parent=self.fv.w.root,
                              buttons=[("Dismiss", 0)])
        vbox = warn.get_content_area()
        vbox.set_margins(4, 4, 4, 4)
        hbox = Widgets.HBox()
        hbox.set_border_width(4)
        hbox.add_widget(Widgets.Label(""), stretch=1)
        img = Widgets.Image()
        # TODO: critical, warning, info -- different icons
        iconfile = os.path.join(ginga_icon_dir, "warning.svg")
        img.load_file(iconfile)
        hbox.add_widget(img, stretch=0)
        hbox.add_widget(Widgets.Label(""), stretch=1)
        vbox.add_widget(hbox, stretch=1)
        vbox.add_widget(Widgets.Label(message))
        warn.add_callback('activated', lambda w, val: w.hide())
        warn.add_callback('close', lambda w: w.hide())
        self.w.warning = warn
        warn.show()

    def get_grism_info(self, grism_name):
        info_dct = dict(grism_info_map.get(grism_name, {}))
        for key, val in list(info_dct.items()):
            # convert quantities to raw numbers
            if isinstance(val, Quantity):
                info_dct[key] = val.value
        return info_dct

    def close(self):
        self.fv.stop_local_plugin(self.chname, str(self))
        return True

    def start(self):
        p_canvas = self.fitsimage.get_canvas()
        if self.canvas not in p_canvas:
            p_canvas.add(self.canvas, tag='maskbuilder-canvas')

        self.redo()

    def stop(self):
        self.gui_up = False
        p_canvas = self.fitsimage.get_canvas()
        if self.canvas in p_canvas:
            p_canvas.delete_object(self.canvas)

    def redo(self):
        # <-- FITS image is loaded
        if not self.gui_up:
            return

        image = self.fitsimage.get_image()
        if image is None:
            self.pixel_scale = self.settings['default_pixscale_arcsec']
            width, height = int(2 * default_x_ctr), int(2 * default_y_ctr)
            self.fits_filename = 'UNKNOWN_IMAGE'

            self.image_pa_deg = 0.0
        else:
            #self.set_fov_center_from_image(image)
            path = image.get('path', 'UNKNOWN_IMAGE')
            self.fits_filename = os.path.basename(path)

            # figure out pixel scale
            header = image.get_header()
            try:
                res = wcs.get_xy_rotation_and_scale(header)
                (xrot_deg, yrot_deg), (cdelt1, cdelt2) = res
                self.logger.debug(f"cdelt1={cdelt1:.8f}, cdelt2={cdelt2:.8f}")
                # convert to arcsec/px
                self.pixel_scale = np.max([np.fabs(cdelt1),
                                        np.fabs(cdelt2)]) * 3600.0

                self.image_pa_deg = xrot_deg

            except Exception as e:
                self.logger.error(f"failed to get scale of image: {e}",
                                  exc_info=True)

        self.w.pa_lbl.set_text(f"{self.image_pa_deg:.1f}")
        self.logger.info(f"setting pixel scale to {self.pixel_scale:.5f}")

        x, y = self.fov_center
        self.fitsimage.set_pan(x, y)
        self.update_fov()

    def __str__(self):
        return 'moircs_mask_builder'
