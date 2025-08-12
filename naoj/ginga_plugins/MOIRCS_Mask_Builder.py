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

* Go to `Plugins > Spectroscopy > MOIRCS Mask Builder` to activate the plugin panel for the current channel.

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
* NOTE (!): Spectral dashed line rendering is under development.
    * Intervals below 200 may degrade performance or impact other features.
    * For stability, resetting to the default value is recommended before using other functions.

**15. Grism Selection and Parameters**

* Default grism is **Zj500**.
* Select a different grism from the **Grism** dropdown menu.
* To adjust grism parameters (e.g., tilt, dispersion), enter numeric values in the corresponding fields and press **Update**.

**16. Save to .mdp**

* Click **Save** and use the default `.mdp` format.
* Enter the desired filename and confirm to save the current layout.

**17. Save to .sbr**

* Change file type to **.sbr** in the save dialog.
* Click **Save** and confirm filename and FOV center (auto-filled from current settings).
* Header info includes the original `.mdp` file name and current center coordinates.

"""

import os.path
import copy

import numpy as np
from qtpy.QtWidgets import (
    QFileDialog, QInputDialog, QVBoxLayout, QLabel, QScrollArea, QWidget,
    QPushButton, QDialog, QComboBox, QLineEdit, QCheckBox, QMessageBox
)

from ginga.gw import Widgets
from ginga import GingaPlugin
from ginga.canvas.CanvasObject import get_canvas_types

# local imports
from naoj.moircs.moircs_fov import MOIRCS_FOV
from naoj.moircs.grism_info import grism_info_map


class MOIRCS_Mask_Builder(GingaPlugin.LocalPlugin):
    def __init__(self, fv, fitsimage):
        super().__init__(fv, fitsimage)

        width, height = self.fitsimage.get_data_size()
        x_center = width / 2
        y_center = height / 2
        pt_center = (x_center, y_center)

        prefs = self.fv.get_preferences()
        self.settings = prefs.create_category('plugin_MOIRCS_Mask_Builder')
        self.settings.add_defaults(display_slitID=True, grism='zJ500')
        self.settings.load(onError='silent')

        self.grismtypes = ('zJ500', 'HK500', 'LS_J', 'LS_H', 'VB_K', 'VPH-Y')
        self.grism_info_map = grism_info_map
        default_grism = self.settings.get('grism', 'zJ500')
        self.grism_info = dict(self.grism_info_map.get(default_grism, {}))

        self.shapes = []  # Unified list for slits and holes
        self.spinboxes = {}
        self.spinbox_scales = {}
        self._undo_stack = []

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

        p_canvas = self.fitsimage.get_canvas()
        if not p_canvas.has_tag('maskbuilder-canvas'):
            p_canvas.add(self.canvas, tag='maskbuilder-canvas')

        self.fov_center = pt_center
        self.fov_overlay = None

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
        hbox_fov.add_widget(Widgets.Label("MOIRCS FOV:"), stretch=0)

        self.cb_ch1 = Widgets.CheckBox("CH1")
        self.cb_ch1.set_state(True)
        self.cb_ch1.add_callback('activated', lambda w, state: self.on_fov_changed(w, state))
        hbox_fov.add_widget(self.cb_ch1, stretch=0)

        self.cb_ch2 = Widgets.CheckBox("CH2")
        self.cb_ch2.set_state(True)
        self.cb_ch2.add_callback('activated', lambda w, state: self.on_fov_changed(w, state))
        hbox_fov.add_widget(self.cb_ch2, stretch=0)

        fov_controls.add_widget(hbox_fov, stretch=0)

        hbox_center = Widgets.HBox()
        hbox_center.set_spacing(4)
        hbox_center.add_widget(Widgets.Label("FOV Center X:"), stretch=0)
        self.w.fov_center_x = Widgets.SpinBox()
        self.w.fov_center_x.set_limits(0, 5000, 1)
        self.w.fov_center_x.set_value(1084)
        hbox_center.add_widget(self.w.fov_center_x, stretch=0)

        hbox_center.add_widget(Widgets.Label("Y:"), stretch=0)
        self.w.fov_center_y = Widgets.SpinBox()
        self.w.fov_center_y.set_limits(0, 5000, 1)
        self.w.fov_center_y.set_value(1786)
        hbox_center.add_widget(self.w.fov_center_y, stretch=0)

        # Update button
        btn_update = Widgets.Button("Update")
        btn_update.add_callback('activated', self.set_fov_center_from_user_input)
        hbox_center.add_widget(btn_update, stretch=0)
        fov_controls.add_widget(hbox_center, stretch=0)

        fr.set_widget(fov_controls)
        vbox.add_widget(fr, stretch=0)

        # --- Frame for Slit and Hole Controls ---
        fr_slit = Widgets.Frame("Slit and Hole Controls")
        vbox_slit = Widgets.VBox()
        vbox_slit.set_spacing(6)

        # Display Options (Slit/Hole ID, Comments, Show Excluded)
        hbox_sh_display = Widgets.HBox()
        hbox_sh_display.set_spacing(4)

        label_sh_display = Widgets.Label("Display Options:")
        hbox_sh_display.add_widget(label_sh_display, stretch=0)

        self.w.display_slit_id = Widgets.CheckBox("Slit/Hole ID")
        self.w.display_slit_id.add_callback('activated', lambda *args: self.draw_slits())
        hbox_sh_display.add_widget(self.w.display_slit_id, stretch=0)

        self.w.display_comments = Widgets.CheckBox("Comments")
        self.w.display_comments.add_callback('activated', lambda *args: self.draw_slits())
        hbox_sh_display.add_widget(self.w.display_comments, stretch=0)

        self.w.show_excluded = Widgets.CheckBox("Excluded")
        self.w.show_excluded.add_callback('activated', lambda w, val: self.toggle_show_excluded(val))
        hbox_sh_display.add_widget(self.w.show_excluded, stretch=0)

        vbox_slit.add_widget(hbox_sh_display, stretch=0)

        # Row 1: Show Slit List + Auto Detection
        hbox_view_auto = Widgets.HBox()
        hbox_view_auto.set_spacing(4)

        btn_view_params = Widgets.Button("Show Slit List")
        btn_view_params.add_callback('activated', lambda w: self.show_slit_and_hole_info())

        btn_auto = Widgets.Button("Auto Detection")
        btn_auto.add_callback('activated', lambda w: self.auto_detect_overlaps())

        hbox_view_auto.add_widget(btn_view_params, stretch=0)
        hbox_view_auto.add_widget(btn_auto, stretch=0)
        vbox_slit.add_widget(hbox_view_auto, stretch=0)

        # Row 2: Add + Edit + Undo
        hbox_add_edit = Widgets.HBox()
        hbox_add_edit.set_spacing(4)
        btn_add = Widgets.Button("Add")
        btn_add.add_callback('activated', lambda w: self.add_slit_or_hole())
        btn_edit = Widgets.Button("Edit")
        btn_edit.add_callback('activated', lambda w: self.edit_slit_or_hole())
        btn_undo = Widgets.Button("Undo")
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
        label_display = Widgets.Label("Display Options:")
        hbox_display.add_widget(label_display, stretch=0)

        self.w.display_spectra = Widgets.CheckBox("Spectra")
        self.w.display_spectra.add_callback('activated', lambda *args: self.draw_spectra())
        hbox_display.add_widget(self.w.display_spectra, stretch=0)

        vbox_controls.add_widget(hbox_display, stretch=0)

        # Spectra Dashed Line Interval Dropdown
        hbox_dashline = Widgets.HBox()
        hbox_dashline.set_spacing(6)
        hbox_dashline.add_widget(Widgets.Label("Tick Marks (pixels):"), stretch=0)

        self.w.dash_interval = Widgets.ComboBox()
        for val in ['none(default)','50', '100', '150', '200', '250', '300']:
            self.w.dash_interval.append_text(val)
        self.w.dash_interval.set_index(0)

        self.w.dash_interval.add_callback('activated', lambda w, idx: (
        self.show_dashline_change_warning() if self.w.dash_interval.get_text() in {'50','100', '150', '200', '250','300'} else None,
        self.redraw_spectra()
))

        hbox_dashline.add_widget(self.w.dash_interval, stretch=0)
        vbox_controls.add_widget(hbox_dashline, stretch=0)

        # Grism selection
        hbox_grism = Widgets.HBox()
        hbox_grism.set_spacing(6)
        hbox_grism.add_widget(Widgets.Label("Grism:"), stretch=0)

        self.w.grism = Widgets.ComboBox()
        for name in self.grismtypes:
            self.w.grism.append_text(name)
        self.w.grism.set_index(self.grismtypes.index(self.settings.get('grism')))
        self.w.grism.add_callback('activated', lambda w, idx: self.set_grism())
        hbox_grism.add_widget(self.w.grism, stretch=0)

        vbox_controls.add_widget(hbox_grism, stretch=0)

        # Float parameter input using TextEntries
        param_fields = (
            "directwave", "wavestart", "waveend", "dispersion",
            "zero_offset", "dx1", "dx2", "tilt1", "tilt2"
        )
        labels = {
            "directwave": "Direct Wave:",
            "wavestart": "Wave Start:",
            "waveend": "Wave End:",
            "dispersion": "Dispersion:",
            "zero_offset": "Zero Offset:",
            "dx1": "DX1:",
            "dx2": "DX2:",
            "tilt1": "Tilt 1:",
            "tilt2": "Tilt 2:",
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

        # --- Bottom Buttons (Load/Save/etc.) ---
        btns = Widgets.HBox()
        btns.set_spacing(3)

        btn_close = Widgets.Button("Close")
        btn_close.add_callback('activated', lambda w: self.close())
        btns.add_widget(btn_close, stretch=0)

        btn_help = Widgets.Button("Help")
        btn_help.add_callback('activated', lambda w: self.help())
        btns.add_widget(btn_help, stretch=0)

        btns.add_widget(Widgets.Label(''), stretch=1)

        self.w.filepath = Widgets.TextEntry()
        btn_browse = Widgets.Button("Browse")
        btn_browse.add_callback('activated', self.browse_file)
        btn_load = Widgets.Button("Load")
        btn_load.add_callback('activated', lambda w: self.load_file(self.w.filepath.get_text()))

        self.w.save_format = Widgets.ComboBox()
        self.w.save_format.append_text(".mdp")
        self.w.save_format.append_text(".sbr")
        self.w.save_format.set_index(0)  # default to .mdp

        btn_save = Widgets.Button("Save")
        btn_save.add_callback('activated', lambda w: self.save_file())

        btns.add_widget(self.w.filepath, stretch=5)
        btns.add_widget(btn_browse, stretch=0)
        btns.add_widget(btn_load, stretch=0)
        btns.add_widget(self.w.save_format, stretch=0)
        btns.add_widget(btn_save, stretch=0)

        # Add to main container
        top.add_widget(sw, stretch=1)
        top.add_widget(btns, stretch=0)
        container.add_widget(top, stretch=1)
        self.on_fov_changed()

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
        x = self.w.fov_center_x.get_value()
        y = self.w.fov_center_y.get_value()
        self.fov_center = (x, y)  # Update internal state
        if hasattr(self, 'fov_overlay') and self.fov_overlay:
            self.fov_overlay.set_pos((x, y))
            self.canvas.redraw(whence=0)
            self.draw_slits()  # Redraw slits to reflect new FOV center
            self.draw_spectra()  # Redraw spectra to reflect new FOV center
            self.logger.info(f"FOV center updated to: ({x:.1f}, {y:.1f})")
        else:
            self.logger.warning("FOV overlay not active; initializing.")
            self.on_fov_changed()  # Trigger FOV overlay update

    def show_fov_overlay(self, ch1, ch2):
        width, height = self.fitsimage.get_data_size()
        if width <= 0 or height <= 0:
            self.logger.warning(f"No image loaded: width={width}, height={height}")
            self.cb_ch1.set_state(False)
            self.cb_ch2.set_state(False)
            self.remove_fov_overlay()
            self.canvas.redraw(whence=0)
            return

        try:
            pt_center = (self.w.fov_center_x.get_value(), self.w.fov_center_y.get_value())
        except AttributeError as e:
            self.logger.warning(f"Error accessing FOV center widgets: {e}. Using fallback center.")
            pt_center = self.fov_center

        self.fov_center = pt_center  # Update internal state

        try:
            if not hasattr(self, 'fov_overlay') or self.fov_overlay is None:
                self.logger.info("Creating new FOV overlay")
                self.remove_fov_overlay()
                self.fov_overlay = MOIRCS_FOV(self.canvas, pt_center)
                self.fov_overlay.scale_to_image(width, height)
            else:
                self.fov_overlay.set_pos(pt_center)

            # Safely remove detector groups if they exist on canvas
            for group in [self.fov_overlay.det1_group, self.fov_overlay.det2_group, self.fov_overlay.fov_base]:
                if group and group in self.canvas.objects:
                    try:
                        self.canvas.delete_object(group)
                        self.logger.debug(f"Removed group from canvas")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove group: {e}")

            # Add groups based on ch1 and ch2
            if ch1 and self.fov_overlay.det1_group:
                self.canvas.add(self.fov_overlay.det1_group)
                self.logger.debug("Added det1_group to canvas")
            if ch2 and self.fov_overlay.det2_group:
                self.canvas.add(self.fov_overlay.det2_group)
                self.logger.debug("Added det2_group to canvas")

            # Ensure fov_base is always present
            if not self.fov_overlay.fov_base:
                self.logger.warning("fov_base is None; rebuilding overlay")
                self.fov_overlay.rebuild()
            else:
                self.canvas.add(self.fov_overlay.fov_base)
                self.logger.debug("Added fov_base to canvas")
            self.canvas.redraw(whence=0)
        except Exception as e:
            self.logger.error(f"Error updating FOV overlay: {e}")
            self.remove_fov_overlay()
            self.canvas.redraw(whence=0)

    def on_fov_changed(self, w=None, state=None):
        if not hasattr(self, 'cb_ch1') or not hasattr(self, 'cb_ch2'):
            self.logger.warning("Checkboxes not initialized; skipping FOV update.")
            return
        try:
            ch1 = self.cb_ch1.get_state()
            ch2 = self.cb_ch2.get_state()
            self.logger.info(f"FOV toggle triggered: CH1={ch1}, CH2={ch2}, Widget={w}, State={state}")
            if ch1 or ch2:
                self.show_fov_overlay(ch1, ch2)
            else:
                self.remove_fov_overlay()
                self.logger.info("Both channels unchecked; FOV overlay removed.")
            self.draw_slits()  # Explicitly redraw slits
            self.draw_spectra()  # Explicitly redraw spectra
            self.canvas.redraw(whence=0)
        except Exception as e:
            self.logger.error(f"Error in on_fov_changed: {e}")
            self.remove_fov_overlay()
            self.draw_slits()
            self.draw_spectra()
            self.canvas.redraw(whence=0)

    def remove_fov_overlay(self):
        if hasattr(self, 'fov_overlay') and self.fov_overlay:
            try:
                self.fov_overlay.remove()  # Calls MOIRCS_FOV.remove() to clean up all groups
                self.logger.info("FOV overlay removed successfully.")
            except Exception as e:
                self.logger.error(f"Error removing FOV overlay: {e}")
            self.fov_overlay = None
        self.canvas.redraw(whence=0)

    def browse_file(self, *args):
        file_path, _ = QFileDialog.getOpenFileName(
            parent=None,
            caption="Select MDP file",
            filter="MDP Files (*.mdp);;All Files (*)"
        )
        if file_path:
            self.w.filepath.set_text(file_path)

    def load_file(self, filepath):
        if isinstance(filepath, tuple):
            filepath = filepath[0]
        if filepath and os.path.exists(filepath):
            self.mdp_filename = filepath
            self.load_mdp(filepath)
            self.draw_slits()
            self.draw_spectra()

    def load_mdp(self, filepath):
        self.shapes.clear()
        img_h = self.fitsimage.get_data_size()[1]

        rows = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                row_dict = {
                    'type': parts[6].strip(),
                    'x': float(parts[0]),
                    'y': float(parts[1]),
                    'width': float(parts[2]),
                    'length': float(parts[3]),
                    'angle': float(parts[4]),
                    'priority': parts[5],
                    'comment': " ".join(parts[7:]) if len(parts) > 7 else ''
                }
                if row_dict['type'].startswith('C'):
                    row_dict['diameter'] = row_dict.pop('width')
                    row_dict.pop('length')
                    row_dict.pop('angle')
                rows.append(row_dict)

        self.shapes = rows

    def show_slit_and_hole_info(self):
        dialog = QDialog()
        dialog.setWindowTitle("Slit and Hole Manager")
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        checkbox_refs = []

        for i, shape in enumerate(self.shapes):
            shape_type = 'Slit' if shape['type'].startswith('B') else 'Hole'
            comment = shape.get('comment', '')
            label = f"{shape_type} #{i} | x={shape['x']:.1f}, y={shape['y']:.1f} | {comment}"

            cb = QCheckBox(label)
            # Checked = included; Unchecked = either _deleted or _excluded
            cb.setChecked(not shape.get('_deleted', False) and not shape.get('_excluded', False))

            def make_callback(index, checkbox):
                def callback(state):
                    checked = checkbox.isChecked()
                    self.shapes[index]['_deleted'] = not checked
                    self.shapes[index]['_excluded'] = not checked
                    self.draw_slits()
                    self.draw_spectra()
                return callback

            cb.stateChanged.connect(make_callback(i, cb))
            checkbox_refs.append(cb)
            scroll_layout.addWidget(cb)

        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.setLayout(layout)
        dialog.exec_()

    def toggle_show_excluded(self, val):
        self.show_excluded = val
        self.draw_slits()
        self.draw_spectra()

    def auto_detect_overlaps(self):
        if not self.shapes:
            QMessageBox.information(None, "Info", "No shapes to analyze.")
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

        self.draw_slits()
        self.draw_spectra()
        QMessageBox.information(None, "Auto Detection", f"Excluded {excluded_count} shape(s).")

    def add_slit_or_hole(self):
        self._undo_stack.append({'shapes': copy.deepcopy(self.shapes)})
        dialog = QDialog()
        dialog.setWindowTitle("Add Slit or Hole")
        layout = QVBoxLayout()
        combo = QComboBox()
        combo.addItems(["Slit (Rectangle)", "Hole (Circle)"])
        layout.addWidget(QLabel("Select shape to add:"))
        layout.addWidget(combo)
        btn_ok = QPushButton("OK")
        layout.addWidget(btn_ok)
        dialog.setLayout(layout)

        def on_ok():
            choice = combo.currentText()
            dialog.accept()
            self._add_shape_type = 'slit' if choice.startswith("Slit") else 'hole'
            p_canvas = self.fitsimage.get_canvas()
            if not p_canvas.has_object(self.canvas):
                p_canvas.add(self.canvas, tag='maskbuilder-canvas')
            self.canvas.set_drawtype('point')
            self.canvas.set_draw_mode('draw')
            self.canvas.set_callback('button-press', self._on_click_event)
            self.canvas.ui_set_active(True, viewer=self.fitsimage)

        btn_ok.clicked.connect(on_ok)
        dialog.exec_()

    def is_within_fov_bounds(self, x, y):
        """Check if (x, y) is within MOIRCS rectangle in x, and circle radius in y."""
        if not hasattr(self, 'fov_overlay') or self.fov_overlay is None:
            self.logger.warning("FOV overlay not initialized; assuming position is valid.")
            return True

        x_center, y_center = self.fov_center  # Use self.fov_center for consistency
        fov = self.fov_overlay
        pixscale = fov.pixscale  # deg/pixel
        xr = (fov.moircs_fov[0] * 0.5) / pixscale  # Rectangle half-width in pixels
        radius_pix = fov.circle_radius_deg / pixscale  # Circle radius in pixels

        within_x = (x_center - xr) <= x <= (x_center + xr)
        r = np.hypot(x - x_center, y - y_center)
        within_radius = r <= radius_pix
        return within_x and within_radius

    def is_within_y_arcsec_limit(self, y, min_arcsec_from_center=10):
        """Ensure the slit/hole is at least ±min_arcsec_from_center from the centerline."""
        if not hasattr(self, 'fov_center') or not hasattr(self, 'fov_overlay'):
            self.logger.warning("FOV center or overlay not initialized; assuming position is valid.")
            return True

        y_center = self.fov_center[1]
        min_pixel_dist = min_arcsec_from_center / (self.fov_overlay.pixscale * 3600)
        return abs(y - y_center) >= min_pixel_dist

    def _on_click_event(self, canvas, button, data_x, data_y):
        self.canvas.set_draw_mode(None)
        self.canvas.remove_callback('button-press', self._on_click_event)
        self.canvas.ui_set_active(False, viewer=self.fitsimage)

        try:
            samplefac = self.common_info.samplefac
            bin_x, bin_y = self.common_info.bin
        except AttributeError:
            samplefac = 1.0
            bin_x, bin_y = 1, 1
        try:
            xoffset = self.xoffset or 0
            yoffset = self.yoffset or 0
        except AttributeError:
            xoffset, yoffset = 0, 0

        x = data_x * bin_x * samplefac + xoffset
        y = data_y * bin_y * samplefac + yoffset

        out_of_bounds = not self.is_within_fov_bounds(x, y) or not self.is_within_y_arcsec_limit(y)
        if out_of_bounds:
            QMessageBox.warning(None, "Out of Bounds", "The selected position is outside the allowed FOV, but it will be added.")

        dialog = QDialog()
        dialog.setWindowTitle("Confirm New Shape")
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Add new {self._add_shape_type} at x={x:.1f}, y={y:.1f}?"))
        comment_field = QLineEdit()
        layout.addWidget(QLabel("Comment:"))
        layout.addWidget(comment_field)

        def on_confirm():
            comment = comment_field.text()
            shape = {'x': x, 'y': y, 'comment': comment}
            if out_of_bounds:
                shape['_excluded'] = True  # Initially excluded from auto detection/export
            if self._add_shape_type == 'slit':
                shape.update({'type': 'B', 'width': 100, 'length': 7, 'angle': 0, 'priority': '1'})
            else:
                shape.update({'type': 'C', 'diameter': 30})
            self.shapes.append(shape)
            self.draw_slits()
            self.draw_spectra()
            dialog.accept()

        btn_confirm = QPushButton("Confirm")
        btn_confirm.clicked.connect(on_confirm)
        layout.addWidget(btn_confirm)
        dialog.setLayout(layout)
        dialog.exec_()

    def edit_slit_or_hole(self):
        dialog = QDialog()
        dialog.setWindowTitle("Edit Slit or Hole")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select ID to edit:"))
        combo = QComboBox()
        id_map = {}
        for i, shape in enumerate(self.shapes):
            if shape.get('_deleted'):
                continue
            prefix = 'B' if shape['type'].startswith('B') else 'C'
            label = f"{prefix}{i}: {shape.get('comment', '')}"
            combo.addItem(label)
            id_map[combo.count() - 1] = shape
        layout.addWidget(combo)
        fields_widget = QWidget()
        fields_layout = QVBoxLayout()
        fields_widget.setLayout(fields_layout)
        layout.addWidget(fields_widget)
        current_fields = {}

        def clear_fields():
            while fields_layout.count():
                item = fields_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            current_fields.clear()

        def add_field(name, initial_value):
            lbl = QLabel(name)
            le = QLineEdit()
            le.setText(str(initial_value))
            fields_layout.addWidget(lbl)
            fields_layout.addWidget(le)
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

        combo.currentIndexChanged.connect(populate_fields)
        populate_fields(0)

        def apply_changes():
            shape = id_map[combo.currentIndex()]
            try:
                x = float(current_fields["X:"].text())
                y = float(current_fields["Y:"].text())

                if not self.is_within_fov_bounds(x, y) or not self.is_within_y_arcsec_limit(y):
                    QMessageBox.warning(dialog, "Out of Bounds", "The specified position is outside the allowed FOV.")
                    return

                if shape['type'].startswith('B'):
                    width = float(current_fields["Width:"].text())
                    length = float(current_fields["Length:"].text())
                    angle = float(current_fields["Angle:"].text())
                    if width < 35:
                        QMessageBox.warning(dialog, "Invalid input", "Width must be at least 35.")
                        return
                    if length < 6.8:
                        QMessageBox.warning(dialog, "Invalid input", "Length must be at least 6.8.")
                        return
                else:
                    diameter = float(current_fields["Diameter:"].text())
                    if diameter < 20 or diameter > 30:
                        QMessageBox.warning(dialog, "Invalid input", "Diameter must be between 20 and 30.")
                        return

                self._undo_stack.append({'shapes': copy.deepcopy(self.shapes)})
                shape['x'] = x
                shape['y'] = y
                shape['comment'] = current_fields["Comment:"].text()
                if shape['type'].startswith('B'):
                    shape['width'] = width
                    shape['length'] = length
                    shape['angle'] = angle
                else:
                    shape['diameter'] = diameter
                self.draw_slits()
                self.draw_spectra()
                dialog.accept()
            except ValueError:
                QMessageBox.warning(dialog, "Invalid input", "Please enter valid numeric values.")

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(apply_changes)
        layout.addWidget(btn_apply)
        dialog.setLayout(layout)
        dialog.exec_()

    def undo_last_edit(self):
        if not self._undo_stack:
            print("Nothing to undo")
            return
        last_state = self._undo_stack.pop()
        self.shapes = last_state['shapes']
        self.draw_slits()
        self.draw_spectra()
        print("Undo performed")

    def on_grism_param_changed(self, key):
        if getattr(self, '_updating_grism_params', False):
            return
        val = self.get_entry_value(key)
        self.grism_info[key] = val

    def set_grism(self):
        grism_name = self.w['grism'].get_text()
        self.settings.set(dict(grism=grism_name))
        self.grism_info = dict(self.grism_info_map.get(grism_name, {}))

        self._updating_grism_params = True
        for key, val in self.grism_info.items():
            self.set_entry_value(key, val)
        self._updating_grism_params = False

        self.draw_spectra()

    def update_all_grism_params(self):
        for key in self.textentries:
            self.grism_info[key] = self.get_entry_value(key)
        self.redraw_spectra()

    def reset_grism_params(self):
        grism_name = self.w['grism'].get_text()
        original_info = self.grism_info_map.get(grism_name, {})

        self._updating_grism_params = True
        for key in self.textentries:
            val = original_info.get(key, 0.0)
            self.set_entry_value(key, val)
            self.grism_info[key] = val
        self._updating_grism_params = False

        self.redraw_spectra()

    def draw_slits(self):
        self.canvas.enable_draw(False)

        # Clear all slit, hole, and label objects
        for obj in list(self.canvas.objects):
            if hasattr(obj, 'tag') and isinstance(obj.tag, str) and (
                obj.tag.startswith("slit") or obj.tag.startswith("hole") or
                obj.tag.startswith("label") or obj.tag.startswith("label_comment")):
                try:
                    self.canvas.delete_object_by_tag(obj.tag)
                except Exception as e:
                    self.logger.warning(f"Failed to delete object with tag {obj.tag}: {e}")

        show_excluded = getattr(self, 'show_excluded', False)
        show_ids = self.w.display_slit_id.get_state()
        show_comments = self.w.display_comments.get_state()
        samplefac = 1.0
        bin_x, bin_y = 1, 1
        xoffset, yoffset = getattr(self, 'xoffset', 0), getattr(self, 'yoffset', 0)

        # Use FOV center for channel assignment
        y_center = self.fov_center[1] / bin_y / samplefac

        drawn_shapes = 0
        skipped_shapes = 0

        for i, shape in enumerate(self.shapes):
            if shape.get('_deleted') or (shape.get('_excluded') and not show_excluded):
                skipped_shapes += 1
                continue

            x, y = shape['x'], shape['y']
            comment = shape.get('comment', '')

            # Convert image coords to canvas coords
            xcen = (x - xoffset) / bin_x / samplefac
            ycen = (y - yoffset) / bin_y / samplefac

            # Assign shape to CH1 (ycen <= y_center) or CH2 (ycen > y_center)
            is_ch1 = ycen <= y_center
            is_ch2 = ycen > y_center

            # Skip if the shape's channel is unchecked
            if is_ch1 and not self.cb_ch1.get_state():
                skipped_shapes += 1
                continue
            if is_ch2 and not self.cb_ch2.get_state():
                skipped_shapes += 1
                continue

            # Draw slit (rectangle) or hole (circle)
            if shape['type'].startswith('B'):
                w = shape.get('width', 100.0) / bin_x / samplefac
                l = shape.get('length', 7.0) / bin_y / samplefac
                angle = shape.get('angle', 0)
                rect = self.dc.Rectangle(
                    xcen - w / 2, ycen - l / 2,
                    xcen + w / 2, ycen + l / 2,
                    rotation_deg=angle,
                    color='purple' if shape.get('_excluded') else 'white',
                    linewidth=1
                )
                rect.coord = 'data'
                self.canvas.add(rect, tag=f"slit{i}")
                if show_ids:
                    self.canvas.add(
                        self.dc.Text(xcen, ycen + l / 2 + 10 / samplefac, text=f"{i}", color='white', fontsize=11),
                        tag=f"label{i}"
                    )
                if show_comments and comment:
                    comment_text = self.dc.Text(xcen, ycen - l / 2 - 30 / samplefac, text=comment, color='white')
                    comment_text.coord = 'data'
                    self.canvas.add(comment_text, tag=f"label_comment{i}")

            elif shape['type'].startswith('C'):
                diameter = shape.get('diameter', 30.0) / samplefac
                radius = diameter / 2
                circle = self.dc.Circle(xcen, ycen, radius,
                                        color='purple' if shape.get('_excluded') else 'yellow', linewidth=1)
                circle.coord = 'data'
                self.canvas.add(circle, tag=f"hole{i}")
                if show_ids:
                    self.canvas.add(
                        self.dc.Text(xcen, ycen + radius + 10 / samplefac, text=f"{i}", color='yellow', fontsize=11),
                        tag=f"label_hole{i}"
                    )
                if show_comments and comment:
                    comment_text = self.dc.Text(xcen, ycen - radius - 30 / samplefac, text=comment, color='yellow')
                    comment_text.coord = 'data'
                    self.canvas.add(comment_text, tag=f"label_comment_hole{i}")

            drawn_shapes += 1

        self.canvas.enable_draw(True)
        self.canvas.redraw(whence=0)

    def draw_spectra(self):
        # Check toggle state first
        if not self.w.display_spectra.get_state():
            # Clean up any previous spectra-related graphics
            for obj in list(self.canvas.objects):
                if hasattr(obj, 'tag') and isinstance(obj.tag, str):
                    if obj.tag.startswith(("dashline_", "dashline_bundle", "spectrum", "footprint", "spectra_bundle")):
                        try:
                            self.canvas.delete_object_by_tag(obj.tag)
                        except Exception:
                            continue
            self.fitsimage.redraw()
            return

        CompoundObject = get_canvas_types().CompoundObject

        # Clean up previously drawn spectra-related objects
        for obj in list(self.canvas.objects):
            if hasattr(obj, 'tag') and isinstance(obj.tag, str):
                if obj.tag.startswith(("dashline_", "dashline_bundle", "spectrum", "footprint", "spectra_bundle")):
                    try:
                        self.canvas.delete_object_by_tag(obj.tag)
                    except Exception:
                        continue

        g = self.grism_info
        if not g:
            self.fitsimage.redraw()
            return

        samplefac = 1.0
        bin_x, bin_y = 1, 1
        xoffset = getattr(self, "xoffset", 0)
        yoffset = getattr(self, "yoffset", 0)

        y_center = self.fov_center[1] / bin_y / samplefac
        direct_wave = g.get('directwave', 0)
        wave_start = g.get('wavestart', 0)
        wave_end = g.get('waveend', 0)
        dispersion = g.get('dispersion', 1)

        if dispersion == 0:
            self.logger.error("Invalid dispersion: 0")
            self.fitsimage.redraw()
            return

        tilt = (g.get('tilt1', 0) + g.get('tilt2', 0)) / 2
        bottom_length = (wave_start - direct_wave) / dispersion / bin_y / samplefac
        top_length = (direct_wave - wave_end) / dispersion / bin_y / samplefac

        objects_to_draw = []
        dash_groups = []

        # --- Efficient "center-outward" dashed line drawing ---

        try:
            dash_text = (self.w.dash_interval.get_text() or "").strip().lower()
            valid_intervals = {'50', '100', '150', '200', '250', '300'}
            if dash_text in valid_intervals:
                dash_interval = int(dash_text)
                interval_y = dash_interval / bin_y / samplefac

                for i, shape in enumerate(self.shapes):
                    if shape.get('_deleted') or (shape.get('_excluded') and not getattr(self, 'show_excluded', False)):
                        continue

                    x = (shape['x'] - xoffset) / bin_x / samplefac
                    y = (shape['y'] - yoffset) / bin_y / samplefac
                    if (y <= y_center and not self.cb_ch1.get_state()) or (y > y_center and not self.cb_ch2.get_state()):
                        continue

                    width = shape.get('width', 100.0) if shape['type'].startswith('B') else shape.get('diameter', 30.0)
                    width /= bin_x * samplefac

                    if y > y_center:
                        spec_y1 = y - top_length
                        spec_y2 = y + bottom_length
                        color = 'red'
                    else:
                        spec_y1 = y + top_length
                        spec_y2 = y - bottom_length
                        color = 'green'

                    # Clamp direction
                    ymin, ymax = sorted([spec_y1, spec_y2])
                    x_start = x - width / 2
                    x_end = x + width / 2

                    # Generate lines from center outwards
                    dash_lines = []
                    for direction in [-1, 1]:  # up and down
                        offset = 0.0
                        while True:
                            y_pos = y + direction * offset
                            if y_pos < ymin or y_pos > ymax:
                                break
                            line = self.dc.Line(
                                x_start, y_pos,
                                x_end, y_pos,
                                color=color,
                                linewidth=0.5,
                                coord='data'
                            )
                            dash_lines.append(line)
                            offset += interval_y

                    if dash_lines:
                        group = CompoundObject(*dash_lines)
                        group.tag = f"dashline_bundle_{i}"
                        dash_groups.append(group)

        except Exception as e:
            self.logger.warning(f"Dash line rendering skipped: {e}")

        if dash_groups:
            self.canvas.add(CompoundObject(*dash_groups), tag="dashline_master")

        # --- Spectral Rectangles ---
        for i, shape in enumerate(self.shapes):
            if shape.get('_deleted') or (shape.get('_excluded') and not getattr(self, 'show_excluded', False)):
                continue

            x, y = shape['x'], shape['y']
            xcen = (x - xoffset) / bin_x / samplefac
            ycen = (y - yoffset) / bin_y / samplefac
            if (ycen <= y_center and not self.cb_ch1.get_state()) or (ycen > y_center and not self.cb_ch2.get_state()):
                continue

            width = shape.get('width', 100.0) if shape['type'].startswith('B') else shape.get('diameter', 30.0)
            width /= bin_x * samplefac

            if ycen > y_center:
                spec_y1 = ycen - top_length
                spec_y2 = ycen + bottom_length
                color = 'red'
            else:
                spec_y1 = ycen + top_length
                spec_y2 = ycen - bottom_length
                color = 'green'

            rect = self.dc.Rectangle(
                xcen - width / 2, spec_y1,
                xcen + width / 2, spec_y2,
                rotation_deg=tilt,
                color=color,
                linewidth=1,
                fill=False
            )
            rect.coord = 'data'
            rect.tag = f"spectrum_{'slit' if shape['type'].startswith('B') else 'hole'}_{i}"
            objects_to_draw.append(rect)

        if objects_to_draw:
            self.canvas.add(CompoundObject(*objects_to_draw), tag="spectra_bundle")

        self.fitsimage.redraw()

    def redraw_spectra(self):
        self.draw_spectra()

    def show_dashline_change_warning(self):
        QMessageBox.warning(None,  # or self.fv.w.root if available
            "Spectral Dash Line Notice",
            "NOTE (!) Spectral dashed line rendering is under development.\n\n"
            "For stability, it is recommended to reset the interval to the default \n"
            "before using other functions."
)


    def save_file(self):
        format_choice = self.w.save_format.get_text()

        if format_choice == '.mdp':
            self.save_as_mdp()
        elif format_choice == '.sbr':
            self.save_as_sbr()

    def save_as_mdp(self):
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save .mdp File", filter="MDP files (*.mdp)"
        )
        if not filename:
            return
        img_h = self.fitsimage.get_data_size()[1]
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

    def save_as_sbr(self):
        filename, _ = QFileDialog.getSaveFileName(
            caption="Save .sbr File", filter="SBR files (*.sbr)"
        )
        if not filename:
            return
        pixscale = getattr(self, 'pixscale', 0.117000)
        beta = getattr(self, 'beta', 0.29898169)
        mos_rot = getattr(self, 'mos_rot', 0.0)
        offset = np.deg2rad(mos_rot)
        conversion = 0.015 / beta / 0.1038 * pixscale
        default_xFOVCenter = self.w.fov_center_x.get_value() if hasattr(self.w, 'fov_center_x') else 1084
        default_yFOVCenter = self.w.fov_center_y.get_value() if hasattr(self.w, 'fov_center_y') else 1786
        xFOVCenter, ok1 = QInputDialog.getDouble(None, "FOV Center", "X (pixels):", default_xFOVCenter, decimals=2)
        if not ok1:
            return
        yFOVCenter, ok2 = QInputDialog.getDouble(None, "FOV Center", "Y (pixels):", default_yFOVCenter, decimals=2)
        if not ok2:
            return
        self.xFOVCenter = xFOVCenter
        self.yFOVCenter = yFOVCenter
        mdp_filename = getattr(self, 'mdp_filename', 'UNKNOWN_MDP')
        image_name = getattr(self, 'image_name', 'UNKNOWN_IMAGE')
        try:
            with open(filename, 'w') as f:
                f.write(f"# mdp: {mdp_filename}\n")
                f.write(f"# Image: {image_name}\n")
                f.write(f"# FOV Center: x={xFOVCenter:.2f}, y={yFOVCenter:.2f}\n")
                shapes_filtered = [s for s in self.shapes if not s.get('_deleted')]
                for i, shape in enumerate(shapes_filtered):
                    x = shape['x']
                    y = shape['y']
                    sl_l = shape['width'] * 0.5 if shape['type'].startswith('B') else shape['diameter'] * 0.5
                    x1_off = x - sl_l - xFOVCenter
                    x2_off = x + sl_l - xFOVCenter
                    y1_off = y - yFOVCenter
                    y2_off = y - yFOVCenter
                    x1_focus = -x1_off * conversion
                    x2_focus = -x2_off * conversion
                    y1_focus = y1_off * conversion
                    y2_focus = y2_off * conversion
                    x1_laser = x1_focus * 1.006
                    x2_laser = x2_focus * 1.006
                    y1_laser = y1_focus * 1.006
                    y2_laser = y2_focus * 1.006
                    if mos_rot != 0:
                        r1 = np.hypot(x1_laser, y1_laser)
                        theta1 = np.arctan2(y1_laser, x1_laser)
                        x1_laser = r1 * np.cos(theta1 + offset)
                        y1_laser = r1 * np.sin(theta1 + offset)
                        r2 = np.hypot(x2_laser, y2_laser)
                        theta2 = np.arctan2(y2_laser, x2_laser)
                        x2_laser = r2 * np.cos(theta2 + offset)
                        y2_laser = r2 * np.sin(theta2 + offset)
                    corners_r = np.array([
                        np.hypot(x1_focus, y1_focus),
                        np.hypot(x1_focus, y2_focus),
                        np.hypot(x2_focus, y1_focus),
                        np.hypot(x2_focus, y2_focus)
                    ])
                    if np.any(corners_r > 90):
                        QMessageBox.warning(None, "Warning", f"{'Slit' if shape['type'].startswith('B') else 'Hole'} {i} is out of laser FOV.")
                        continue
                    if shape['type'].startswith('B') and np.any(np.abs([x1_focus, x2_focus]) > 60):
                        QMessageBox.warning(None, "Warning", f"Slit {i} is out of MOIRCS FOV.")
                        continue
                    if shape['type'].startswith('B'):
                        width = (shape['length'] * pixscale / 2.06218 * 1.006) * 1.08826 - 0.126902
                        f.write(f"B,{x1_laser:9.4f},{y1_laser:9.4f},{x2_laser:9.4f},{y2_laser:9.4f},{width:9.4f}\n")
                    else:
                        radius = shape['diameter'] / 2 * 0.015 / beta / 0.1038 * pixscale
                        f.write(f"C,{(x1_laser + x2_laser)/2:9.4f},{(y1_laser + y2_laser)/2:9.4f},{abs((x2_laser-x1_laser)/2):9.4f}\n")
        except IOError as e:
            QMessageBox.critical(None, "Error", f"Failed to write SBR file: {str(e)}")

    def close(self):
        self.fv.stop_local_plugin(self.chname, str(self))
        return True

    def start(self):
        canvas = self.fitsimage.get_canvas()
        if self.canvas not in canvas.objects:
            canvas.add(self.canvas)
        self.fitsimage.redraw()

    def stop(self):
        self.canvas.delete_all_objects()
        self.fitsimage.get_canvas().delete_object_by_tag('maskbuilder-canvas')
        self.shapes.clear()
        self._undo_stack.clear()
        self.remove_fov_overlay()

    def __str__(self):
        return 'moircs_mask_builder'
