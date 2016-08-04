#
# HSCPlanner.py -- HSCPlanner plugin for Ginga reference viewer
#
# Eric Jeschke (eric@naoj.org)
#
# This is open-source software licensed under a BSD license.
# Please see the file LICENSE.txt for details.
#
import os
import numpy
import tempfile

from ginga import GingaPlugin
from ginga.misc import Widgets, Bunch, Future
from ginga.util import wcs, dp, catalog
from ginga.util.six.moves import map

from hsc_ccd_info import info

sdo_map = {
    0: '1_53',
    1: '1_54',
    2: '1_55',
    3: '1_56',
    4: '1_42',
    5: '1_43',
    6: '1_44',
    7: '1_45',
    8: '1_46',
    9: '1_47',
    10: '1_36',
    11: '1_37',
    12: '1_38',
    13: '1_39',
    14: '1_40',
    15: '1_41',
    16: '0_30',
    17: '0_29',
    18: '0_28',
    19: '1_32',
    20: '1_33',
    21: '1_34',
    22: '0_27',
    23: '0_26',
    24: '0_25',
    25: '0_24',
    26: '1_00',
    27: '1_01',
    28: '1_02',
    29: '1_03',
    30: '0_23',
    31: '0_22',
    32: '0_21',
    33: '0_20',
    34: '1_04',
    35: '1_05',
    36: '1_06',
    37: '1_07',
    38: '0_19',
    39: '0_18',
    40: '0_17',
    41: '0_16',
    42: '1_08',
    43: '1_09',
    44: '1_10',
    45: '1_11',
    46: '0_15',
    47: '0_14',
    48: '0_13',
    49: '0_12',
    50: '1_12',
    51: '1_13',
    52: '1_14',
    53: '1_15',
    54: '0_11',
    55: '0_10',
    56: '0_09',
    57: '0_08',
    58: '1_16',
    59: '1_17',
    60: '1_18',
    61: '1_19',
    62: '0_07',
    63: '0_06',
    64: '0_05',
    65: '0_04',
    66: '1_20',
    67: '1_21',
    68: '1_22',
    69: '1_23',
    70: '0_03',
    71: '0_02',
    72: '0_01',
    73: '0_00',
    74: '1_24',
    75: '1_25',
    76: '1_26',
    77: '1_27',
    78: '0_34',
    79: '0_33',
    80: '0_32',
    81: '1_28',
    82: '1_29',
    83: '1_30',
    84: '0_41',
    85: '0_41',
    86: '0_39',
    87: '0_38',
    88: '0_37',
    89: '0_36',
    90: '0_47',
    91: '0_46',
    92: '0_45',
    93: '0_44',
    94: '0_43',
    95: '0_42',
    96: '0_56',
    97: '0_55',
    98: '0_54',
    99: '0_53',
    100: '0_31',
    101: '1_35',
    102: '0_35',
    103: '1_31',
    }

# ESO DSS server
image_archives = [('ESO', 'eso', catalog.ImageServer,
                  "http://archive.eso.org/dss/dss?ra=%(ra)s&dec=%(dec)s&mime-type=application/x-fits&x=%(width)s&y=%(height)s",
                   "ESO DSS archive"),
                  ## ('SDSS', 'sdss', catalog.ImageServer,
                  ##  "http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx?ra=%(ra)s&dec=%(dec)s&scale=0.396127&width=%(width)s&height=%(height)s&opt=",
                  ##  "SDSS DR12 Image archive"),
                  ]

class HSCPlanner(GingaPlugin.LocalPlugin):

    def __init__(self, fv, fitsimage):
        # superclass defines some variables for us, like logger
        super(HSCPlanner, self).__init__(fv, fitsimage)

        self.layertag = 'hscplanner-canvas'
        self.dither_types = ['1', '5', 'N']
        self.dither_type = '1'
        self.dither_steps = 5

        self.ctr_ra_deg = 0.0
        self.ctr_deg_deg = 0.0
        self.targets = []
        self.target_radius = 0.01
        self.pa_deg = 0.0

        # default dra/ddec is 120"
        self.dra = 120.0
        self.ddec = 120.0
        self.tdith = 15.0
        self.rdith = 120.0

        # TODO: get this from a module
        #self.px_scale = 0.000280178318866
        self.px_scale = 0.000047
        self.fov_deg = 2.0

        # Add ESO DSS image server to our set of image servers
        bank = fv.get_ServerBank()
        for (longname, shortname, klass, url, description) in image_archives:
            obj = klass(self.logger, longname, shortname, url, description)
            bank.addImageServer(obj)

        # Get image server name
        self.servername = 'eso'
        self.dsscnt = 0
        self.tmpdir = tempfile.mkdtemp()

        self.dc = fv.getDrawClasses()
        canvas = self.dc.DrawingCanvas()
        canvas.add_callback('cursor-down', self.btn_down_cb)
        canvas.setSurface(self.fitsimage)
        self.canvas = canvas

    def build_gui(self, container):
        top = Widgets.VBox()
        top.set_border_width(4)

        vbox, sw, orientation = Widgets.get_oriented_box(container)
        vbox.set_border_width(4)
        vbox.set_spacing(2)

        self.msgFont = self.fv.getFont("sansFont", 12)
        tw = Widgets.TextArea(wrap=True, editable=False)
        tw.set_font(self.msgFont)
        self.tw = tw

        fr = Widgets.Expander("Instructions")
        vbox2 = Widgets.VBox()
        vbox2.add_widget(tw)
        #vbox2.add_widget(Widgets.Label(''), stretch=1)
        fr.set_widget(vbox2)
        vbox.add_widget(fr, stretch=0)

        fr = Widgets.Frame("Position")

        captions = (('RA:', 'label', 'RA', 'entry',
                     'DEC:', 'label', 'DEC', 'entry',),
                    ('Equinox:', 'label', 'Equinox', 'entry',
                     'Object:', 'label', 'Name', 'entry'),
                    ('Add Target', 'button', 'Clear All', 'button'),
                    ('Set Pointing', 'button'),
                    ('Pointing:', 'label', 'pra', 'llabel', 'pdec', 'llabel'),
                    ('FOV:', 'label', 'fov', 'entry',
                     'Create Blank', 'button', 'Get DSS', 'button'),
                    )
        w, b = Widgets.build_info(captions, orientation=orientation)
        self.w = b

        b.equinox.set_text("2000")
        # Currently assume J2000 targets--this will be fixed in future
        b.equinox.set_enabled(False)
        b.add_target.add_callback('activated', lambda w: self.add_target_cb())
        b.clear_all.add_callback('activated', lambda w: self.clear_targets_cb())
        b.fov.set_text(str(self.fov_deg))

        b.set_pointing.add_callback('activated',
                                    lambda w: self.set_pointing_cb())
        b.create_blank.add_callback('activated',
                                    lambda w: self.create_blank_image())
        b.get_dss.add_callback('activated',
                                    lambda w: self.get_dss_image())

        fr.set_widget(w)
        vbox.add_widget(fr, stretch=0)

        fr = Widgets.Frame("Acquisition")

        vbox2 = Widgets.VBox()
        captions = (('Dither Type:', 'label', 'Dither Type', 'combobox',
                     'Dither Steps:', 'label', 'Dither Steps', 'spinbutton'),
                    ('INSROT_PA:', 'label', 'PA', 'entry'),
                    ('RA Offset:', 'label', 'RA Offset', 'entry',
                     'DEC Offset:', 'label', 'DEC Offset', 'entry',),
                    ('Dith1:', 'label', 'Dith1', 'entry',
                     'Dith2:', 'label', 'Dith2', 'entry',),
                    ('Skip:', 'label', 'Skip', 'spinbutton',
                     'Stop:', 'label', 'Stop', 'spinbutton'),
                    )
        w, b = Widgets.build_info(captions, orientation=orientation)
        self.w.update(b)

        combobox = b.dither_type
        for name in self.dither_types:
            combobox.append_text(name)
        index = self.dither_types.index(self.dither_type)
        combobox.set_index(index)
        combobox.add_callback('activated',
                              lambda w, idx: self.set_dither_type_cb())
        b.pa.set_text(str(self.pa_deg))
        b.dither_steps.set_limits(1, 20)
        b.dither_steps.add_callback('value-changed',
                                    lambda w, idx: self.set_dither_steps_cb(idx))
        b.pa.set_tooltip("Angle of instrument rotator in deg")

        b.ra_offset.set_text(str(0.0))
        b.dec_offset.set_text(str(0.0))
        b.ra_offset.set_tooltip("RA offset from center of field in arcsec")
        b.dec_offset.set_tooltip("DEC offset from center of field in arcsec")
        b.skip.set_value(0)
        b.stop.set_value(1)

        vbox2.add_widget(w)

        captions = (('_x3', 'spacer', '_x4', 'spacer',
                     '_x5', 'spacer'),
                    ('Update Image', 'button'),
                    )
        w, b = Widgets.build_info(captions, orientation=orientation)
        self.w.update(b)

        b.update_image.add_callback('activated',
                                       lambda w: self.update_info_cb())

        vbox2.add_widget(w)

        fr.set_widget(vbox2)
        vbox.add_widget(fr, stretch=0)

        captions = (("Show Step:", 'label', 'Show Step', 'spinbutton'),
                    )
        w, b = Widgets.build_info(captions, orientation=orientation)
        self.w.update(b)

        b.show_step.add_callback('value-changed',
                                 lambda w, idx: self.show_step_cb(idx))

        vbox.add_widget(w, stretch=0)

        spacer = Widgets.Label('')
        vbox.add_widget(spacer, stretch=1)

        top.add_widget(sw, stretch=1)

        btns = Widgets.HBox()
        btns.set_spacing(3)

        btn = Widgets.Button("Close")
        btn.add_callback('activated', lambda w: self.close())
        btns.add_widget(btn, stretch=0)
        btns.add_widget(Widgets.Label(''), stretch=1)
        top.add_widget(btns, stretch=0)

        container.add_widget(top, stretch=1)

    def set_dither_type_cb(self):
        index = self.w.dither_type.get_index()
        self.dither_type = self.dither_types[index]

        if self.dither_type == '1':
            self.w.dither_steps.set_value(1)
            self.w.dither_steps.set_enabled(False)
            ## self.w.show_step.set_limits(1, 1)
            self.w.dith1.set_enabled(False)
            self.w.dith2.set_enabled(False)
            self.w.dith1.set_text('')
            self.w.dith2.set_text('')
            self.w.skip.set_enabled(False)
            self.w.stop.set_enabled(False)

        elif self.dither_type == '5':
            self.w.dith1.set_text(str(self.dra))
            self.w.dith2.set_text(str(self.ddec))
            self.w.dither_steps.set_value(5)
            self.w.dither_steps.set_enabled(False)
            ## self.w.show_step.set_limits(1, 5)
            self.w.dith1.set_enabled(True)
            self.w.dith2.set_enabled(True)
            self.w.lbl_dith1.set_text("Delta RA:")
            self.w.lbl_dith2.set_text("Delta DEC:")
            self.w.skip.set_enabled(True)
            self.w.skip.set_limits(0, 4)
            self.w.skip.set_value(0)
            self.w.stop.set_enabled(True)
            self.w.stop.set_limits(1, 5)
            self.w.stop.set_value(5)

        else:
            N = self.dither_steps
            self.w.dith1.set_text(str(self.rdith))
            self.w.dith2.set_text(str(self.tdith))
            self.w.dither_steps.set_value(N)
            self.w.dither_steps.set_enabled(True)
            ## self.w.show_step.set_limits(1, N)
            self.w.dith1.set_enabled(True)
            self.w.dith2.set_enabled(True)
            self.w.lbl_dith1.set_text("RDITH:")
            self.w.lbl_dith2.set_text("TDITH:")
            self.w.skip.set_enabled(True)
            self.w.skip.set_limits(0, N-1)
            self.w.skip.set_value(0)
            self.w.stop.set_enabled(True)
            self.w.stop.set_limits(1, N)
            self.w.stop.set_value(N)

        #self.fv.error_wrap(self.draw_dither_positions)
        #self.show_step(1)

        return True

    def update_info_cb(self):
        try:
            # calculate center and target coordinates
            ra_off_deg = float(self.w.ra_offset.get_text()) / 3600.0
            dec_off_deg = float(self.w.dec_offset.get_text()) / 3600.0
            self.ra_off_deg = ra_off_deg
            self.dec_off_deg = dec_off_deg

            self.pa_deg = float(self.w.pa.get_text())

            # save dither params
            if self.dither_type == '5':
                self.dra = float(self.w.dith1.get_text())
                self.ddec = float(self.w.dith2.get_text())

            elif self.dither_type == 'N':
                self.rdith = float(self.w.dith1.get_text())
                self.tdith = float(self.w.dith2.get_text())

            # create a blank image if we don't have one already
            image = self.fitsimage.get_image()
            if image is None:
                self.create_blank_image()

            # add targets to canvas
            #self.draw_targets()

            self.draw_dither_positions()

            self.fv.error_wrap(self.draw_ccds, self.ctr_ra_deg, self.ctr_dec_deg)

            start = int(self.w.skip.get_value()) + 1
            stop = int(self.w.stop.get_value())
            self.w.show_step.set_limits(start, stop)
            self.w.show_step.set_value(start)

            self.show_step(start)

        except Exception as e:
            self.fv.show_error(str(e))
        return True

    def calc_dither1(self, n):
        ctr_ra, ctr_dec = wcs.add_offset_radec(
            self.ctr_ra_deg, self.ctr_dec_deg,
            self.ra_off_deg, self.dec_off_deg)
        return (ctr_ra, ctr_dec)

    def calc_dither5(self, n):
        idx = n - 1
        l = ((0.0, 0.0), (1.0, -2.0), (2.0, 1.0), (-1.0, 2.0), (-2.0, -1.0))
        mra, mdec = l[idx]

        dra = float(self.w.dith1.get_text()) / 3600.0
        ddec = float(self.w.dith2.get_text()) / 3600.0

        ## ctr_ra, ctr_dec = wcs.add_offset_radec(
        ##     self.ctr_ra_deg, self.ctr_dec_deg, mra * dra, mdec * ddec)
        ctr_ra, ctr_dec = wcs.add_offset_radec(
            self.ctr_ra_deg, self.ctr_dec_deg,
            mra * dra + self.ra_off_deg, mdec * ddec + self.dec_off_deg)
        return (ctr_ra, ctr_dec)

    def calc_ditherN(self, n):
        n = n - 1
        rdith = float(self.w.dith1.get_text()) / 3600.0
        tdith = float(self.w.dith2.get_text())
        ndith = float(self.dither_steps)

        sin_res = numpy.sin(numpy.radians(n * 360.0/ndith + tdith))
        cos_res = numpy.cos(numpy.radians(n * 360.0/ndith + tdith))
        self.logger.debug("sin=%f cos=%f" % (sin_res, cos_res))

        ## ctr_ra, ctr_dec = wcs.add_offset_radec(
        ##     self.ctr_ra_deg, self.ctr_dec_deg, cos_res * rdith, sin_res * rdith)
        ctr_ra, ctr_dec = wcs.add_offset_radec(
            self.ctr_ra_deg, self.ctr_dec_deg,
            cos_res * rdith + self.ra_off_deg,
            sin_res * rdith + self.dec_off_deg)
        return (ctr_ra, ctr_dec)

    def calc_dither(self, n):
        dith_type = self.dither_type
        if dith_type == '1':
            ra, dec = self.calc_dither1(n)
        elif dith_type == '5':
            ra, dec = self.calc_dither5(n)
        elif dith_type == 'N':
            ra, dec = self.calc_ditherN(n)
        return (ra, dec)

    def get_dither_positions(self):
        dith_type = self.dither_type
        skip = self.w.skip.get_value()
        stop = self.w.stop.get_value()

        if dith_type == '1':
            return 1, 1, [ self.calc_dither1(n) for n in range(1, 2) ]
        elif dith_type == '5':
            return skip+1, stop, [ self.calc_dither5(n) for n in range(skip+1, stop+1) ]
        elif dith_type == 'N':
            #N = self.dither_steps
            return skip+1, stop, [ self.calc_ditherN(n) for n in range(skip+1, stop+1) ]

    def set_dither_steps_cb(self, n):
        self.dither_steps = n
        ## self.w.show_step.set_limits(1, n)
        self.w.skip.set_limits(0, n-1)
        self.w.skip.set_value(0)
        self.w.stop.set_limits(1, n)
        self.w.stop.set_value(n)
        #self.fv.error_wrap(self.draw_dither_positions)
        return True

    def _show_step(self, n):
        self.logger.info("moving to step %d" % (n))
        ra, dec = self.calc_dither(n)
        image = self.fitsimage.get_image()
        data_x, data_y = image.radectopix(ra, dec)
        delta_x = data_x - self.ctr_pt.x
        delta_y = data_y - self.ctr_pt.y
        self.ccd_overlay.move_delta(delta_x, delta_y)
        self.canvas.update_canvas()
        return True

    def show_step(self, n):
        #self.w.show_step.set_text(str(n))
        self.w.show_step.set_value(n)
        self._show_step(n)

    def show_step_cb(self, n):
        #n = int(strip(self.w.show_step.get_text()))
        self.logger.info("step %d!" % (n))
        self.fv.error_wrap(self._show_step, n)
        return True

    def close(self):
        chname = self.fv.get_channelName(self.fitsimage)
        self.fv.stop_local_plugin(chname, str(self))
        return True

    def instructions(self):
        self.tw.set_text(instructions)

    def start(self):
        self.instructions()
        # start operation
        try:
            obj = self.fitsimage.getObjectByTag(self.layertag)

        except KeyError:
            # Add our layer
            self.fitsimage.add(self.canvas, tag=self.layertag)

        self.resume()

    def pause(self):
        self.canvas.ui_setActive(False)

    def resume(self):
        self.canvas.ui_setActive(True)
        self.fv.showStatus("")

    def stop(self):
        # remove the canvas from the image
        try:
            self.fitsimage.deleteObjectByTag(self.layertag)
        except:
            pass
        self.canvas.ui_setActive(False)
        self.fv.showStatus("")

    def redo(self):
        image = self.fitsimage.get_image()
        wd, ht = image.get_size()
        ra_deg, dec_deg = image.pixtoradec(wd / 2.0, ht / 2.0)
        ra, dec = self._set_radec(ra_deg, dec_deg)

        # set pointing
        self.ctr_ra_deg, self.ctr_dec_deg = ra_deg, dec_deg
        self.w.pra.set_text(ra)
        self.w.pdec.set_text(dec)

    def _ccd_in_dither(self, dither, polygon):
        x_arr, y_arr = dither.T[0], dither.T[1]
        res = polygon.contains_arr(x_arr, y_arr)
        res = numpy.any(res)
        return res

    def get_paths(self, ctr_ra, ctr_dec, info):
        paths = []
        keys = list(info.keys())
        keys.sort()
        for key in keys:
            path = [ wcs.add_offset_radec(ctr_ra, ctr_dec, dra, ddec)
                     for dra, ddec in info[key]['polygon'] ]
            paths.append((key, path))
        return paths


    def draw_ccds(self, ctr_ra_deg, ctr_dec_deg):
        image = self.fitsimage.get_image()
        if image is None:
            return

        ctr_x, ctr_y = image.radectopix(ctr_ra_deg, ctr_dec_deg)
        self.ctr_x, self.ctr_y = ctr_x, ctr_y

        l = []

        ctr_pt = self.dc.Point(ctr_x, ctr_y, 50,
                               linewidth=4, color='orange',
                               style='plus', coord='data')
        l.append(ctr_pt)
        self.ctr_pt = ctr_pt

        paths = self.get_paths(ctr_ra_deg, ctr_dec_deg, info)

        start, stop, dither_positions = self.get_dither_positions()
        dither = list(map(lambda pt: image.radectopix(pt[0], pt[1]),
                          dither_positions))
        dither = numpy.array(dither)

        ## only_ccds_in_dithers = self.w.only_dithers.get_state()

        crdmap = self.fitsimage.get_coordmap('data')
        for key, path in paths:
            points = list(map(lambda pt: image.radectopix(pt[0], pt[1]), path))
            points = numpy.array(points)

            if 'color' in info[key]:
                color = info[key]['color']
            else:
                color = 'lightgreen'
            p = self.dc.Polygon(points, color=color, showcap=False)

            # hack to be able to check containment before object is on
            # the canvas
            p.crdmap = crdmap

            ## # exclude this ccd if not involved in the dither
            ## if only_ccds_in_dithers and not self._ccd_in_dither(dither, p):
            ##     continue

            # annotate with the CCD name
            pcx, pcy = p.get_center_pt()
            name = sdo_map[key]
            t = self.dc.Text(pcx, pcy, text=name, color=color, fontsize=12,
                             coord='data')

            l.append(self.dc.CompoundObject(p, t))
            #l.append(p)

        obj = self.dc.CompoundObject(*l)

        self.canvas.deleteObjectByTag('ccd_overlay')
        self.canvas.add(obj, tag='ccd_overlay', redraw=False)

        # rotate for pa
        obj.rotate(-self.pa_deg, xoff=ctr_x, yoff=ctr_y)
        self.ccd_overlay = obj

        self.canvas.update_canvas()
        self.logger.debug("canvas rotated")

    def draw_dither_positions(self):
        Text = self.canvas.getDrawClass('text')
        Point = self.canvas.getDrawClass('point')
        CompoundObject = self.canvas.getDrawClass('compoundobject')

        image = self.fitsimage.get_image()

        l = []
        start, stop, posns = self.get_dither_positions()
        i = start
        for ra, dec in posns:
            l.append(Text(ra, dec, text="%d" % i, color='yellow',
                          coord='wcs'))
            l.append(Point(ra, dec, self.target_radius, color='yellow',
                           style='plus', coord='wcs'))
            i += 1
        obj = CompoundObject(*l)

        self.canvas.deleteObjectByTag('dither_positions')
        self.canvas.add(obj, tag='dither_positions')

    def draw_targets(self):
        self.canvas.deleteObjects(self.canvas.getObjectsByTagpfx('target'))

        Point = self.canvas.getDrawClass('point')
        Circle = self.canvas.getDrawClass('circle')
        CompoundObject = self.canvas.getDrawClass('compoundobject')

        for i, tgt in enumerate(self.targets):
            self.canvas.add(CompoundObject(
                Point(tgt.ra_deg, tgt.dec_deg, self.target_radius,
                      linewidth=1, color='skyblue', coord='wcs'),
                Circle(tgt.ra_deg, tgt.dec_deg, self.target_radius,
                       linewidth=4, color='skyblue', coord='wcs')),
                            tag='target%d' % (i))

    def create_blank_image(self):
        self.fitsimage.onscreen_message("Creating blank field...",
                                        delay=1.0)
        self.fv.update_pending()

        fov_deg = float(self.w.fov.get_text())

        image = dp.create_blank_image(self.ctr_ra_deg, self.ctr_dec_deg,
                                      fov_deg,
                                      self.px_scale, self.pa_deg,
                                      cdbase=[-1, 1],
                                      logger=self.logger)
        image.set(nothumb=True)
        self.fitsimage.set_image(image)

    def _set_radec(self, ra_deg, dec_deg):
        ra_txt = wcs.raDegToString(ra_deg)
        self.w.ra.set_text(ra_txt)
        dec_txt = wcs.decDegToString(dec_deg)
        self.w.dec.set_text(dec_txt)
        return ra_txt, dec_txt

    def set_pointing_cb(self):
        try:
            ra = self.w.ra.get_text()
            dec = self.w.dec.get_text()

            tgt_ra_deg = wcs.hmsStrToDeg(ra)
            tgt_dec_deg = wcs.dmsStrToDeg(dec)
            self.ctr_ra_deg, self.ctr_dec_deg = tgt_ra_deg, tgt_dec_deg

            self.w.pra.set_text(ra)
            self.w.pdec.set_text(dec)
        except Exception as e:
            self.fv.show_error(str(e))

    def get_dss_image(self):
        self.fitsimage.onscreen_message("Getting DSS image; please wait...")
        # offload to non-GUI thread to keep viewer somewhat responsive
        self.fv.nongui_do(self._get_dss_image)

    def _get_dss_image(self):
        try:
            fov_deg = float(self.w.fov.get_text())

            # width and height are specified in arcmin
            sgn, deg, mn, sec = wcs.degToDms(fov_deg)
            wd = deg*60.0 + float(mn) + sec/60.0
            sgn, deg, mn, sec = wcs.degToDms(fov_deg)
            ht = deg*60.0 + float(mn) + sec/60.0

            # coordinates
            ra_txt = wcs.raDegToString(self.ctr_ra_deg)
            dec_txt = wcs.decDegToString(self.ctr_dec_deg)

            # these are the params to DSS
            params = dict(ra=ra_txt, dec=dec_txt, width=wd, height=ht)

            # Query the server and download file
            srvbank = self.fv.get_ServerBank()
            filename = 'sky-' + str(self.dsscnt) + '.fits'
            self.dsscnt = (self.dsscnt + 1) % 5
            filepath = os.path.join(self.tmpdir, filename)
            try:
                os.remove(filepath)
            except Exception as e:
                self.logger.warn("failed to remove tmp file '%s': %s" % (
                    filepath, str(e)))
            try:
                dstpath = srvbank.getImage(self.servername, filepath, **params)

            except Exception as e:
                errmsg = "Failed to download sky image: %s" % (str(e))
                self.fv.gui_do(self.fv.show_error, errmsg)
                return

            try:
                image = self.fv.load_image(dstpath)

            except Exception as e:
                errmsg = "Failed to load downloaded sky image: %s" % (str(e))
                self.fv.gui_do(self.fv.show_error, errmsg)
                return

            image.set(nothumb=True)
            self.fv.gui_do(self.fitsimage.set_image, image)

        finally:
            self.fv.gui_do(self.fitsimage.onscreen_message_off)

    def add_target_cb(self):
        ra = self.w.ra.get_text()
        dec = self.w.dec.get_text()
        eq = self.w.equinox.get_text()
        name = self.w.name.get_text()

        try:
            tgt_ra_deg = wcs.hmsStrToDeg(ra)
            tgt_dec_deg = wcs.dmsStrToDeg(dec)
            tgt_equinox = float(eq)

            # TODO: use 'target' entity from qplan?
            tgt = Bunch.Bunch(ra_deg=tgt_ra_deg, dec_deg=tgt_dec_deg,
                              equinox=tgt_equinox, name=name)
            self.targets.append(tgt)

            self.draw_targets()

        except Exception as e:
            errmsg = "Failed to process target: %s" % (str(e))
            self.fv.show_error(errmsg)
        return True

    def clear_targets_cb(self):
        self.targets = []

        self.draw_targets()
        return True

    def btn_down_cb(self, canvas, event, data_x, data_y):
        self.logger.info("cursor at %f,%f" % (data_x, data_y))
        image = self.fitsimage.get_image()
        ra_deg, dec_deg = image.pixtoradec(data_x, data_y)
        self._set_radec(ra_deg, dec_deg)
        self.logger.info("cursor callback done")
        return True

    def __str__(self):
        return 'hscplanner'


instructions = """
HSCPlanner works according to the following steps:

A) establish the pointing of the telescope
B) create a blank field or DSS field from the established pointing
C) place one or more targets within the field
D) set the acquisition parameters and visualize
E) repeat D) or from earlier steps as needed or desired

We will go over each of these steps in turn.

A) Establishing Pointing

To establish pointing, you can type RA and DEC coordinates (sexigesimal) into the corresponding boxes under the "Position" section of the GUI and click "Set Pointing".

Another way that is fairly easy is to drag a FITS image that has a reasonably accurate WCS with pointing for the desired field into the main window. Click anywhere on the image to set the RA and DEC boxes, and you can then click "Set Pointing".

You can use this image as the background image (and skip Step B) if the FOV is wide enough to show your target of interest (HSC FOV is approx 1.5 deg).


B) Create Field from Pointing

Once pointing is established, we need to create a background field with correct WCS to do the correct overplotting to visualize the acquisition. You can either load your own background image (already discussed), create a blank field, or download a DSS image of the field (if available).

To create a blank image click "Create Blank". To download a DSS field click "Get DSS". To use the DSS function you will need a functioning internet connection.

----------------------------------------------
NOTE: the default location for DSS download is from ESO's web site and it may take up to a minute to download and update the background image.  If you experience trouble acquiring a DSS image it is recommended that you download your own background FITS image and load it in step A.
----------------------------------------------

B) Placing Targets within the Field

To place targets within the field, you can type RA and DEC coordinates as in step A above or simply click in the field where you want a target (as in step A the RA and DEC boxes will be filled when you click). Press "Add Target" to add the current RA/DEC as a target. To clear the target list, press "Clear All".

Set the Acquisition Parameters

Now we are finally ready to set the acquisition parameters and visualize the field throughout the dither. In the section labeled "Acquisition" you can set any of the parameters normally used for HSC acquisition.

The parameters are:

- Dither type: 1 for a single shot, 5 for a 5-point box pattern, and N for an N-point circular pattern

- Dither steps: only settable for N-type dither, set it to the number of dither positions

- INSROT_PA: this parameter will set up the instrument rotator to set the rotation of the field on the CCD plane--see the instrument documentation for details

- RA Offset, DEC Offset: offsets in arc seconds from the pointing position in the center of the field

- Dith1, Dith2 (Delta RA, Delta DEC or RDITH, TDITH): the names of these parameters change according to the dither type selected.  For Dither Type 1 they are not used.  For Dither Type 5, these parameters specify the offsets in arc seconds for Delta RA and Delta DEC to accomplish the dither between positions.  For Dither Type N they specify the offset in arc seconds (RDITH) and the angle offset in degrees (TDITH) for the circular dither.  See the instrument documentation for more information.

- Skip: the number of shots to skip from the beginning of a dither. Leave at the default for the full dither.

- Stop: used to terminate a dither early after a certain number of shots.  Leave at the default for the full dither.

Once you have set the parameters as desired, press the "Update Image" button to update the overlays. You can then use the "Show Step" control to step through your dither.

NOTE

There is currently a bug with the "Show Step" control that you need to change at least two steps before the program starts responding to the step adjustment after doing an "Update Image".

HINTS

It may be helpful to view the field first with the image zoomed out, and then later to pan to your target (hint: use Shift+click to set pan position) and zoom in to more closely watch the detailed positioning of the target(s) on the detector grid.

Repeat as Desired

You can go back to any step and repeat from there as needed.
"""

#END
