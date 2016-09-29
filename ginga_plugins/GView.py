#
# GView.py -- plugin for Ginga implementing some of the commands from
#               the old ZView viewer
#
# Eric Jeschke (eric@naoj.org)
#
# This is open-source software licensed under a BSD license.
# Please see the file LICENSE.txt for details.
#
"""

Examples:
    rd 1 /home/eric/testdata/SPCAM/SUPA01118760.fits
    v 1
    cm jt
    cd /home/eric/testdata/SPCAM
    rd 1 SUPA01118760.fits
    ql 1 SUPA0111876?.fits
    ql 1 SUPA0146974?.fits
    cd /opt/gen2/data/HSC
    hql 1 HSCA05560*.fits
"""
import os
import glob
import time
import math

import numpy
from astropy.io import fits

from ginga.misc.plugins.Command import Command, CommandInterpreter
from ginga import AstroImage, cmap
from ginga.gw import Plot
from ginga.util import iqcalc, plots, wcs, dp
from ginga.misc import Bunch, Task

# add any matplotlib colormaps we have lying around
cmap.add_matplotlib_cmaps(fail_on_import_error=False)

from naoj.spcam.spcam_dr import SuprimeCamDR
from naoj.hsc.hsc_dr import HyperSuprimeCamDR, hsc_ccd_data
# add "Jon Tonley" color map
from naoj.cmap import jt
cmap.add_cmap("jt", jt.cmap_jt)



class GView(Command):

    def __init__(self, fv):
        # superclass defines some variables for us, like logger
        super(GView, self).__init__(fv)

        self._cmdobj = GViewInterpreter(fv, self)

        fv.add_callback('add-channel', self._cmdobj.add_channel_cb)
        ## fv.add_callback('delete-channel', self.delete_channel)
        ## fv.set_callback('channel-change', self.focus_cb)

    def __str__(self):
        return 'gview'


class GViewInterpreter(CommandInterpreter):

    def __init__(self, fv, plugin):
        super(GViewInterpreter, self).__init__(fv, plugin)

        self._view = None
        self.buffers = Bunch.Bunch()

        self.iqcalc = iqcalc.IQCalc(self.logger)
        self._plot = None
        self._plot_w = None

        # Peak finding parameters and selection criteria
        self.radius = 20
        self.settings = {}
        self.max_side = self.settings.get('max_side', 1024)
        self.radius = self.settings.get('radius', 10)
        self.threshold = self.settings.get('threshold', None)
        self.min_fwhm = self.settings.get('min_fwhm', 2.0)
        self.max_fwhm = self.settings.get('max_fwhm', 50.0)
        self.min_ellipse = self.settings.get('min_ellipse', 0.5)
        self.edgew = self.settings.get('edge_width', 0.01)
        self.show_candidates = self.settings.get('show_candidates', False)
        # Report in 0- or 1-based coordinates
        self.pixel_coords_offset = self.settings.get('pixel_coords_offset',
                                                     0.0)

        self.contour_radius = 10

        self.spcam_dr = SuprimeCamDR(logger=self.logger)
        self.hsc_dr = HyperSuprimeCamDR(logger=self.logger)

        self.sub_bias = True
        # For flat fielding
        self.flat = {}
        self.flat_dir = '.'
        self.flat_filter = None
        self.use_flat = False

    def add_channel_cb(self, gvshell, channel):
        fi = channel.fitsimage
        bm = fi.get_bindmap()

        # add a new "zview" mode
        bm.add_mode('z', 'zview', mode_type='locked', msg=None)

        # zview had this kind of zooming function
        bm.map_event('zview', (), 'ms_left', 'zoom_in')
        bm.map_event('zview', (), 'ms_right', 'zoom_out')
        bm.map_event('zview', ('ctrl',), 'ms_left', 'zoom_out')

        # borrow some bindings from pan mode
        bm.map_event('zview', (), 'kp_left', 'pan_left')
        bm.map_event('zview', (), 'kp_right', 'pan_right')
        bm.map_event('zview', (), 'kp_up', 'pan_up')
        bm.map_event('zview', (), 'kp_down', 'pan_down')
        bm.map_event('zview', (), 'kp_s', 'pan_zoom_save')
        bm.map_event('zview', (), 'kp_1', 'pan_zoom_set')

        bm.map_event('zview', (), 'kp_p', 'radial-plot')
        bm.map_event('zview', (), 'kp_r', 'radial-plot')
        fi.set_callback('keydown-radial-plot',
                        self.plot_cmd_cb, self.do_radial_plot,
                        "Radial Profile")
        bm.map_event('zview', (), 'kp_e', 'contour-plot')
        fi.set_callback('keydown-contour-plot',
                        self.plot_cmd_cb, self.do_contour_plot,
                        "Contours")
        bm.map_event('zview', (), 'kp_g', 'gaussians-plot')
        fi.set_callback('keydown-gaussians-plot',
                        self.plot_cmd_cb, self.do_gaussians_plot,
                        "FWHM")

        # bindings customizations
        bd = fi.get_bindings()
        settings = bd.get_settings()

        # ZVIEW has a faster zoom ratio, by default
        settings.set(scroll_zoom_direct_scale=True)

    ##### COMMANDS #####

    def cmd_rd(self, bufname, path, *args):
        """rd bufname path

        Read file from `path` into buffer `bufname`.  If the buffer does
        not exist it will be created.

        If `path` does not begin with a slash it is assumed to be relative
        to the current working directory.
        """
        if not path.startswith('/'):
            path = os.path.join(os.getcwd(), path)
        if bufname in self.buffers:
            self.log("Buffer %s is in use. Will discard the previous data" % (
                bufname))
            image = self.buffers[bufname]
        else:
            # new buffer
            image = AstroImage.AstroImage(logger=self.logger)
            self.buffers[bufname] = image

        self.log("Reading file...(%s)" % (path))
        image.load_file(path)
        # TODO: how to know if there is an error
        self.log("File read")

    def cmd_tv(self, bufname, *args):
        """tv bufname [min max] [colormap]

        Display buffer `bufname` in the current viewer.  If no viewer
        exists one will be created.

        Optional:
        `min` and `max` specify lo/hi cut levels to scale the image
        data for display.

        `colormap` specifies a color map to use for the image.
        """
        if not bufname in self.buffers:
            self.log("!! No such buffer: '%s'" % (bufname))
            return
        image = self.buffers[bufname]

        if self._view is None:
            self.make_viewer("GView")

        self._view.add_image(image)

        gw = self._view.viewer

        args = list(args)

        locut = None
        if len(args) > 0:
            try:
                locut = float(args[0])
                hicut = float(args[1])
                args = args[2:]
            except ValueError:
                pass

        if locut is not None:
            gw.cut_levels(locut, hicut)

        if 'bw' in args:
            # replace "bw" with gray colormap
            i = args.index('bw')
            args[i] = 'gray'

        if len(args) > 0:
            cm_name = args[0]
            if cm_name == 'inv':
                gw.invert_cmap()
            else:
                gw.set_color_map(cm_name)

    def cmd_head(self, bufname, *args):
        """head buf [kwd ...]

        List the headers for the image in the named buffer.
        """
        if bufname not in self.buffers:
            self.log("No such buffer: '%s'" % (bufname))
            return

        image = self.buffers[bufname]
        header = image.get_header()
        res = []
        # TODO: include the comments
        if len(args) > 0:
            for kwd in args:
                if not kwd in header:
                    res.append("%-8.8s  -- NOT FOUND IN HEADER --" % (kwd))
                else:
                    res.append("%-8.8s  %s" % (kwd, str(header[kwd])))
        else:
            for kwd in header.keys():
                res.append("%-8.8s  %s" % (kwd, str(header[kwd])))

        self.log('\n'.join(res))

    def cmd_exps(self, n=20, hdrs=None):
        """exps  [n=20, time=]

        List the last n exposures in the current directory
        """
        cwd = os.getcwd()
        files = glob.glob(cwd + '/HSCA*[0,2,4,6,8]00.fits')
        files.sort()

        n = int(n)
        files = files[-n:]

        res = []
        for filepath in files:
            with fits.open(filepath, 'readonly', memmap=False) as in_f:
                header = in_f[0].header
            line = "%(EXP-ID)-12.12s  %(HST-STR)12.12s  %(OBJECT)14.14s  %(FILTER01)8.8s" % header

            # add user specified headers
            if hdrs is not None:
                for kwd in hdrs.split(','):
                    fmt = "%%(%s)12.12s" % kwd
                    line += '  ' + (fmt % header)
            res.append(line)

        self.log('\n'.join(res))

    def cmd_lsb(self):
        """lsb

        List the buffers
        """
        names = list(self.buffers.keys())
        names.sort()

        if len(names) == 0:
            self.log("No buffers")
            return

        res = []
        for name in names:
            d = self.get_buffer_info(name)
            d.size = "%dx%d" % (d.width, d.height)
            res.append("%(name)-10.10s  %(size)13s  %(path)s" % d)
        self.log("\n".join(res))

    def cmd_rmb(self, *args):
        """rmb NAME ...

        Remove buffer NAME
        """
        for name in args:
            if name in self.buffers:
                del self.buffers[name]
            else:
                self.log("No such buffer: '%s'" % (name))
        self.cmd_lsb()

    def cmd_rm(self, *args):
        """command to be deprecated--use 'rmb'
        """
        self.log("warning: this command will be deprecated--use 'rmb'")
        self.cmd_rmb(*args)

    def _ql(self, bufname, glob_pat, dr):
        if isinstance(glob_pat, str):
            pattern = "%s/%s" % (os.getcwd(), glob_pat)
            files = glob.glob(pattern)
        else:
            # arg is a "visit" number
            exp_num = int(str(int(glob_pat)) + "00")
            files = dr.exp_num_to_file_list(os.getcwd(), exp_num)

        fov_deg = dr.fov

        # read first image to seed mosaic
        seed = files[0]
        self.logger.debug("Reading seed image '%s'" % (seed))
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(seed, memmap=False)

        name = 'mosaic'
        self.logger.debug("Preparing blank mosaic")

        if bufname in self.buffers:
            self.log("Buffer %s is in use. Will discard the previous data" % (
                bufname))
            del self.buffers[bufname]

        # new buffer
        mosaic_img = self.prepare_mosaic(image, fov_deg, name=name)
        self.buffers[bufname] = mosaic_img

        self.mosaic(files, mosaic_img, fov_deg=fov_deg, dr=dr, merge=True)

    def cmd_ql(self, bufname, glob_pat, *args):
        """ql bufname glob_pat
        """
        self._ql(bufname, glob_pat, self.spcam_dr)

    def cmd_hql(self, bufname, glob_pat, *args):
        """hql bufname glob_pat
        """
        self._ql(bufname, glob_pat, self.hsc_dr)

    def cmd_bias(self, *args):
        """bias on | off
        """
        if len(args) == 0:
            self.log("bias %s" % (self.sub_bias))
            return
        res = str(args[0]).lower()
        if res in ('y', 'yes', 't', 'true', '1', 'on'):
            self.sub_bias = True
        elif res in ('n', 'no', 'f', 'false', '0', 'off'):
            self.sub_bias = False
        else:
            self.log("Don't understand parameter '%s'" % (onoff))

    def cmd_flat(self, *args):
        """flat on | off
        """
        if len(args) == 0:
            self.log("flat %s" % (self.use_flat))
            return
        res = str(args[0]).lower()
        if res in ('y', 'yes', 't', 'true', '1', 'on'):
            self.use_flat = True
        elif res in ('n', 'no', 'f', 'false', '0', 'off'):
            self.use_flat = False
        else:
            self.log("Don't understand parameter '%s'" % (onoff))

    def cmd_flatdir(self, *args):
        """flatdir /some/path/to/flats
        """
        if len(args) > 0:
            path = str(args[0])
            if not os.path.isdir(path):
                self.log("Not a directory: %s" % (path))
                return
            self.flat_dir = path
        self.log("using (%s) for flats" % (self.flat_dir))

    def get_buffer_info(self, name):
        image = self.buffers[name]
        path = image.get('path', "None")
        res = Bunch.Bunch(dict(name=name, path=path, width=image.width,
                               height=image.height))
        return res

    def make_viewer(self, name):
        if self.fv.has_channel(name):
            channel = self.fv.get_channel(name)
        else:
            channel = self.fv.add_channel(name, num_images=0)

        self._view = channel
        return channel

    ##### PLOTS #####

    def initialize_plot(self):
        wd, ht = 800, 600
        self._plot = plots.Plot(logger=self.logger,
                                width=wd, height=ht)

        pw = Plot.PlotWidget(self._plot)
        pw.resize(wd, ht)

        self._plot_w = self.fv.make_window("Plots")
        self._plot_w.set_widget(pw)
        self._plot_w.show()

    def plot_cmd_cb(self, viewer, event, data_x, data_y, fn, title):
        try:
            fn(viewer, event, data_x, data_y)

            self._plot_w.set_title(title)
            #self._plot_w.raise_()
        finally:
            # this keeps the focus on the viewer widget, in case a new
            # window was popped up
            #viewer.get_widget().focus()
            pass

    def make_contour_plot(self):
        if self._plot is None:
            self.initialize_plot()

        fig = self._plot.get_figure()
        fig.clf()

        # Replace plot with Contour plot
        self._plot = plots.ContourPlot(logger=self.logger,
                                       figure=fig,
                                       width=600, height=600)
        self._plot.add_axis(axisbg='black')

    def do_contour_plot(self, viewer, event, data_x, data_y):
        self.log("d> (contour plot)", w_time=True)
        try:
            results = self.find_objects(viewer, data_x, data_y)
            qs = results[0]
            x, y = qs.objx, qs.objy

        except Exception as e:
            self.log("No objects found")
            # we can still proceed with a contour plot at the point
            # where the key was pressed
            x, y = data_x, data_y

        self.make_contour_plot()

        image = viewer.get_image()
        self._plot.plot_contours(x, y, self.contour_radius, image,
                                 num_contours=12)
        return True


    def make_gaussians_plot(self):
        if self._plot is None:
            self.initialize_plot()

        fig = self._plot.get_figure()
        fig.clf()

        # Replace plot with FWHM gaussians plot
        self._plot = plots.FWHMPlot(logger=self.logger,
                                    figure=fig,
                                    width=600, height=600)
        self._plot.add_axis(axisbg='white')

    def do_gaussians_plot(self, viewer, event, data_x, data_y):
        self.log("d> (gaussians plot)", w_time=True)
        try:
            results = self.find_objects(viewer, data_x, data_y)
            qs = results[0]

        except Exception as e:
            self.log("No objects found")
            return

        self.make_gaussians_plot()

        image = viewer.get_image()
        x, y = qs.objx, qs.objy

        self._plot.plot_fwhm(x, y, self.radius, image)
        return True

    def make_radial_plot(self):
        if self._plot is None:
            self.initialize_plot()

        fig = self._plot.get_figure()
        fig.clf()

        # Replace plot with Radial profile plot
        self._plot = plots.RadialPlot(logger=self.logger,
                                       figure=fig,
                                       width=700, height=600)
        self._plot.add_axis(axisbg='white')

    def do_radial_plot(self, viewer, event, data_x, data_y):
        self.log("d> (radial plot)", w_time=True)
        try:
            results = self.find_objects(viewer, data_x, data_y)
            qs = results[0]

        except Exception as e:
            self.log("No objects found")
            return

        self.make_radial_plot()

        image = viewer.get_image()
        x, y = qs.objx, qs.objy

        self._plot.plot_radial(x, y, self.radius, image)

        rpt = self.make_report(image, qs)
        self.log("seeing size %5.2f" % (rpt.starsize))
        # TODO: dump other stats from the report
        return True

    def find_objects(self, viewer, x, y):
        #x, y = viewer.get_last_data_xy()
        image = viewer.get_image()

        msg, results, qs = None, [], None
        try:
            data, x1, y1, x2, y2 = image.cutout_radius(x, y, self.radius)

            # Find bright peaks in the cutout
            self.logger.debug("Finding bright peaks in cutout")
            peaks = self.iqcalc.find_bright_peaks(data,
                                                  threshold=self.threshold,
                                                  radius=self.radius)
            num_peaks = len(peaks)
            if num_peaks == 0:
                raise Exception("Cannot find bright peaks")

            # Evaluate those peaks
            self.logger.debug("Evaluating %d bright peaks..." % (num_peaks))
            objlist = self.iqcalc.evaluate_peaks(peaks, data,
                                                 fwhm_radius=self.radius)

            num_candidates = len(objlist)
            if num_candidates == 0:
                raise Exception("Error evaluating bright peaks: no candidates found")

            self.logger.debug("Selecting from %d candidates..." % (num_candidates))
            height, width = data.shape
            results = self.iqcalc.objlist_select(objlist, width, height,
                                                 minfwhm=self.min_fwhm,
                                                 maxfwhm=self.max_fwhm,
                                                 minelipse=self.min_ellipse,
                                                 edgew=self.edgew)
            if len(results) == 0:
                raise Exception("No object matches selection criteria")

            # add back in offsets from cutout to result positions
            for qs in results:
                qs.x += x1
                qs.y += y1
                qs.objx += x1
                qs.objy += y1

        except Exception as e:
            msg = str(e)
            self.logger.error("Error finding object: %s" % (msg))
            raise e

        return results

    def make_report(self, image, qs):
        d = Bunch.Bunch()
        try:
            x, y = qs.objx, qs.objy
            equinox = float(image.get_keyword('EQUINOX', 2000.0))

            try:
                ra_deg, dec_deg = image.pixtoradec(x, y, coords='data')
                ra_txt, dec_txt = wcs.deg2fmt(ra_deg, dec_deg, 'str')

            except Exception as e:
                self.logger.warning("Couldn't calculate sky coordinates: %s" % (str(e)))
                ra_deg, dec_deg = 0.0, 0.0
                ra_txt = dec_txt = 'BAD WCS'

            # Calculate star size from pixel pitch
            try:
                header = image.get_header()
                ((xrot, yrot),
                 (cdelt1, cdelt2)) = wcs.get_xy_rotation_and_scale(header)

                starsize = self.iqcalc.starsize(qs.fwhm_x, cdelt1,
                                                qs.fwhm_y, cdelt2)
            except Exception as e:
                self.logger.warning("Couldn't calculate star size: %s" % (str(e)))
                starsize = 0.0

            rpt_x = x + self.pixel_coords_offset
            rpt_y = y + self.pixel_coords_offset

            # make a report in the form of a dictionary
            d.setvals(x = rpt_x, y = rpt_y,
                      ra_deg = ra_deg, dec_deg = dec_deg,
                      ra_txt = ra_txt, dec_txt = dec_txt,
                      equinox = equinox,
                      fwhm = qs.fwhm,
                      fwhm_x = qs.fwhm_x, fwhm_y = qs.fwhm_y,
                      ellipse = qs.elipse, background = qs.background,
                      skylevel = qs.skylevel, brightness = qs.brightness,
                      starsize = starsize,
                      time_local = time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime()),
                      time_ut = time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.gmtime()),
                      )
        except Exception as e:
            self.logger.error("Error making report: %s" % (str(e)))

        return d

    def prepare_mosaic(self, image, fov_deg, name=None, skew_limit=0.1):
        """Prepare a new (blank) mosaic image based on the pointing of
        the parameter image
        """
        header = image.get_header()
        ra_deg, dec_deg = header['CRVAL1'], header['CRVAL2']

        data_np = image.get_data()
        #dtype = data_np.dtype
        dtype = None

        # handle skew (differing rotation for each axis)?
        (rot_deg, cdelt1, cdelt2) = wcs.get_rotation_and_scale(header,
                                                               skew_threshold=skew_limit)
        self.logger.debug("image0 rot=%f cdelt1=%f cdelt2=%f" % (
            rot_deg, cdelt1, cdelt2))

        # TODO: handle differing pixel scale for each axis?
        px_scale = math.fabs(cdelt1)
        cdbase = [numpy.sign(cdelt1), numpy.sign(cdelt2)]

        self.logger.debug("creating blank image to hold mosaic")

        mosaic_img = dp.create_blank_image(ra_deg, dec_deg,
                                           fov_deg, px_scale,
                                           rot_deg,
                                           cdbase=cdbase,
                                           logger=self.logger,
                                           pfx='mosaic',
                                           dtype=dtype)

        if name is not None:
            mosaic_img.set(name=name)
        imname = mosaic_img.get('name', image.get('name', "NoName"))

        # avoid making a thumbnail of this
        mosaic_img.set(nothumb=True, path=None)

        header = mosaic_img.get_header()
        (rot, cdelt1, cdelt2) = wcs.get_rotation_and_scale(header,
                                                           skew_threshold=skew_limit)
        self.logger.debug("mosaic rot=%f cdelt1=%f cdelt2=%f" % (
            rot, cdelt1, cdelt2))

        mosaic_img.set(nothumb=True)

        return mosaic_img

    def ingest_images(self, images, mosaic_img, merge=False,
                      allow_expand=True, expand_pad_deg=0.010):

        self.logger.debug("ingesting images")
        mosaic_img.mosaic_inline(images,
                                 bg_ref=None,
                                 trim_px=None,
                                 update_minmax=False,
                                 merge=merge,
                                 allow_expand=allow_expand,
                                 expand_pad_deg=expand_pad_deg,
                                 suppress_callback=True)
        self.logger.debug("images digested")


    def load_flat(self, ccd_id, filter_name):
        self.logger.info("loading flat ccd_id=%d filter='%s'" % (
            ccd_id, filter_name))
        flat_file = os.path.join(self.flat_dir, filter_name,
                                 "FLAT-%03d.fits[1]" % ccd_id)
        self.log("attempting to load flat '%s'" % (flat_file))
        image = AstroImage.AstroImage(logger=self.logger)
        image.load_file(flat_file, memmap=False)

        data_np = image.get_data()

        # Adjust for how superflats are standardized in storage
        # (channels L-R)
        if hsc_ccd_data[ccd_id].swapxy:
            data_np = data_np.swapaxes(0, 1)
        if hsc_ccd_data[ccd_id].flipv:
            data_np = numpy.flipud(data_np)
        if hsc_ccd_data[ccd_id].fliph:
            data_np = numpy.fliplr(data_np)

        self.flat[ccd_id] = data_np

    def preprocess(self, image, dr):
        filter_name = image.get_keyword('FILTER01').strip().upper()
        ccd_id = int(image.get_keyword('DET-ID'))

        dr.remove_overscan(image, sub_bias=self.sub_bias)

        if self.use_flat:
            # flat field this piece, if flat provided
            if filter_name != self.flat_filter:
                self.log("Change of filter detected--resetting flats")
                self.flat = {}
                self.flat_filter = filter_name

            try:
                if not ccd_id in self.flat:
                    self.load_flat(ccd_id, filter_name)

                flat = self.flat[ccd_id]

                data_np = image.get_data()
                if data_np.shape == flat.shape:
                    data_np /= flat

                else:
                    raise ValueError("flat for CCD %d shape %s does not match image CCD shape %s" % (ccd_id, flat.shape, data_np.shape))

                header = {}
                image = dp.make_image(data_np, image, header)

            except Exception as e:
                self.logger.warning("Error applying flat field: %s" % (str(e)))

        return image

    def mosaic_some(self, paths, mosaic_img, dr=None, merge=False):
        images = []
        #self.log("paths are %s" % (str(paths)))
        for path in paths:
            self.logger.info("reading %s ..." % (path))
            dirname, filename = os.path.split(path)
            self.log("reading %s ..." % (filename))

            image = AstroImage.AstroImage(logger=self.logger)
            image.load_file(path, memmap=False)

            if dr is not None:
                image = self.preprocess(image, dr)
            images.append(image)

        self.ingest_images(images, mosaic_img, merge=merge)

        num_groups, self.num_groups = self.num_groups, self.num_groups-1
        ## if num_groups == 1:
        ##     self._update_gui(0, mosaic_img, self.total_files, self.start_time)

    def __update_gui(self, res, mosaic_img, total_files, start_time):
        self.fv.gui_do(self._update_gui, res, mosaic_img, total_files,
                       start_time)

    def _update_gui(self, res, mosaic_img, total_files, start_time):
        end_time = time.time()
        elapsed = end_time - start_time
        self.log("mosaiced %d files in %.3f sec" % (
            total_files, elapsed))
        imname = mosaic_img.get('name', 'mosaic')
        #self.fv.gui_do(mosaic_img.make_callback, 'modified')
        self.fv.add_image(imname, mosaic_img, chname='GView')
        self.log("done mosaicing")

    def mosaic(self, paths, mosaic_img, name='mosaic', fov_deg=0.2,
               num_threads=6, dr=None, merge=False):

        self.total_files = len(paths)
        if self.total_files == 0:
            return

        ingest_count = 0
        self.start_time = time.time()

        groups = dp.split_n(paths, num_threads)
        self.num_groups = len(groups)
        self.logger.info("len groups=%d" % (self.num_groups))
        ## tasks = []
        for group in groups:
            ## self.fv.nongui_do(self.mosaic_some, group, mosaic_img,
            ##                   dr=dr, merge=merge)
            ## tasks.append(Task.FuncTask(self.mosaic_some, (group, mosaic_img),
            ##                            dict(dr=dr, merge=merge),
            ##                            logger=self.logger))
            self.mosaic_some(group, mosaic_img, dr=dr, merge=merge)
        ## t = Task.ConcurrentAndTaskset(tasks)

        ## t.register_callback(self.__update_gui, args=[mosaic_img,
        ##                                              self.total_files,
        ##                                              self.start_time])
        ## t.init_and_start(self.fv)
        self._update_gui(0, mosaic_img, self.total_files, self.start_time)

#END
