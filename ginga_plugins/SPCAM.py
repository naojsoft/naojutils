#
# SPCAM.py -- Suprime-Cam quick look plugin for Ginga FITS viewer
#
# Eric Jeschke (eric@naoj.org)
#
# Copyright (c) 2014-2015  National Astronomical Observatory of Japan.
#   All rights reserved.
# This is open-source software licensed under a BSD license.
# Please see the file LICENSE.txt for details.
#
"""
A plugin for the Ginga scientific image viewer for quick look viewing and
mosaicing Suprime-Cam data.

Installation:
  $ mkdir $HOME/.ginga
  $ mkdir $HOME/.ginga/plugins
  $ cp SPCAM.py $HOME/.ginga/plugins

Running:
  $ ginga [other options] --plugins=SPCAM ...

NOTE: this requires the "naojutils" module, available at
          https://github.com/naojsoft/naojutils
"""
import os, re, glob
import threading, Queue

from ginga import AstroImage
from ginga.misc.plugins import Mosaic
from ginga.misc import Widgets, Bunch, Future
from ginga.util import dp

# You naojutils to run this plugin.
#   Get it here--> https://github.com/naojsoft/naojutils
from naoj.frame import Frame
from naoj import spcam


class SPCAM(Mosaic.Mosaic):

    def __init__(self, fv, fitsimage):
        # superclass defines some variables for us, like logger
        super(SPCAM, self).__init__(fv, fitsimage)

        # Set preferences for destination channel
        prefs = self.fv.get_preferences()
        self.settings = prefs.createCategory('plugin_SPCAM')
        self.settings.setDefaults(annotate_images=False, fov_deg=0.72,
                                  match_bg=False, trim_px=0,
                                  merge=True, num_threads=4,
                                  drop_creates_new_mosaic=True,
                                  mosaic_hdus=False, skew_limit=0.1,
                                  allow_expand=False, expand_pad_deg=0.01,
                                  use_flats=False, flat_dir='',
                                  mosaic_new=True, make_thumbs=False,
                                  reuse_image=False)
        self.settings.load(onError='silent')

        self.queue = Queue.Queue()
        self.current_exp_num = 0

        # map of exposures -> list of paths
        self.exp_pathmap = {}

        # define mosaic preprocessing step for SPCAM
        self.set_preprocess(self.mangle_image)

        self.fv.enable_callback('file-notify')
        self.fv.add_callback('file-notify', self.file_notify_cb)

        self.timer = self.fv.get_timer()
        self.timer.add_callback('expired', self.process_frames)
        self.process_interval = 0.2

        self.dr = spcam.SuprimeCamDR(logger=self.logger)

        self.mosaic_chname = 'SPCAM_Online'
        # For flat fielding
        self.flat = {}


    def build_gui(self, container):
        super(SPCAM, self).build_gui(container)

        vbox = self.w.vbox

        fr = Widgets.Frame("Flats")

        captions = [
            ("Use flats", 'checkbutton'),
            ("Flat dir:", 'label', 'flat_dir', 'entry'),
            ("Load Flats", 'button'),
            ]
        w, b = Widgets.build_info(captions)
        self.w.update(b)

        b.flat_dir.set_length(512)
        b.flat_dir.set_text(self.settings.get('flat_dir', ''))
        b.load_flats.add_callback('activated', self.load_flats_cb)
        b.use_flats.set_tooltip("Flat field tiles as they arrive")
        use_flats = self.settings.get('use_flats', False)
        b.use_flats.set_state(use_flats)
        b.flat_dir.set_tooltip("Directory containing flat field tiles")
        b.load_flats.set_tooltip("Load flat field tiles from directory")

        fr.set_widget(w)
        vbox.add_widget(fr, stretch=0)


    def instructions(self):
        self.tw.set_text("""Frames will be mosaiced as they arrive.""")

    def get_exp_num(self, frame):
        exp_num = (frame.number // self.dr.num_frames) * self.dr.num_frames
        return exp_num

    def get_latest_frames(self, pathlist):
        new_frlist = []
        new_exposure = False
        exposures = set([])

        for path in pathlist:
            info = self.fv.get_fileinfo(path)
            self.logger.info("getting path")
            path = info.filepath
            self.logger.info("path is %s" % (path))

            frame = Frame(path=path)
            # if not an instrument frame then drop it
            if frame.inscode != self.dr.inscode:
                continue

            # calculate exposure number
            #exp_num = (frame.number // self.dr.num_frames) * self.dr.num_frames
            exp_num = self.get_exp_num(frame)

            # add paths to exposure->paths map
            exp_id = Frame(path=path)
            exp_id.number = exp_num
            exp_frid = str(exp_id)
            bnch = Bunch.Bunch(paths=set([]))
            exp_bnch = self.exp_pathmap.setdefault(exp_frid, bnch)
            exp_bnch.paths.add(path)
            if len(exp_bnch.paths) == 1:
                exp_bnch.setvals(typical=path, added_to_contents=False)
            exposures.add(exp_frid)

            # if frame number doesn't belong to current exposure
            # then drop it
            if frame.number < self.current_exp_num:
                continue

            if exp_num > self.current_exp_num:
                # There is a new exposure
                self.current_exp_num = exp_num
                new_frlist = [ path ]
                new_exposure = True
            else:
                new_frlist.append(path)

        return (new_frlist, new_exposure, exposures)


    def mk_loader(self, bnch):
        def load_mosaic(filepath):
            paths = list(bnch.paths)
            return self.mosaic(paths, new_mosaic=True)
        return load_mosaic

    def add_to_contents(self, exposures):
        try:
            #pluginInfo = self.fv.gpmon.getPluginInfo('Contents')

            for exp_frid in exposures:
                # get the information about this exposure
                exp_bnch = self.exp_pathmap[exp_frid]
                if exp_bnch.added_to_contents:
                    continue

                self.logger.debug("Exposure '%s' not yet added to contents" % (
                    exp_frid))
                # load the representative image
                image = AstroImage.AstroImage(logger=self.logger)
                # TODO: is this load even necessary?  Would be good
                # if we could just load the headers
                path = exp_bnch.typical
                image.load_file(path)
                # make a new loader that will load the mosaic and attach
                # it to this image as the loader
                image_loader = self.mk_loader(exp_bnch)

                self.logger.debug("making future")
                future = Future.Future()
                future.freeze(image_loader, path)
                image.set(loader=image_loader, name=exp_frid,
                          image_future=future, path=path)

                exp_bnch.added_to_contents = True

                # add this to the contents pane
                ## self.logger.debug("calling into Contents plugin")
                ## self.fv.gui_do(pluginInfo.obj.add_image, self.fv,
                ##                self.mosaic_chname, image)
                ## self.fv.gui_do(self.fv.add_image, exp_frid, image,
                ##                self.mosaic_chname)
                self.fv.gui_do(self.fv.advertise_image,
                               self.mosaic_chname, image)

        except Exception as e:
            self.logger.warn("'Contents' plugin not available: %s" % (str(e)))
            return

    def process_frames(self, timer):
        self.logger.info("processing queued frames")

        # Get all files stored in the queue
        paths = []
        while True:
            try:
                path = self.queue.get(block=False)
                paths.append(path)
            except Queue.Empty:
                break

        self.logger.debug("paths=%s" % str(paths))
        if len(paths) == 0:
            return

        self.logger.info("adding to contents: %s" % (str(paths)))
        try:
            paths, new_mosaic, exposures = self.get_latest_frames(paths)

            self.logger.info("adding to contents")
            self.add_to_contents(exposures)
        except Exception as e:
            self.logger.error("error adding to contents: %s" % (str(e)))

        mosaic_new = self.settings.get('mosaic_new', False)
        self.logger.info("mosaic_new=%s new_mosaic=%s" % (mosaic_new,
                                                          new_mosaic))
        if self.gui_up and mosaic_new:
            self.logger.info("mosaicing %s" % (str(paths)))
            self.mosaic(paths, new_mosaic=new_mosaic)

    def file_notify_cb(self, fv, path):
        self.logger.debug("file notify: %s" % (path))
        self.queue.put(path)
        self.timer.cond_set(self.process_interval)

    def drop_cb(self, canvas, paths):
        self.logger.info("files dropped: %s" % str(paths))
        for path in paths:
            self.queue.put(path)
        self.timer.cond_set(self.process_interval)
        return True

    def mangle_image(self, image):
        d = self.dr.get_regions(image)
        header = {}
        data_np = image.get_data()

        # subtract overscan region
        result = self.dr.subtract_overscan_np(data_np, d,
                                              header=header)

        # flat field this piece, if flat provided
        do_flat = self.w.use_flats.get_state()
        if do_flat and (len(self.flat) > 0):
            try:
                ccd_id = int(image.get_keyword('DET-ID'))
                result /= self.flat[ccd_id]
            except Exception as e:
                self.logger.warn("Error applying flat field: %s" % (str(e)))

        newimage = dp.make_image(result, image, header)
        return newimage

    def _load_flats(self, datadir):
        # TODO: parallelize this
        self.fv.assert_nongui_thread()

        path_glob = os.path.join(datadir, '*-*.fits')
        d = {}
        paths = glob.glob(path_glob)
        if len(paths) != self.dr.num_ccds:
            self.fv.gui_do(self.fv.show_error, "Number of flat files (%d) does not match number of CCDs (%d)" % (
                len(paths), self.dr.num_frames))
            return

        self.update_status("Loading flats...")
        self.init_progress()

        self.total_files = max(1, len(paths))
        self.ingest_count = 0

        num_threads = self.settings.get('num_threads', 4)
        groups = self.split_n(paths, num_threads)
        for group in groups:
            self.fv.nongui_do(self._load_some, d, group)

    def _load_some(self, d, paths):
        for path in paths:
            match = re.match(r'^.+\-(\d+)\.fits$', path)
            if match:
                ccd_id = int(match.group(1))
                image = AstroImage.AstroImage(logger=self.logger)
                image.load_file(path)

                with self.lock:
                    d[ccd_id] = image.get_data()
                    self.ingest_count += 1
                    count = self.ingest_count

                self.update_progress(float(count)/self.total_files)

        if count == self.total_files:
            self.flat = d
            self.end_progress()
            self.update_status("Flats loaded.")

    def load_flats_cb(self, w):
        dirpath = self.w.flat_dir.get_text().strip()
        # Save the setting
        self.settings.set(flat_dir=dirpath)

        self.fv.nongui_do(self._load_flats, dirpath)

    def __str__(self):
        return 'spcam'


#END
