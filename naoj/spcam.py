#
# spcam.py -- Suprime-Cam data processing routines
#
# Eric Jeschke (eric@naoj.org)
#
# This is open-source software licensed under a BSD license.
# Please see the file LICENSE.txt for details.
#
import os
import re, glob
import time

import numpy

from ginga import AstroImage
from ginga.misc import Bunch, log
from ginga.util import dp
from ginga.util import mosaic, wcs

from .frame import Frame

class SuprimeCamDR(object):

    def __init__(self, logger=None):
        super(SuprimeCamDR, self).__init__()

        if logger is None:
            logger = log.get_logger(level=20, log_stderr=True)
        self.logger = logger

        self.pfx = 'S'
        self.num_ccds = 10
        self.num_frames = 10
        self.inscode = 'SUP'
        self.fov = 0.72

        self.frameid_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # SPCAM keywords that should be added to the primary HDU
        self.prihdr_kwds = [
            'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'BZERO',
            'BSCALE', 'BUNIT', 'BLANK', 'DATE-OBS', 'UT', 'UT-STR', 'UT-END',
            'HST', 'HST-STR', 'HST-END', 'LST', 'LST-STR', 'LST-END', 'MJD',
            'TIMESYS', 'MJD-STR', 'MJD-END', 'ZD-STR', 'ZD-END', 'SECZ-STR',
            'SECZ-END', 'AIRMASS', 'AZIMUTH', 'ALTITUDE', 'PROP-ID', 'OBSERVER',
            'EXP-ID', 'DATASET', 'OBS-MOD', 'OBS-ALOC', 'DATA-TYP', 'OBJECT',
            'RA', 'DEC', 'RA2000', 'DEC2000', 'OBSERVAT', 'TELESCOP', 'FOC-POS',
            'TELFOCUS', 'FOC-VAL', 'FILTER01', 'EXPTIME', 'INSTRUME', 'INS-VER',
            'WEATHER', 'SEEING', 'ADC-TYPE', 'ADC-STR', 'ADC-END', 'INR-STR',
            'INR-END', 'DOM-WND', 'OUT-WND', 'DOM-TMP', 'OUT-TMP', 'DOM-HUM',
            'OUT-HUM', 'DOM-PRS', 'OUT-PRS', 'EXP1TIME', 'COADD', 'M2-POS1',
            'M2-POS2', 'M2-POS3', 'M2-ANG1', 'M2-ANG2', 'M2-ANG3', 'AUTOGUID',
            'COMMENT', 'INST-PA', 'EQUINOX',
            ]

        # SPCAM keywords that should be added to the image HDUs
        self.imghdr_kwds = [
            'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'BZERO',
            'BSCALE', 'BUNIT', 'BLANK', 'FRAMEID', 'EXP-ID', 'DETECTOR', 'DET-ID',
            'DET-A01', 'DET-P101', 'DET-P201', 'DET-TMP', 'DET-TMED', 'DET-TMIN',
            'DET-TMAX', 'GAIN', 'EFP-MIN1', 'EFP-RNG1', 'EFP-MIN2', 'EFP-RNG2',
            'PRD-MIN1', 'PRD-RNG1', 'PRD-MIN2', 'PRD-RNG2', 'BIN-FCT1', 'BIN-FCT2',
            'DET-VER', 'S_UFNAME', 'S_FRMPOS', 'S_BCTAVE', 'S_BCTSD', 'S_AG-OBJ',
            'S_AG-RA', 'S_AG-DEC', 'S_AG-EQN', 'S_AG-X', 'S_AG-Y', 'S_AG-R',
            'S_AG-TH', 'S_ETMED', 'S_ETMAX', 'S_ETMIN', 'S_XFLIP', 'S_YFLIP',
            'S_M2OFF1', 'S_M2OFF2', 'S_M2OFF3', 'S_DELTAZ', 'S_DELTAD', 'S_SENT',
            'S_GAIN1', 'S_GAIN2', 'S_GAIN3', 'S_GAIN4', 'S_OSMN11', 'S_OSMX11',
            'S_OSMN21', 'S_OSMX21', 'S_OSMN31', 'S_OSMX31', 'S_OSMN41', 'S_OSMX41',
            'S_OSMN12', 'S_OSMX12', 'S_OSMN22', 'S_OSMX22', 'S_OSMN32', 'S_OSMX32',
            'S_OSMN42', 'S_OSMX42', 'S_EFMN11', 'S_EFMX11', 'S_EFMN21', 'S_EFMX21',
            'S_EFMN31', 'S_EFMX31', 'S_EFMN41', 'S_EFMX41', 'S_EFMN12', 'S_EFMX12',
            'S_EFMN22', 'S_EFMX22', 'S_EFMN32', 'S_EFMX32', 'S_EFMN42', 'S_EFMX42',
            'EQUINOX',  'CRVAL1',  'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2',
            'LONGPOLE', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'WCS-ORIG',
            'RADECSYS', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
            ]

    def get_exp_num(self, frameid):
        frame = Frame(frameid)
        exp_num = (frame.number // self.num_frames) * self.num_frames
        return exp_num

    def get_file_list(self, path):
        frame = Frame(path)
        exp_num = self.get_exp_num(path)
        nums = map(lambda off: exp_num+off, self.frameid_offsets)
        res = []
        for num in nums:
            frame.number = num
            res.append(os.path.join(frame.directory, str(frame)+'.fits'))
        return res

    def get_images(self, path):
        filelist = self.get_file_list(path)
        res = []
        for path in filelist:
            img = AstroImage.AstroImage(logger=self.logger)
            img.load_file(path)
            res.append(img)
        return res

    def get_regions(self, image):
        """Extract the keywords defining the overscan and effective pixel
        regions in a SPCAM image.  The data is returned in a dictionary of
        bunches.  The keys of the dictionary are the channel numbers, plus
        'image'.
        """
        wd, ht = image.get_size()
        d = {}
        xcut = 0
        newwd = 0
        l = []
        for channel in (1, 2, 3, 4):
            base = self.pfx + '_EF'
            efminx = int(image.get_keyword("%sMN%d1" % (base, channel))) - 1
            efmaxx = int(image.get_keyword("%sMX%d1" % (base, channel))) - 1
            efminy = int(image.get_keyword("%sMN%d2" % (base, channel))) - 1
            efmaxy = int(image.get_keyword("%sMX%d2" % (base, channel))) - 1
            base = self.pfx + '_OS'
            osminx = int(image.get_keyword("%sMN%d1" % (base, channel))) - 1
            osmaxx = int(image.get_keyword("%sMX%d1" % (base, channel))) - 1
            osminy = int(image.get_keyword("%sMN%d2" % (base, channel))) - 1
            osmaxy = int(image.get_keyword("%sMX%d2" % (base, channel))) - 1
            xcut += osmaxx - osminx + 1
            newwd += efmaxx + 1 - efminx

            gain = float(image.get_keyword("%s_GAIN%d" % (self.pfx, channel)))
            d[channel] = Bunch.Bunch(
                efminx=efminx, efmaxx=efmaxx, efminy=efminy, efmaxy=efmaxy,
                osminx=osminx, osmaxx=osmaxx, osminy=osminy, osmaxy=osmaxy,
                gain=gain)
            l.append(d[channel])

        # figure out starting x position of channel within image
        l.sort(cmp=lambda x, y: x.efmaxx - y.efmaxx)
        startposx = 0
        for ch in l:
            ch.setvals(startposx=startposx)
            startposx += ch.efmaxx + 1 - ch.efminx

        ycut = osmaxy - osminy + 1
        newht = efmaxy + 1 - efminy
        d['image'] = Bunch.Bunch(xcut=xcut, ycut=ycut,
                                 newwd=newwd, newht=newht)

        return d


    def subtract_overscan_np(self, data_np, d, header=None):
        """Subtract the median bias calculated from the overscan regions
        from a SPCAM image data array.  The resulting image is trimmed to
        remove the overscan regions.

        Parameters
        ----------
        data_np: numpy array
            a 2D data array of pixel values
        d: dict
            a dictionary of information about the overscan and effective
            pixel regions as returned by get_regions().

        Returns:
        out: numpy array
            a new, smaller array with the result data
        """

        # create new output array the size of the sum of the image
        # effective pixels
        info = d['image']
        newwd, newht = info.newwd, info.newht
        self.logger.debug("effective pixel size %dx%d" % (newwd, newht))
        out = numpy.empty((newht, newwd), dtype=float)
        if header is not None:
            header['NAXIS1'] = newwd
            header['NAXIS2'] = newht

        # original image size
        ht, wd = data_np.shape[:2]

        for channel in (1, 2, 3, 4):
            #print "processing channel %d" % (channel)
            ch = d[channel]

            # get median of each row in overscan area for this channel
            ovsc_median = numpy.median(data_np[ch.efminy:ch.efmaxy+1,
                                               ch.osminx:ch.osmaxx+1], axis=1)
            # calculate size of effective pixels area for this channel
            efwd = ch.efmaxx + 1 - ch.efminx
            efht = ch.efmaxy + 1 - ch.efminy
            len_ovsc = ovsc_median.shape[0]

            assert len_ovsc == efht, \
                   ValueError("median array len (%d) doesn't match effective pixel len (%d)" % (
                len_ovsc, efht))

            ovsc_median = ovsc_median.reshape((efht, 1))
            ovsc_median = numpy.repeat(ovsc_median, efwd, axis=1)

            j = ch.startposx

            # Cut effective pixel region into output array
            xlo, xhi, ylo, yhi = j, j + efwd, 0, efht
            out[ylo:yhi, xlo:xhi] = data_np[ch.efminy:ch.efmaxy+1,
                                            ch.efminx:ch.efmaxx+1] - ovsc_median
            # Subtract overscan medians
            #out[ylo:yhi, xlo:xhi] -= ovsc_median

            # Update header for effective regions
            if header is not None:
                base = self.pfx + '_EF'
                header["%sMN%d1" % (base, channel)] = xlo + 1
                header["%sMX%d1" % (base, channel)] = xhi + 1
                header["%sMN%d2" % (base, channel)] = ylo + 1
                header["%sMX%d2" % (base, channel)] = yhi + 1

        return out


    def make_flat(self, flatlist, bias=None, flat_norm=None,
                  logger=None):

        flats = []
        for path in flatlist:
            image = AstroImage.AstroImage(logger=logger)
            image.load_file(path)

            data_np = image.get_data()
            # TODO: subtract optional bias image

            # subtract overscan and trim
            d = self.get_regions(image)
            header = {}
            newarr = self.subtract_overscan_np(data_np, d,
                                               header=header)

            flats.append(newarr)

        # Take the median of the individual frames
        flat = numpy.median(numpy.array(flats), axis=0)
        #print flat.shape

        # Normalize flat, if normalization term provided
        if flat_norm is not None:
            flat = flat / flat_norm

        img_flat = dp.make_image(flat, image, header)
        return img_flat


    def make_flat_tiles(self, datadir, explist, output_pfx='flat',
                        output_dir=None):

        # Get the median values for each CCD image
        flats = []
        for i in xrange(self.num_frames):
            flatlist = []
            for exp in explist:
                path = os.path.join(datadir, exp.upper()+'.fits')
                if not os.path.exists(path):
                    continue
                frame = Frame(path=path)
                frame.number += i
                path = os.path.join(datadir, str(frame)+'.fits')
                if not os.path.exists(path):
                    continue
                flatlist.append(path)

            if len(flatlist) > 0:
                flats.append(self.make_flat(flatlist))

        # Normalize the flats
        # TODO: can we use a running median to speed this up without
        # losing much precision?
        # flatarr = numpy.array([ image.get_data() for image in flats ])
        # mval = numpy.median(flatarr.flat)
        flatarr = numpy.array([ numpy.median(image.get_data())
                                for image in flats ])
        mval = numpy.mean(flatarr)

        d = {}
        for image in flats:
            flat = image.get_data()
            flat /= mval
            # no zero divisors
            flat[flat == 0.0] = 1.0
            ccd_id = int(image.get_keyword('DET-ID'))

            if output_dir is None:
                d[ccd_id] = image
            else:
                # write the output file
                name = '%s-%d.fits' % (output_pfx, ccd_id)
                outfile = os.path.join(output_dir, name)
                d[ccd_id] = outfile
                self.logger.debug("Writing output file: %s" % (outfile))
                try:
                    os.remove(outfile)
                except OSError:
                    pass
                image.save_as_file(outfile)

        return d


    def get_flat_name(self, pfx, image):
        hdr = image.get_header()
        kwds = dict([ (kwd, hdr[kwd]) for kwd in ('OBJECT', 'FILTER01',
                                                  'DATE-OBS', 'UT-STR') ])
        match = re.match(r'^(\d\d):(\d\d):(\d\d)\.\d+$', kwds['UT-STR'])
        ut = ''.join(match.groups())
        match = re.match(r'^(\d\d\d\d)\-(\d+)\-(\d+)$', kwds['DATE-OBS'])
        date = ''.join(match.groups())
        fname = '%s-flat-%s-%s-%s-%s.fits' % (pfx, date, ut,
                                              kwds['OBJECT'],
                                              kwds['FILTER01'])
        return fname, kwds


    def make_flat_tiles_exp(self, datadir, expstart, num_exp,
                            output_pfx='flat', output_dir=None):

        path = os.path.join(datadir, expstart.upper()+'.fits')

        # make a list of all the exposure ids
        explist = []
        for i in xrange(num_exp):
            frame = Frame(path=path)
            frame.number += i * self.num_frames
            explist.append(str(frame))

        d = self.make_flat_tiles(datadir, explist,
                                 output_pfx=output_pfx,
                                 output_dir=output_dir)
        return d


    def load_flat_tiles(self, datadir):

        path_glob = os.path.join(datadir, '*-*.fits')
        d = {}
        for path in glob.glob(path_glob):
            match = re.match(r'^.+\-(\d+)\.fits$', path)
            if match:
                ccd_id = int(match.group(1))
                image = AstroImage.AstroImage(logger=self.logger)
                image.load_file(path)

                d[ccd_id] = image.get_data()

        return d


    def prepare_mosaic(self, image, fov_deg, skew_limit=0.1):
        """Prepare a new (blank) mosaic image based on the pointing of
        the parameter image
        """
        header = image.get_header()
        ra_deg, dec_deg = header['CRVAL1'], header['CRVAL2']

        data_np = image.get_data()

        (rot_deg, cdelt1, cdelt2) = wcs.get_rotation_and_scale(header,
                                                               skew_threshold=skew_limit)
        self.logger.debug("image0 rot=%f cdelt1=%f cdelt2=%f" % (
            rot_deg, cdelt1, cdelt2))

        # TODO: handle differing pixel scale for each axis?
        px_scale = numpy.fabs(cdelt1)
        cdbase = [numpy.sign(cdelt1), numpy.sign(cdelt2)]
        #cdbase = [1, 1]

        img_mosaic = dp.create_blank_image(ra_deg, dec_deg,
                                           fov_deg, px_scale,
                                           rot_deg,
                                           cdbase=cdbase,
                                           logger=self.logger,
                                           pfx='mosaic')

        # TODO: fill in interesting/select object headers from seed image
        return img_mosaic

    def remove_overscan(self, img):
        d = self.get_regions(img)
        header = {}
        new_data_np = self.subtract_overscan_np(img.get_data(), d,
                                                header=header)
        img.set_data(new_data_np)
        img.update_keywords(header)

    def make_quick_mosaic(self, path):

        # get the list of files making up this exposure
        files = self.get_file_list(path)

        img = AstroImage.AstroImage(logger=self.logger)
        img.load_file(files[0])

        img_mosaic = self.prepare_mosaic(img, self.fov)
        self.remove_overscan(img)
        img_mosaic.mosaic_inline([img])

        time_start = time.time()
        t1_sum = 0.0
        t2_sum = 0.0
        t3_sum = 0.0
        for filen in files[1:]:
            time_t1 = time.time()
            img = AstroImage.AstroImage(logger=self.logger)
            img.load_file(filen)
            time_t2 = time.time()
            t1_sum += time_t2 - time_t1
            self.remove_overscan(img)
            time_t3 = time.time()
            t2_sum += time_t3 - time_t2
            img_mosaic.mosaic_inline([img], merge=True, allow_expand=False,
                                     update_minmax=False)
            time_t4 = time.time()
            t3_sum += time_t4 - time_t3

        time_end = time.time()
        time_total = time_end - time_start

        print ("total time: %.2f t1=%.3f t2=%.3f t3=%.3f" % (
            time_total, t1_sum, t2_sum, t3_sum))
        return img_mosaic

    def make_multi_hdu(self, images, compress=False):
        """
        Pack a group of separate FITS files (a single exposure) into one
        FITS file with one primary HDU with no data and 10 image HDUs.

        Parameters
        ----------
        images : list of AstroImage objects
        compress : bool (optional)
            if True, will try to Rice-compress the image HDUs.  Note that
            this slows down the process considerably.  Default: False
        """

        import astropy.io.fits as pyfits

        fitsobj = pyfits.HDUList()

        i = 0
        hdus = {}

        for image in images:
            header = image.get_header()
            if i == 0:
                # prepare primary HDU header
                hdu = pyfits.PrimaryHDU()
                prihdr = hdu.header
                for kwd in header.keys():
                    if kwd.upper() in self.prihdr_kwds:
                        card = header.get_card(kwd)
                        val, comment = card.value, card.comment
                        prihdr[kwd] = (val, comment)

                fitsobj.append(hdu)

            # create each image HDU
            data = numpy.copy(image.get_data().astype(numpy.uint16))

            if not compress:
                hdu = pyfits.ImageHDU(data=data)
            else:
                hdu = pyfits.CompImageHDU(data=data,
                                          compression_type='RICE_1')

            for kwd in header.keys():
                if kwd.upper() in self.imghdr_kwds:
                    card = header.get_card(kwd)
                    val, comment = card.value, card.comment
                    hdu.header[kwd] = (val, comment)

            # add CHECKSUM and DATASUM keywords
            if not compress:
                hdu.add_checksum()

            det_id = int(hdu.header['DET-ID'])
            hdus[det_id] = hdu
            i += 1

        # stack HDUs in detector ID order
        for i in xrange(self.num_ccds):
            fitsobj.append(hdus[i])

        # fix up to FITS standard as much as possible
        fitsobj.verify('silentfix')

        return fitsobj


    def step2(self, image):
        """
        Corresponds to step 2 in the SPCAM data reduction instructions.

        Takes an image and removes the overscan regions.  In the process
        it also subtracts the bias median calculated from the overscan
        regions.
        """
        d = self.get_regions(image)
        header = {}
        data_np = image.get_data()

        result = self.subtract_overscan_np(data_np, d, header=header)

        newimage = dp.make_image(result, image, header)
        return newimage


    def step3(self, image, flat):
        """
        Corresponds to step 3 in the SPCAM data reduction instructions.

        Divides an image by a flat and returns a new image.
        """

        data_np = image.get_data()
        flat_np = flat.get_data()

        result = data_np / flat_np

        newimage = dp.make_image(result, image, {})
        return newimage


#END
