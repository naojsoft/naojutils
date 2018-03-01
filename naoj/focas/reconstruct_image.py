#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Shinobu Ozaki
"""
USAGE: python reconstruct_image.py right.fits left.fits -o output.fits

    right.fits: right image file name (smaller file number)
    left.fits: left image file name (larger file number)
    output_fits: output file name

"""
import sys, os
import re
import numpy as np
from astropy.io import fits

# local imports
import biassub as bs

version = "20180226.0"

# this will hold region data
regions = {}

def read_region_file(region_file):
    # Reading the region file
    with open(region_file, 'r') as in_f:
        lines = in_f.readlines()

    # Parsing the region data
    regdata = np.zeros((23, 4), dtype='float')
    i = 0
    for line in lines:
        data = re.split('[(),]', line)
        if data[0] == "box":
            regdata[i, :] = [float(data[1]), float(data[2]),
                             float(data[3]), float(data[4])]
            i = i + 1

    return regdata

def reconstruct_image(fitsfile_ch1, fitsfile_ch2, region_data=None):
    global regions

    # subtract bias and combine channels
    hdl = bs.bias_subtraction(fitsfile_ch1, fitsfile_ch2)

    scidata = hdl[0].data
    # Getting the binning information
    hdr = hdl[0].header
    binfac1 = hdr['BIN-FCT1'] # X direction on DS9
    binfac2 = hdr['BIN-FCT2'] # Y direction on DS9
    if binfac1 not in (1, 2, 4):
        raise ValueError("X binning factor is not 1, 2 or 4: BIN-FCT1=%d" % (
            binfac1))

    reg = region_data
    if reg is None:
        reg = regions[int(binfac1)]

    # extract data from regions
    # 最初のボックスの縦横幅を全ての領域の縦横幅にしているので注意。
    # numpy配列では (Y,X) のようになる。
    xw, yw = int(reg[1, 2]), int(reg[1, 3])

    # Factor 'int(4/binfac1)' is to match Y scale to X scale in the
    # reconstructed image.
    bindata = np.zeros((reg.shape[0] * int(4 / binfac1), xw))
    for j in range(reg.shape[0]):
        xs = int(reg[j, 0] - reg[j, 2] / 2.0)
        ys = int(reg[j, 1] - reg[j, 3] / 2.0)
        for i in range(ys, ys + yw):
            bindata[j * int(4 / binfac1), :] = \
                     bindata[j * int(4 / binfac1), :] + scidata[i, xs:xs + xw]
        for k in range(int(4 / binfac1) - 1):
            bindata[j * int(4 / binfac1) + k + 1, :] = \
                     bindata[j * int(4 / binfac1), :]


    # Creating an output fits file
    newhdu = fits.PrimaryHDU(data=bindata)
    hdulst = fits.HDUList([newhdu])

    # Transfering FITS keywords
    transfer_kwds = ['RADECSYS', 'EQUINOX', 'DATE-OBS', 'CTYPE1', 'CTYPE2',
                     'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CUNIT1', 'CUNIT2',
                     #'CDELT1', 'CDELT2', 'PC001001', 'PC001002',
                     #'PC002001', 'PC002002', 'LONGPOLE'
                     ]

    newhdr = hdulst[0].header
    for kwd in transfer_kwds:
        newhdr[kwd] = hdr[kwd]

    # Extraction rotation matrix from CD matrix
    cd = np.matrix([[hdr['CD1_1'], hdr['CD1_2']], [hdr['CD2_1'], hdr['CD2_2']]])
    det = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    cd_rot = cd / np.sqrt(det)

    # Rotation matrix of the IFU field with respect to hte FOCAS field
    theta = 21.38 # degree
    theta = np.deg2rad(theta)
    rot = np.matrix([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    # Creating the new CD matrix for the reconstruction image
    newrot = rot * cd_rot
    xscale = 0.104 / 3600.0 * binfac1
    yscale = 0.43 / 3600.0 / int(4 / binfac1)
    newhdr['CD1_1'] = newrot[0, 0] * xscale
    newhdr['CD1_2'] = newrot[0, 1] * yscale
    newhdr['CD2_1'] = newrot[1, 0] * xscale
    newhdr['CD2_2'] = newrot[1, 1] * yscale

    return hdulst

# read in our regions data, so we don't have to re-read it over and over
modulehome, _xx = os.path.split(bs.__file__)
for binning in (1, 2, 4):
    # if no region data is passed, read one based on the binning
    regionfile = os.path.join(modulehome, "ifu_regions",
                              "pseudoslits%d.reg" % binning)
    regions[binning] = read_region_file(regionfile)


def main(options, args):

    # read slit positions from region file
    regdata = read_region_file(options.regionfile)

    # bias subtract, combine and extract regions
    hdulst = reconstruct_image(args[0], args[1], regdata)

    # Writing the output fits file
    hdulst.writeto(options.outputfile)

if __name__ == '__main__':
    # Parse command line options with optparse module
    from optparse import OptionParser

    usage = "usage: %prog [options] ch1.fits ch2.fits"
    optprs = OptionParser(usage=usage,
                          version=('%%prog %s' % version))

    optprs.add_option("-o", "--outputfile", dest="outputfile", metavar="NAME",
                      default='output.fits',
                      help="Specify output file name")
    optprs.add_option("-r", "--regionfile", dest="regionfile", metavar="NAME",
                      default='pseudoslits.reg',
                      help="Specify region file name")
    (options, args) = optprs.parse_args(sys.argv[1:])

    main(options, args)
