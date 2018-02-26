#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Shinobu Ozaki
"""
USAGE: python biassub.py right.fits left.fits -o output.fits

    right.fits: right image file name (smaller file number)
    left.fits: left image file name (larger file number)
    output_fits: output file name

"""
import sys
import numpy as np
from astropy.io import fits

version = "20180226.0"

# Format
# 1: start of left overscan region
# 2: end  of left overscan region
# 3: start of image region
# 4: end of image region
# 5: start of right overscan region
# 6: end  of right overscan region
#
overscan = {}
# binning == 1
overscan[1] = np.asarray([
    # For left image
    [2, 8, 9, 520, 521, 536],
    [537, 552, 553, 1064, 1065, 1071],
    [1074, 1080, 1081, 1592, 1593, 1608],
    [1609, 1624, 1625, 2136, 2137, 2142],
    #
    # For right image
    [2, 8, 9, 520, 521, 536],
    [537, 552, 553, 1064, 1065, 1071],
    [1074, 1080, 1081, 1592, 1593, 1608],
    [1610, 1625, 1626, 2137, 2138, 2143],
    ])

# binning == 2
overscan[2] = np.asarray([
    # For left image
    [2, 4, 5, 260, 261, 276],
    [277, 292, 293, 548, 549, 551],
    [553, 556, 557, 812, 813, 828],
    [829, 844, 845, 1100, 1101, 1104],
    #
    # For right image
    [1, 4, 5, 260, 261, 276],
    [277, 292, 293, 548, 549, 551],
    [553, 556, 557, 812, 813, 828],
    [829, 845, 846, 1101, 1102, 1104],
    ])

# binning == 4
overscan[4] = np.asarray([
    # For left image
    [1, 2, 3, 130, 131, 146],
    [147, 162, 163, 290, 291, 292],
    [293, 294, 295, 422, 423, 438],
    [439, 454, 455, 582, 583, 584],
    #
    # For right image
    [1, 2, 3, 130, 131, 146],
    [147, 162, 163, 290, 291, 292],
    [293, 294, 295, 422, 423, 438],
    [439, 455, 456, 583, 584, 584]
    ])

def bias_subtraction(fname1, fname2):
    # Opening the input fits files
    with fits.open(fname1) as hdulist1:
        # right image
        hdr = hdulist1[0].header
        scidata1 = hdulist1[0].data

    with fits.open(fname2) as hdulist2:
        # left image
        scidata2 = hdulist2[0].data

    # Getting the binning information
    binfac1 = hdr['BIN-FCT1']  # X direction on DS9
    binfac2 = hdr['BIN-FCT2']  # Y direction on DS9

    # Get the appropriate over scan region area
    ovs = overscan[binfac1]

    # Subtracting the bias level and combining to one data array
    bs_data = np.zeros((scidata1.shape[0], scidata1.shape[1] * 2))
    # left image
    for i in range(scidata2.shape[0]):
        x1 = 0
        x2 = 0
        for j in range(4):
            ovlevel = np.mean(scidata2[i, ovs[j, 0]-1:ovs[j, 1]])
            ovlevel = ovlevel + np.mean(scidata2[i, ovs[j, 4]-1:ovs[j, 5]])
            ovlevel = ovlevel / 2.0
            x2 = x1 + ovs[j, 3] - ovs[j, 2] + 1
            bs_data[i, x1:x2] = scidata2[i, ovs[j, 2]-1:ovs[j, 3]] - ovlevel
            x1 = x2

    # Set the start X of right half image
    # CCD gap: 5 arcsec
    # pixel scale: 0.104 arcsec/pix
    right_start_x = x2 + int(5 / 0.104 / binfac1)

    # right image
    for i in range(scidata1.shape[0]):
        x1 = right_start_x
        x2 = 0
        for j in range(4, 8):
            ovlevel = np.mean(scidata1[i, ovs[j, 0] - 1:ovs[j, 1]])
            ovlevel = ovlevel + np.mean(scidata1[i, ovs[j, 4] - 1:ovs[j, 5]])
            ovlevel = ovlevel / 2.0
            x2 = x1 + ovs[j, 3] - ovs[j, 2] + 1
            bs_data[i, x1:x2] = scidata1[i, ovs[j, 2] - 1:ovs[j, 3]] - ovlevel
            x1 = x2

    # Creating HDU data
    hdu = fits.PrimaryHDU(data=bs_data)
    hdl = fits.HDUList([hdu])
    # remove BLANK keyword because is only applicable to integer data
    hdulist1[0].header.remove('BLANK')
    hdulist1[0].header.remove('BSCALE')
    hdulist1[0].header.remove('BZERO')
    # copy header keywords of the input file
    hdl[0].header = hdulist1[0].header
    return hdl

def main(options, args):
    # USAGE: biassub.py -o output.fits right.fits left.fits
    # right.fits: right image file name (smaller file number)
    # left.fits: left image file name (larger file number)
    # output.fits: output file name

    hdulst = bias_subtraction(args[0], args[1])

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
    (options, args) = optprs.parse_args(sys.argv[1:])

    main(options, args)
