# -*- coding: utf-8 -*-
# Shinobu Ozaki
#
# 2019/06/30:
# X shift is calculated based on the flexure data.
# 'focasifu' module is no longer imported.


import sys, os
import re
import numpy as np
from astropy.io import fits
from scipy.ndimage.interpolation import shift
import math
from . import focasifu as fi

# local imports
from . import biassub as bs
version = "20180226.0"
xrange = {1: [11, 148],
          2: [6, 74],
          4: [4, 36]}

# Transfering FITS keywords
transfer_kwds = ['EXP-ID', 'FOC-VAL', 'OBJECT', 'EXPTIME',
                 'FILTER01', 'FILTER02', 'FILTER03', 'FRAMEID', 'HST',
                 'LST', 'RADECSYS', 'EQUINOX', 'DATE-OBS', 'CTYPE1',
                 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
                 'CUNIT1', 'CUNIT2', 'PROP-ID', 'OBS-MOD', 'EXP-ID',
                 'RA', 'DEC']

# this will hold region data
regions = {}

def read_region_file(region_file):
    # Reading the region file
    with open(region_file, 'r') as in_f:
        lines = in_f.readlines()

    # Parsing the region data
    regdata = np.zeros((24, 4), dtype='float')
    i = 0
    for line in lines:
        data = re.split('[(),]', line)
        if data[0] == "box":
            regdata[i, :] = [float(data[1]), float(data[2]),
                             float(data[3]), float(data[4])]
            i = i + 1

    return regdata

def get_shift_flex(hdl):
    hdr = hdl[0].header
    el = hdr['ALTITUDE']
    insrot = hdr['INSROT']
    binfac1 = hdr['BIN-FCT1'] # X direction on DS9
    binfac2 = hdr['BIN-FCT2'] # Y direction on DS9

    el = math.radians(el)
    insrot = math.radians(insrot)

    a = -6.98763
    b = -0.299919
    c = 6.65883
    d = -16.2355
    e = -0.0359595
    f = 2.77652

    dx = a*math.cos(el)*math.cos(insrot+b)
    #dy = c*math.cos(el)*math.sin(insrot+d)+e*el+f

    dx = dx/binfac1
    #dy = dy/binfac2
    return dx

def get_shift_corr(hdl, flat):
    # if object is on Ch10, this function does not work well.
    data = hdl[0].data
    dx = fi.cross_correlate(data[9,:], flat[9,:], sep=0.01, fit=False)
    return dx

def flatfielding(hdl, flat):
    # Shifting data
    dx = get_shift_flex(hdl)
    shifted_data = shift(hdl[0].data, (0.0, -dx), order=5, mode='nearest')

    # Flat fielding
    flatted_data = shifted_data / flat * np.average(flat)

    return flatted_data, dx


def creating_header(hdl, outhdl):
    global transfer_kwds

    hdr = hdl[0].header
    outhdr = outhdl[0].header
    for kwd in transfer_kwds:
        outhdr[kwd] = hdr[kwd]

    # Extraction rotation matrix from CD matrix
    cd = np.matrix([[hdr['CD1_1'], hdr['CD1_2']], [hdr['CD2_1'], hdr['CD2_2']]])
    det = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    cd_rot = cd / np.sqrt(det)

    # Rotation matrix of the IFU field with respect to hte FOCAS field
    theta = -21.38 # degree
    theta = np.deg2rad(theta)
    rot = np.matrix([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    # Creating the new CD matrix for the reconstruction image
    binfct1 = hdr['BIN-FCT1']
    newrot = rot * cd_rot
    xscale = 0.104 / 3600.0 * binfct1
    yscale = -0.43 / 3600.0 / int(4 / binfct1)
    outhdr['CD1_1'] = newrot[0, 0] * xscale
    outhdr['CD1_2'] = newrot[0, 1] * yscale
    outhdr['CD2_1'] = newrot[1, 0] * xscale
    outhdr['CD2_2'] = newrot[1, 1] * yscale

    return


def integrate(hdl, regionfile=None):
    scidata = hdl[0].data
    # Getting the binning information
    hdr = hdl[0].header
    binfac1 = hdr['BIN-FCT1'] # X direction on DS9
    binfac2 = hdr['BIN-FCT2'] # Y direction on DS9
    if binfac1 not in (1, 2, 4):
        raise ValueError("X binning factor is not 1, 2 or 4: BIN-FCT1=%d" % (
            binfac1))

    global regions
    if regionfile is None:
        reg = regions[int(binfac1)]
    else:
        reg = read_region_file(regionfile)

    # extract data from regions
    xw, yw = int(reg[1, 2]), int(reg[1, 3])

    # Factor 'int(4/binfac1)' is to match Y scale to X scale in the
    # reconstructed image.
    # integrated_data[0,:] is CH24, integrated_data[23,:] is Ch01.
    integrated_data = np.zeros((reg.shape[0], xw), np.float32)
    for j in range(reg.shape[0]-1,-1,-1):
        xs = int(reg[j, 0] - reg[j, 2] / 2.0)
        ys = int(reg[j, 1] - reg[j, 3] / 2.0)
        integrated_data[j, :] = np.sum(scidata[ys:ys+yw, xs:xs+xw], axis=0)

    # creating HDU list of the reconstruct image
    integrated_hdu = fits.PrimaryHDU(data=integrated_data)
    integrated_hdl = fits.HDUList([integrated_hdu])
    integrated_hdl[0].header = hdr

    return integrated_hdl


def reconstruct_image(fitsfile_ch1, fitsfile_ch2,
                      template_pfx='bias_template', regionfile=None,
                      flatfile=None, shift_flg=True, smooth_flg=False):

    # subtract bias and combine channels
    hdl = bs.biassub(fitsfile_ch1, fitsfile_ch2, template_pfx=template_pfx)
    hdr = hdl[0].header
    binfac1 = hdr['BIN-FCT1'] # X direction on DS9

    # Get integrated data
    integrated_hdl = integrate(hdl, regionfile=regionfile)

    if flatfile is not None:
        with fits.open(flatfile) as flathdulst:
            flatted_data, xshift =  flatfielding(integrated_hdl, flathdulst[0].data)
        is_flatted = True
    else:
        flatted_data = integrated_hdl[0].data
        is_flatted = False

    # creating output data array
    itnum = int(4/binfac1)
    recon_im = np.zeros((flatted_data.shape[0]*itnum+1, flatted_data.shape[1]),
                        dtype=np.float32)

    # creating HDU list of the reconstruct image
    outhdu = fits.PrimaryHDU(data=recon_im)
    outhdl = fits.HDUList([outhdu])

    # creating header information
    outhdr = outhdl[0].header
    if is_flatted:
        outhdr['XSHFT'] = (xshift, 'Xshift value of flat image (pix)')
    outhdr['ISFLATED'] = (is_flatted, 'True: Flat fielding is applied')
    creating_header(hdl, outhdl)

    # Inputing the output data
    ymax = flatted_data.shape[0]-1
    ## Ch01
    for i in range(itnum):
        recon_im[i,:] = \
                        flatted_data[ymax,:]

    ## Ch02 - Ch23
    for j in range(1, flatted_data.shape[0]-1):
        for i in range(itnum):
            if smooth_flg:
                recon_im[j*itnum + i,:] = \
                    ((itnum-i)*flatted_data[ymax-j,:] + i*flatted_data[ymax-j-1,:]) \
                    / itnum
            else:
                recon_im[j*itnum + i,:] = flatted_data[ymax - j,:]

    ## Border line
    recon_im[ymax*itnum,:] = np.nan

    ## Ch24 (Sky)
    for i in range(itnum):
        recon_im[ymax*itnum+i+1,:] = flatted_data[0,:]

    return outhdl


# read in our regions data, so we don't have to re-read it over and over
modulehome, _xx = os.path.split(bs.__file__)
for binning in (1, 2, 4):
    # if no region data is passed, read one based on the binning
    regionfile = os.path.join(modulehome, "ifu_regions",
                              "pseudoslits%d.reg" % binning)
    regions[binning] = read_region_file(regionfile)
