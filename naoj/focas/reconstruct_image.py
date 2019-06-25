# -*- coding: utf-8 -*-
# Shinobu Ozaki
from __future__ import absolute_import
import sys, os
import re
import numpy as np
from astropy.io import fits
from scipy.ndimage.interpolation import shift

# local imports
from . import biassub as bs
from . import focasifu as fi
version = "20180226.0"


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

def flatfielding(data, flat, binfac1, shift_flg=True):
    # data: image data
    # flat: flat data

    xrange = {1: [11, 148],
              2: [6, 74],
              4: [4, 36]}

    xshift = np.zeros(data.shape[0])
    for i in range(len(xshift)):
        xshift[i] = fi.cross_correlate(data[i,:], flat[i,:], sep=0.1)
    xshift_std = np.std(xshift)
    xshift_ave = np.average(xshift)
    if xshift_std != 0.0 and xshift_std < 0.3 and shift_flg:
        shifted_flat = shift(flat, (0.0, xshift_ave), order=5,
                                      mode='nearest')
        temp = data / shifted_flat * np.average(shifted_flat)
        xs = xrange[binfac1][0] + int(xshift_ave+0.5)
        xe = xrange[binfac1][1] + int(xshift_ave+0.5)
        flatted_data = temp[:, xs:xe]
        is_shifted = True
    else:
        flatted_data =  data / flat * np.average(flat)
        is_shifted = False

    return flatted_data, xshift_ave, xshift_std, is_shifted


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
    yscale = 0.43 / 3600.0 / int(4 / binfct1)
    outhdr['CD1_1'] = newrot[0, 0] * xscale
    outhdr['CD1_2'] = newrot[0, 1] * yscale
    outhdr['CD2_1'] = newrot[1, 0] * xscale
    outhdr['CD2_2'] = newrot[1, 1] * yscale

    return


def get_binneddata(hdl, regionfile=None):
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
    binned_data = np.zeros((reg.shape[0], xw), np.float32)
    for j in range(reg.shape[0]):
        xs = int(reg[j, 0] - reg[j, 2] / 2.0)
        ys = int(reg[j, 1] - reg[j, 3] / 2.0)
        binned_data[j, :] = np.sum(scidata[ys:ys+yw, xs:xs+xw], axis=0)

    return binned_data


def reconstruct_image(fitsfile_ch1, fitsfile_ch2,
                      template_pfx='bias_template', regionfile=None,
                      flatfile=None, shift_flg=True, smooth_flg=False):

    # subtract bias and combine channels
    hdl = bs.biassub(fitsfile_ch1, fitsfile_ch2, template_pfx=template_pfx)
    hdr = hdl[0].header
    binfac1 = hdr['BIN-FCT1'] # X direction on DS9

    # Get binned data
    binned_data = get_binneddata(hdl, regionfile=regionfile)

    # Flatfielding if needed
    if flatfile is not None:
        with fits.open(flatfile) as flathdulst:
            normdata = flathdulst[0].data / np.mean(flathdulst[0].data)
        flatted_data, xshift_ave, xshift_std, is_shifted = \
                            flatfielding(binned_data, normdata, binfac1,
                                         shift_flg=shift_flg)
        is_flatted = True
    else:
        flatted_data = binned_data
        xshift_ave = 0.0
        xshift_std = 0.0
        is_shifted = False
        is_flatted = False

    # creating output data array
    itnum = int(4/binfac1)
    recon_im = np.zeros((flatted_data.shape[0]*itnum, flatted_data.shape[1]),
                        dtype=np.float32)

    # creating HDU list of the reconstruct image
    outhdu = fits.PrimaryHDU(data=recon_im)
    outhdl = fits.HDUList([outhdu])

    # creating header information
    outhdr = outhdl[0].header
    outhdr['XSHFTAVE'] = (xshift_ave, 'Average xshift of flat image')
    outhdr['XSHFTSTD'] = (xshift_std,
                          'Standard deviation of xshift for all slices')
    outhdr['ISSHFTED'] = (is_shifted, 'True: flat image is shifted')
    outhdr['ISFLATED'] = (is_flatted, 'True: Flat fielding is applied')
    creating_header(hdl, outhdl)

    ## Ch24 (Sky)
    for i in range(itnum):
        recon_im[i,:] = flatted_data[0,:]

    ## Ch23 - Ch02
    for j in range(flatted_data.shape[0]-1):
        for i in range(itnum):
            if smooth_flg:
                recon_im[j*itnum + i,:] = \
                    ((itnum-i)*flatted_data[j,:] + i*flatted_data[j+1,:]) \
                    / itnum
            else:
                recon_im[j*itnum + i,:] = flatted_data[j,:]

    ## Ch01
    for i in range(itnum):
        recon_im[(flatted_data.shape[0]-1)*itnum+i,:] = \
                        flatted_data[flatted_data.shape[0]-1,:]


    return outhdl


# read in our regions data, so we don't have to re-read it over and over
modulehome, _xx = os.path.split(bs.__file__)
for binning in (1, 2, 4):
    # if no region data is passed, read one based on the binning
    regionfile = os.path.join(modulehome, "ifu_regions",
                              "pseudoslits%d.reg" % binning)
    regions[binning] = read_region_file(regionfile)
