#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from astropy.io import fits
import argparse
from . import focasifu as fi

# Dfinitions for the over scan regions in the DS9 image coordinate.
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

def bias_subtraction(inhdl, template_pfx):
    #print(('\t Bias subtracting for the frame ID, %s.'
    #      %inhdl[0].header['FRAMEID']))
    # Bias subtraction
    scidata = inhdl[0].data

    # Getting the binning and detector information
    inhdr = inhdl[0].header
    binfac1 = inhdr['BIN-FCT1']  # X direction on DS9
    binfac2 = inhdr['BIN-FCT2']  # Y direction on DS9
    detid = inhdr['DET-ID']

    # Put the version number in the FITS header
    inhdl = fi.put_version(inhdl)

    # Get the appropriate over scan region area
    ovs = overscan[binfac1]

    # Discriminating whether the flame is for left or right.
    # detid = 1 => right image
    # detid = 2 => left image
    if detid == 1:
        k = 4
    else:
        k = 0

    # Checking the bias template file and making bias 1D data
    bias_template_name = template_pfx+str(binfac1)+str(detid)+'.fits'
    use_template = False
    if os.path.isfile(bias_template_name):
        bias_template_hdl = fits.open(bias_template_name)
        if fi.check_version(bias_template_hdl):
            bias_template_data = bias_template_hdl[0].data
            use_template = True
        bias_template_hdl.close()
    
    if not use_template:
        #print('!!! There is no bias template file, '+bias_template_name+'.')
        #print('!!! Top overscan region is refered as bias.')
        bias_template_data = \
                np.mean(scidata[scidata.shape[0]-13:scidata.shape[0],:],
                        axis=0)

    # Subtracting the bias pattern scaled by the derived sum.
    bsdata = np.zeros((scidata.shape[0], scidata.shape[1]), dtype=np.float32)
    for i in range(2, scidata.shape[0]-3):
        for j in range(k,k+4):
            template_level = np.mean(bias_template_data[ovs[j,0]-1:ovs[j,1]])
            template_level = template_level + \
                     np.mean(bias_template_data[ovs[j,4]-1:ovs[j,5]])
            ovlevel = np.mean(scidata[i-2:i+3,ovs[j,0]-1:ovs[j,1]])
            ovlevel = ovlevel + \
                      np.mean(scidata[i-2:i+3,ovs[j,4]-1:ovs[j,5]])
            bsdata[i,ovs[j,0]-1:ovs[j,5]] = \
                        scidata[i,ovs[j,0]-1:ovs[j,5]] - \
                        bias_template_data[ovs[j,0]-1:ovs[j,5]] / \
                        template_level * ovlevel
        
    # Creating HDU data
    outhdu = fits.PrimaryHDU(data=bsdata)
    outhdl = fits.HDUList([outhdu])

    # Removing some keywords because they are only applicable to integer data
    # If thesee header keywords are removed after copying, then
    # the following WARNING is appeared.
    # "WARNING: VerifyWarning: Invalid 'BLANK' keyword in header.  The
    #  'BLANK' keyword is only applicable to integer data, and will be
    #  ignored in this HDU. [astropy.io.fits.hdu.image]"

    inhdr.remove('BLANK')
    inhdr.remove('BSCALE')
    inhdr.remove('BZERO')

    # copy header keywords of the input file
    outhdl[0].header = inhdr
    return outhdl, True

def remove_overscan(inhdl):
    #print(('\t Overscan region removing for %s.'%inhdl[0].header['FRAMEID']))

    # Checking the version consistency
    if not fi.check_version(inhdl):
        return inhdl, False

    # Overscan region removing
    scidata = inhdl[0].data

    # Getting the binning and detector information
    inhdr=inhdl[0].header
    binfac1 = inhdr['BIN-FCT1'] # X direction on DS9
    binfac2 = inhdr['BIN-FCT2'] # Y direction on DS9
    detid = inhdr['DET-ID']

    # Get the appropriate over scan region area
    ovs = overscan[binfac1]

    # Trim Y range
    yrange = [[51, 4220], # for 1 bin
              [26, 2110], # for 2 bin
              [17, 1406], # for 3 bin
              [13,1055]]  # for 4 bin

    # Discriminating whether the flame is for left or right.
    topovlevel = 0.0
    if detid == 1:
        k = 4
    else:
        k = 0

    # Generating the new data array
    newdata = np.zeros((scidata.shape[0],scidata.shape[1]),dtype=np.float32)

    # Convert the count values to electrons.
    # From the leftmost amp of DET-ID=2 to the rightmost amp of DET-ID=1
    gain = [2.054, 1.987, 1.999, 1.918, 2.081, 2.047, 2.111, 2.087] # electron/ADU

    # Generating the new data without overscan regions
    xmax = 0
    for i in range(k,k+4):
        d = ovs[i,3]-ovs[i,2] + 1
        newdata[:,xmax:xmax+d] = scidata[:,ovs[i,2]-1:ovs[i,3]] * gain[i]
        #print(gain[i])
        xmax = xmax + d

    # Creating HDU data
    outhdu = fits.PrimaryHDU(data=newdata[yrange[binfac2-1][0]:yrange[binfac2-1][1],:xmax])
    outhdl = fits.HDUList([outhdu])
    outhdl[0].header = inhdr
    outhdl[0].header['BUNIT'] = 'electrons'
    outhdl[0].header['TRM_Y1'] = (yrange[binfac2-1][0]+1,
                                  'Y start of adopped area for trimming')
    outhdl[0].header['TRM_Y2'] = (yrange[binfac2-1][1],
                                  'Y end of adopped area for trimming')

    return outhdl, True


def restore_badpix(inhdl):
    # Bad pixel correction
    #print('\t Restoring bad pixels for %s.'%inhdl[0].header['FRAMEID'])

    # Checking the version consistency
    if not fi.check_version(inhdl):
        return inhdl, False

    scidata = inhdl[0].data
    binfct2 = inhdl[0].header['BIN-FCT2']
    i = binfct2 - 1
    if inhdl[0].header['DET-ID'] == 1:
        # Bad pixel coordinates before removing overscan regions.
        # x1,x2,y1,y2
        badpix = np.array([
            [397,397,2387,4225],  # for 2x1 bin
            [397,397,1189,2105]   # for 2x2 bin
        ])

        scidata[badpix[i,2]-1:badpix[i,3],badpix[i,0]-1:badpix[i,1]] = \
            (scidata[badpix[i,2]-1:badpix[i,3],badpix[i,0]-2:badpix[i,1]-1] + \
             scidata[badpix[i,2]-1:badpix[i,3],badpix[i,0]:badpix[i,1]+1])/2.0

    # Creating HDU data
    outhdu = fits.PrimaryHDU(data=scidata)
    outhdl = fits.HDUList([outhdu])
    outhdl[0].header = inhdl[0].header
    return outhdl, True


def stack_data(hdl_right, hdl_left):
    # hdl_right: right image
    # hdl_left: left image
    #
    #print('\t Stacking the frames.')
    binfac1 = hdl_right[0].header['BIN-FCT1']

    # Set the start X of right half image
    # CCD gap: 5 arcsec
    # pixel scale: 0.104 arcsec/pix
    gap_width = int(5 / 0.104 / binfac1)
    gap = np.zeros((hdl_right[0].header['NAXIS2'], gap_width),\
                   dtype=np.float32)

    # stack the left and right images
    stacked_data = np.hstack((hdl_left[0].data, gap))
    stacked_data = np.hstack((stacked_data, hdl_right[0].data))

    # Writing the output fits file
    outhdu = fits.PrimaryHDU(data=stacked_data)
    outhdl = fits.HDUList([outhdu])
    outhdl[0].header = hdl_right[0].header
    outhdl[0].header['GAP_X1'] = (hdl_left[0].data.shape[1]+1, 'Gap start X')
    outhdl[0].header['GAP_X2'] = (hdl_left[0].data.shape[1]+gap_width, \
                                  'Gap end X')
    return outhdl


def correct_header(hdl):
    #print('\t Correcting the header information.')
    hdr=hdl[0].header
    # Original keys are renamed.
    #hdr['OCUNIT1'] = (hdr['CUNIT1'], 'Original CUNIT1')
    #hdr['OCUNIT2'] = (hdr['CUNIT2'], 'Original CUNIT2')
    hdr['OCRVAL1'] = (hdr['CRVAL1'], 'Original CRVAL1')
    hdr['OCRVAL2'] = (hdr['CRVAL2'], 'Original CRVAL2')
    #hdr['OCRPIX1'] = (hdr['CRPIX1'], 'Original CRPIX1')
    #hdr['OCRPIX2'] = (hdr['CRPIX2'], 'Original CRPIX2')
    #hdr['OCDELT1'] = (hdr['CDELT1'], 'Original CDELT1')
    #hdr['OCDELT2'] = (hdr['CDELT2'], 'Original CDELT2')
    #hdr['OCTYPE1'] = (hdr['CTYPE1'], 'Original CTYPE1')
    #hdr['OCTYPE2'] = (hdr['CTYPE2'], 'Original CTYPE2')
    #hdr['OCD1_1']  = (hdr['CD1_1'], 'Original CD1_1')
    #hdr['OCD1_2']  = (hdr['CD1_2'], 'Original CD1_2')
    #hdr['OCD2_1']  = (hdr['CD2_1'], 'Original CD2_1')
    #hdr['OCD2_2']  = (hdr['CD2_2'], 'Original CD2_2')

    # REIDENTIFY task need this keyword.
    hdr['CD1_1'] = 1.
    hdr['CD2_2'] = 1.

    hdr.remove('CUNIT1')
    hdr.remove('CUNIT2')
    hdr.remove('CRVAL1')
    hdr.remove('CRVAL2')
    hdr.remove('CRPIX1')
    hdr.remove('CRPIX2')
    hdr.remove('CDELT1')
    hdr.remove('CDELT2')
    hdr.remove('CTYPE1')
    hdr.remove('CTYPE2')
    hdr.remove('CD1_2')
    hdr.remove('CD2_1')
    hdr.remove('PC001001')
    hdr.remove('PC001002')
    hdr.remove('PC002001')
    hdr.remove('PC002002')

    return hdl


def bias_overscan(ifname, rawdatadir='', template_pfx='bias_template',
                  overwrite=False):
    #print('\n#############################')
    #print('bias subtraction, overscan region removing, bad pixel correction, hedear correction')

    basename = fits.getval(rawdatadir+ifname, 'FRAMEID')
    ovname = basename+'.ov.fits'
    if os.path.isfile(ovname) and not overwrite:
        print(('\t Bias-subtructed and overscan-removed frame already exits. %s'%ovname))
        print('\t This procedure is skipped.')
        return ovname, True

    hdl = fits.open(rawdatadir+ifname)
    hdl, stat = bias_subtraction(hdl, template_pfx)
    if stat == False:
        hdl.close()
        return ovname, False

    hdl, stat = remove_overscan(hdl)
    if stat == False:
        hdl.close()
        return ovname, False

    hdl_right, stat = restore_badpix(hdl)
    if stat == False:
        hdl.close()
        return ovname, False

    hdl.close()

    ifname = str('FCSA%08d.fits'%(int(basename[4:])+1))
    hdl = fits.open(rawdatadir+ifname)
    hdl = fi.put_version(hdl)

    hdl, stat = bias_subtraction(hdl, template_pfx)
    if stat == False:
        hdl.close()
        return ovname, False

    hdl, stat = remove_overscan(hdl)
    if stat == False:
        hdl.close()
        return ovname, False

    hdl_left, stat = restore_badpix(hdl)
    if stat == False:
        hdl.close()
        return ovname, False

    hdl.close()

    hdl_stacked = stack_data(hdl_right, hdl_left)
    hdl_stacked = correct_header(hdl_stacked)
    hdl_stacked.writeto(ovname, overwrite=overwrite)
    hdl_stacked.close()
    return ovname, True

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description= \
            'This is the script for bias subtraction,'+\
            ' overscan removing, and bad pixel correction.')
    parser.add_argument('-o', help='Overwrite flag',\
            dest='overwrite', action='store_true', default=False)
    parser.add_argument('ifname',\
                    help='Input FITS file for Chip 1')
    parser.add_argument('-d', help='Raw data directory', \
            dest='rawdatadir', action='store', default='')
    args = parser.parse_args()

    bias_overscan(args.ifname, rawdatadir=args.rawdatadir, \
                  overwrite=args.overwrite)
