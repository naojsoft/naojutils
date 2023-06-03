import sys
import numpy as np
from astropy.io import fits
from .bias_overscan import bias_subtraction, remove_overscan
from . import focasifu as fi

def biassub(fname1, fname2, template_pfx='bias_template'):
    # Reading the image files
    rhdl0 = fits.open(fname1)
    lhdl0 = fits.open(fname2)

    # Getting the binning information
    binfac1 = rhdl0[0].header['BIN-FCT1'] # X direction on DS9

    # Bias subtraction
    rhdl1, stat = bias_subtraction(rhdl0, template_pfx)
    lhdl1, stat  = bias_subtraction(lhdl0, template_pfx)

    # Over scan region removing
    rhdl2, stat  = remove_overscan(rhdl1)
    lhdl2, stat  = remove_overscan(lhdl1)

    # Set the start X of right half image
    # CCD gap: 5 arcsec
    # pixel scale: 0.104 arcsec/pix
    gap_width = int(5 / 0.104 / binfac1)
    gap = np.zeros((rhdl0[0].header['NAXIS2'], gap_width),dtype=np.float32)

    # stack the left and right images
    stacked_data = np.hstack((lhdl2[0].data, gap))
    stacked_data = np.hstack((stacked_data, rhdl2[0].data))

    # Writing the output fits file
    outhdu = fits.PrimaryHDU(data=stacked_data)
    outhdl = fits.HDUList([outhdu])
    outhdl[0].header = rhdl2[0].header
    outhdl = fi.put_version(outhdl)
    rhdl0.close()
    lhdl0.close()
    return outhdl
