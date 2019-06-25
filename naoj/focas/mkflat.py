#!/usr/bin/env python

import numpy as np
from astropy.io import fits

from . import biassub as bs
from .reconstruct_image import get_binneddata, creating_header


def mkflat(fitsfile1, fitsfile2, out_file=None):
    # Filter dictionary
    filter_dict={'SCFCFLBI01':'I','SCFCFLBR01':'R'}

    # subtract bias and combine channels
    hdl = bs.biassub(fitsfile1, fitsfile2)

    # Get binned data
    binned_data = get_binneddata(hdl)

    # creating HDU list of the reconstruct image
    outhdu = fits.PrimaryHDU(data=binned_data)
    outhdl = fits.HDUList([outhdu])

    # creating header information
    creating_header(hdl, outhdl)

    # creatign an output file name if it is not given.
    filter = []
    if out_file is None:
        hdr = hdl[0].header
        binfct1 = hdr['BIN-FCT1']
        filter.append(hdr['FILTER01'])
        filter.append(hdr['FILTER02'])
        filter.append(hdr['FILTER03'])
        out_file = 'flat_'+str(binfct1)+'_'
        for i in filter:
            if i != 'NONE':
                out_file += filter_dict[i]
        out_file += '.fits'

    hdl.close()

    return outhdl, out_file
