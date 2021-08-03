#!/usr/bin/env python
#
# 2019/06/30:
# Modification for the change of 'get_binneddata' function to 'integrate'
#   function in 'reconstruct_image.py'
#

import numpy as np
from astropy.io import fits

from . import biassub as bs
from .reconstruct_image import integrate


def mkflat(fitsfile1, fitsfile2, out_file=None):
    # Filter dictionary
    filter_dict={'SCFCFLBI01':'I',
                 'SCFCFLBR01':'R',
                 'SCFCFLBV01':'V',
                 'SCFCFLBSZ1':'Z'}

    # subtract bias and combine channels
    hdl = bs.biassub(fitsfile1, fitsfile2)

    # Get integrated data
    integrated_hdl = integrate(hdl)

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

    return integrated_hdl, out_file
