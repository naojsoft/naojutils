from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from scipy.stats import sigmaclip
from astropy.io import fits
import os
from . import focasifu as fi

def MkBiasTemplate(filename, nsigma=4.0, rawdatadir='', overwrite=False):
    hdulist = fits.open(rawdatadir + filename)
    detid = hdulist[0].header['DET-ID']
    scidata = hdulist[0].data
    hdulist.close()

    average1d = np.zeros(scidata.shape[1])
    for i in range(len(average1d)):
        clipped, low, upp = sigmaclip(scidata[:,i], low=nsigma, high=nsigma)
        average1d[i] = np.mean(clipped)

    outfilename = 'bias_template'+str(detid)+'.fits'
    if os.path.isfile(outfilename) and not overwrite:
        print(('File exists. '+outfilename))
        return

    hdu = fits.PrimaryHDU(data=average1d)
    hdulist = fits.HDUList([hdu])
    hdulist = fi.put_version(hdulist)
    hdulist.writeto(outfilename, overwrite=overwrite)
    print(('Bias template file was created. '+outfilename))

    return

def MkTwoBiasTemplate(filename, rawdatadir='', overwrite = False):
    MkBiasTemplate(filename, rawdatadir=rawdatadir, overwrite=overwrite)
    basename = fits.getval(rawdatadir + filename, 'FRAMEID')
    filename2 = str('FCSA%08d.fits'%(int(basename[4:])+1))
    MkBiasTemplate(filename2, rawdatadir=rawdatadir, overwrite=overwrite)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the script for making bias template files..')
    parser.add_argument('filename',help='Bias FITS file')
    parser.add_argument('-o', help='Overwrite flag', dest='overwrite',
                    action='store_true', default=False)
    parser.add_argument('-d', help='Raw data directory', \
            dest='rawdatadir', action='store', default='')
    args = parser.parse_args()

    MkTwoBiasTemplate(args.filename, rawdatadir=args.rawdatadir, \
                      overwrite=args.overwrite)
