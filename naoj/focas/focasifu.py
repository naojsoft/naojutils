import numpy as np
#import os
from astropy.io import fits
import math
from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter
import copy

version = 20190130
# these seem to be unused....EJ
#filibdir = os.path.dirname(os.path.abspath(__file__))+'/../lib/'
#chimagedir = 'chimages/'
#bias_template_file = 'bias_template'
ifu_soft_key = 'IFU_SOFT'


def weighted_average_std(data, w):
    sum = w.sum()
    waverage = np.dot(data, w)/sum
    a = data - waverage
    wvariance = np.dot(a**2, w)/sum
    wstd = math.sqrt(wvariance)
    return waverage, wstd


def getmedian(data, lower=np.nan):
    data1 = np.zeros(data.shape)
    if lower != np.nan:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] < lower:
                    data1[i, j] = np.nan
                else:
                    data1[i, j] = data[i, j]
    return np.nanmedian(data1)



def cheb1Dfit(x, y, order=5, weight=None, niteration=0, \
              high_nsig=0.0, low_nsig=0.0):
    n = len(x)
    if weight is None:
        weight1 = np.ones(n)
    else:
        weight1 = copy.deepcopy(weight)

    c = chebfit(x,y,order,w=weight1)

    for k in range(niteration):
        residualdata = y - chebval(x,c)
        # Calculating weighted standard deviation
        mean, sig = weighted_average_std(residualdata, weight1)

        # Clipiing the data
        highlimit = high_nsig * sig
        lowlimit = -low_nsig * sig
        for j in range(n):
            if residualdata[j] > highlimit or residualdata[j] < lowlimit:
                weight1[j] = 0.0

        # Fitting again
        c = chebfit(x,y,order,w=weight1)

    return c, weight1


def datafiltering(data, weight):
    data_accept = np.zeros(len(data))
    data_reject = np.zeros(len(data))

    accept_num = 0
    reject_num = 0
    for i in range(len(data)):
        if weight[i] == 1:
            data_accept[accept_num] = data[i]
            accept_num += 1
        elif weight[i] == 0:
            data_reject[reject_num] = data[i]
            reject_num += 1

    return data_accept[:accept_num], data_reject[:reject_num]


def put_version(hdl):
    global ifu_soft_key
    if not ifu_soft_key in list(hdl[0].header):
        hdl[0].header[ifu_soft_key] = (version, 'IFU pipline version')
    elif hdl[0].header[ifu_soft_key] != version:
        print(('Bias subtraction seems to be already applied '+\
              'because the version number is in the FITS header.'))
        return False
    return hdl


def check_version(hdl):
    global ifu_soft_key
    if ifu_soft_key in list(hdl[0].header):
        if hdl[0].header[ifu_soft_key] != version and \
           hdl[0].header[ifu_soft_key] != int(version):
            print('Inconsistent software version!')
            print(('This script veision: %s'%version))
            print(('Fits file: %s'%hdl[0].header[ifu_soft_key]))
            return False
    else:
        print('This has never been reduced by FOACSIFU.')
        return False
    return True


def check_version_f(infile):
    hdl = fits.open(infile)
    stat = check_version(hdl)
    hdl.close()
    return stat

def cross_correlate(indata, refdata, sep=0.01, kind='cubic', fit=False, \
                    niteration=3, high_nsig=3.0, low_nsig=3.0):

    x = np.array(list(range(len(indata))))
    xnew = np.arange(1, len(indata)-1, sep)

    f_CS = interp1d(x, indata, kind=kind)
    innew = f_CS(xnew)
    if fit:
        coef, weight = cheb1Dfit(xnew, innew ,order=1, weight=None, \
                            niteration=niteration, \
                            high_nsig=high_nsig, low_nsig=low_nsig)
        insub = innew - chebval(xnew, coef)
    else:
        insub = innew

    f_CS = interp1d(x, refdata, kind=kind)
    refnew = f_CS(xnew)
    if fit:
        coef, weight = cheb1Dfit(xnew, refnew ,order=1, weight=None, \
                            niteration=niteration, \
                            high_nsig=high_nsig, low_nsig=low_nsig)
        refsub = refnew - chebval(xnew, coef)
    else:
        refsub = refnew

    corr = np.correlate(insub, refsub, 'full')
    delta = corr.argmax() - (len(insub) - 1)
    delta = delta * sep

    return delta

def gaussfit1d(y):
    global click, ii
    click = np.zeros((2,2))
    ii=0

    def on_key(event):
        global click, ii
        #print 'event.key=%s,  event.x=%d, event.y=%d, event.xdata=%f, \
        #event.ydata=%f'%(event.key, event.x, event.y, event.xdata, event.ydata)
        click[ii,0] = event.xdata
        click[ii,1] = event.ydata
        ii = ii + 1
        return

    x = np.array(list(range(len(y))))
    print(('Press any key at two points for specifying the '+\
          'fitting region and the baselevel.'))
    fig=plt.figure()
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.plot(x,y)
    plt.draw()
    while ii != 2:
        plt.pause(0.1)

    x1 = int(np.min(click[:,0])+0.5)
    x2 = int(np.max(click[:,0])+0.5)

    m = (click[1,1]-click[0,1])/(click[1,0]-click[0,0])
    yy = y - m * ( x - click[0,0]) - click[0,1]

    # Fit the data using a Gaussian
    g_init = Gaussian1D(amplitude=np.min(yy[x1:x2+1]), \
                               mean=(x1+x2)/2.0, stddev=1.)
    fit_g = LevMarLSQFitter()
    g = fit_g(g_init, x[x1:x2+1], yy[x1:x2+1])
    #print(g)

    plt.plot((x1,x2), (y[x1],y[x2]))
    xx = np.linspace(x1,x2,100)
    yy = g(xx) + m * (xx - click[0,0]) + click[0,1]
    plt.plot(xx,yy)
    plt.draw()
    try:
        plt.pause(-1)
    except:
        pass
    return g
