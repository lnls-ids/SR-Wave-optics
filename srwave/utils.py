
import typing

import numpy as np
import scipy.constants as cte
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# from .radiation_source import SynchrotronRadiation




def find_zeros(pos, data):
    """Find zero positions on data.

    Consecutive data pairs with changing sign data are first determined:
        Pair:       (pos[i], data[i]) and (pos[i+1], data[i+1])
        In which:   data[i] x data[i+1] = -1
    Each data pair results in a zero position between pos[i] and pos[i+1]:
        (data[i+1]*pos[i] - data[i]*pos[i+1]) / (data[i+1] - data[i])
    Which is the position in which the linear interpolation between the
    points crosses the data==0 axis.

    Args:
        pos (numpy.ndarray): Positions list.
        data (numpy.ndarray): Data list.

    Returns:
        numpy.ndarray: List of the zeros' positions.

    Notes:
        Function adapted from the [imaids package](https://github.com/lnls-ima/insertion-devices)
    """

    sign = np.sign(data)

    idxleft, = np.nonzero(sign[:-1] + sign[1:] == 0) # left, before zeros
    idxright = idxleft + 1                           # right, after zeros

    xl = pos[idxleft]   # x left
    xr = pos[idxright]  # x right
    yl = data[idxleft]  # y left
    yr = data[idxright] # y right

    # linear interpolation to find the x-values at height
    zeros = (yr*xl-yl*xr)/(yr-yl)

    return zeros


def wavelength_to_energy(wl):
    """
    wl : float
        wavelength [m]
    """
    return (cte.h*cte.c)/(cte.e*wl)

def calc_gamma(E):
    """E: accelerator energy"""
    E0 = cte.m_e*cte.c**2/cte.e # eV
    gamma = E/E0
    return gamma



# -------------------------------- wavefront -------------------------------- #

def resize_1d(x0,y0,x):
    f = interp1d(x0,y0)
    return f(x)


def convolution():
    pass



# ---------------------------------- width ---------------------------------- #

def fw_calc(x,y,hfactor):
    """
    Calculates the Full Width at some fraction of the data maximum.
    
    Args:
        x (numpy.ndarray): x values
        y (numpy.ndarray): y values
        hfactor (float): fraction of the maximum
    
    Returns:
        float: full width
        numpy.ndarray: x values where the half maximum crosses the data
    """
    zeros = find_zeros(x,y-hfactor*np.max(y))
    return zeros[-1]-zeros[0], zeros

def fwhm_calc(x,y):
    """
    Calculates the Full Width at Half Maximum (fwhm) of some data.
    
    Args:
        x (numpy.ndarray): x values
        y (numpy.ndarray): y values
    
    Returns:
        float: fwhm
        numpy.ndarray: x values where the half maximum crosses the data
    """
    return fw_calc(x,y,0.5)

def fwhm_SR(SR,
            coord:typing.Literal['x','y'],energy:float,X:float,Y:float,
            polarization='total',intType='SE'):
    """FWHM of PSF [um]"""
    arrIxn, [rangexn] = SR.calc_intensity(coord,energy,X,Y,polarization,intType)
    xn = np.linspace(*rangexn)*1e6

    fwhm, _ = fwhm_calc(xn,arrIxn)

    return fwhm



# --------------------------- gaussian functions --------------------------- #

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gaussian_fit(x, data):
    """
    Fits a gaussian curve to data points using least squares.

    Parameters
    ----------
    x : array_like
        x values
    data : array_like
        y values

    Returns
    -------
    params : array_like
        parameters of the gaussian curve in the form (a, x0, sigma)
    """
    
    initial_guess = [np.max(data), x[np.argmax(data)], 1]
    params, _ = curve_fit(gaussian, x, data, p0=initial_guess)
    
    return params

def gaussian_fwhm_to_rms(fwhm):
    """Conversor for a gaussian distribution"""
    c = 2*np.sqrt(2*np.log(2))
    sigma = fwhm/c
    return sigma

def gaussian_rms_to_fwhm(rms):
    """Conversor for a gaussian distribution"""
    c = 2*np.sqrt(2*np.log(2))
    fwhm = c*rms
    return fwhm



# -------------------------- numerical integration -------------------------- #

# simpson rule
def S(h,f,x0):
    return (h/3)*(f(x0)+4*f(x0+h)+2*f(x0+2*h)+4*f(x0+3*h)+f(x0+4*h))

# integrate arbitrary interval
def isimpson(f,xi,xf,N):
    # works with:
    # - real or complex function f
    # - xi, xf numbers
    # - xi or xf 1d arrays
    h = (xf-xi)/(4*N)
    integral = [S(h,f,xi+n*4*h) for n in range(N)]
    return sum(integral)


