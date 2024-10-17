
import typing

import numpy as np
from scipy.interpolate import interp1d


#todo: aprimorar fazendo interpolacao entre os dois pontos que cruzam a metade
# full width at half maximum; largura a meia altura
def FWHM(x,y):
    """
    Calculates the Full Width at Half Maximum (FWHM) of a Gaussian-type function.

    Args:
        x: Array of x-values.
        y: Array of corresponding y-values (function values).

    Returns:
        The FWHM value.
    """
    half_max = np.max(y) / 2
    # Find the indices of the points closest to half maximum on either side of the peak
    left_idx = np.argmin(np.abs(y[:np.argmax(y)] - half_max))
    right_idx = np.argmin(np.abs(y[np.argmax(y):] - half_max)) + np.argmax(y)

    # Calculate the FWHM
    return x[right_idx] - x[left_idx], left_idx, right_idx


#todo: test later
def calculate_fwhm(x, y):
  """
  Calculates the Full Width at Half Maximum (FWHM) of a peak.

  Args:
    x: Array of x-values.
    y: Array of y-values.

  Returns:
    The FWHM value.
  """

  half_max = np.max(y) / 2
  
  # find when function crosses line half_max (when sign of diff flips)
  # take the 'derivative' of signum(half_max - y[])
  d = np.sign(half_max - np.array(y[0:-1])) - np.sign(half_max - np.array(y[1:]))
  
  # find the left and right most indexes
  l = np.where(d > 0)[0][0]
  r = np.where(d < 0)[0][-1]

  # Use linear interpolation to find the x-value at half_max
  left_x = x[l] + (x[l+1] - x[l]) * ((half_max - y[l]) / (y[l+1] - y[l]))
  right_x = x[r] + (x[r+1] - x[r]) * ((half_max - y[r]) / (y[r+1] - y[r]))

  return right_x - left_x


def SR_fwhm(SR,
            coord:typing.Literal['x','y'],energy:float,X:float,Y:float,
            polarization='total',intType='SE'):
    """FWHM of PSF [um]"""
    arrIxi, [rangexi] = SR.calc_intensity(coord,energy,X,Y,polarization,intType)
    xi = np.linspace(*rangexi)*1e6

    fwhm = calculate_fwhm(xi,arrIxi)

    return fwhm



def FWHM_to_RMS_g(fwhm):
    """conversor for a gaussian distribution"""
    c = 2*np.sqrt(2*np.log(2))
    sigma = fwhm/c
    return sigma

def RMS_to_FWHM_g(rms):
    """conversor for a gaussian distribution"""
    c = 2*np.sqrt(2*np.log(2))
    fwhm = c*rms
    return fwhm


#todo: talvez aplicar o resize do srwlib, conferir como ele funciona
def resize_1d(x0,y0,x):
    f = interp1d(x0,y0)
    return f(x)







#todo: testar se os f0, f1, f2, f3, f4 precisam ser necessariamente igualmente espaçados (acredito que não)
# simpson rule
def S(h,f,x0):
    return (h/3)*(f(x0)+4*f(x0+h)+2*f(x0+2*h)+4*f(x0+3*h)+f(x0+4*h))

#todo: upgrade to accept xi, xf nd arrays
# integrate arbitrary interval
def isimpson(f,xi,xf,N):
    # works with:
    # - real or complex function f
    # - xi, xf numbers
    # - xi or xf 1d arrays
    h = (xf-xi)/(4*N)
    integral = [S(h,f,xi+n*4*h) for n in range(N)]
    return sum(integral)

