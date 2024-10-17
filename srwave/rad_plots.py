
import numpy as np
import matplotlib.pyplot as plt

from .radiation_source import SynchrotronRadiation



def plot_spectrum(SR: SynchrotronRadiation, 
                  energy: float, X: float, Y: float,
                  coords: str = 'e',polarization='total',intType='SE',
                  ax=None,show=True):
    
    arrI, [rangee] = SR.calc_intensity(coords, energy, X, Y,
                                       polarization,intType)
    e = np.linspace(*rangee)*1e-3

    if not ax:
        fig, ax = plt.subplots()

    ax.plot(e,arrI)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Spectral Intensity [ph/s/0.1%bw/mm^2]')
    ax.set_yscale()
    if show:
        plt.show()

def plot_int_wfr(SR: SynchrotronRadiation,
                 coords: str, energy: float, X: float, Y: float,
                 polarization='total',intType='SE',
                 ax=None,xlim=None,ylim=None,xlabel='',ylabel='',show=True,
                 **kwargs):

    if not ax:
        _, ax = plt.subplots()

    if len(coords) == 1:
        
        arrI, [rangei] = SR.calc_intensity(coords, energy, X, Y,
                                           polarization,intType)
        xi = np.linspace(*rangei)

        ax.plot(xi*1e6,arrI)

    elif len(coords) == 2:

        arrI, [rangex,rangey] = SR.calc_intensity(coords, energy, X, Y,
                                                polarization,intType)
        xi, xf, nx, yi, yf, ny = *rangex, *rangey
        arrI = np.array(arrI).reshape(ny,nx)

        limits = np.array([xi,xf,yi,yf])*1e6
        ax.imshow(arrI,extent=limits,origin='lower',cmap='gray',**kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if show:
        plt.show()



if __name__ == '__main__':

    SR = SynchrotronRadiation(energy=10e3,d=10,x=[-3e-3,3e-3,200],y=[-3e-3,3e-3,200],field={'BM':[0.5642,3]})
