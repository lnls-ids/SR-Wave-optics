
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from . import radiation_source as rs
from . import mag_elements as me





_Coordinate = Literal['e', 'x', 'y', 'xy', 'ex', 'ey', 'exy']
_CoordSpectrum = Literal['e', 'ex', 'ey', 'exy']
_Polarization = Literal['linH', 'linV',
                        'lin45', 'lin135',
                        'circR', 'circL',
                        'total']
_IntensityI = Literal['SE', 'ME', 'SE Fluence', 'SE J']
_UnitXY = Literal['m','cm','mm','um']





#todo: colocar coords permitidas apenas as que contem 'e'
def plot_spectrum(SR: rs.SynchrotronRadiation,
        X:float,
        Y:float,
        coords:_CoordSpectrum='e',
        polarization='total',
        intType='SE',
        ax=None,
        show=True,
        **kwargs
    ):

    """
    Plot the spectral intensity of the radiation wavefront.

    Parameters
    ----------
    SR : SynchrotronRadiation
        Instance of the SynchrotronRadiation class.
    X,Y : float
        Horizontal and vertical position for calculation.
    coords : {'e', 'ex', 'ey', 'exy'}, optional
        Coordinates for calculation, either 'e', 'ex', 'ey', or 'exy'.
        Defaults to 'e'.
    polarization : {'total', 'linH', 'linV', 'lin45', 'lin135', 'circR',
        'circL'}, optional\n
        Polarization of the radiation. Defaults to 'total'.
    intType : {'SE', 'ME', 'SE Fluence', 'SE J'}, optional
        Intensity type. Defaults to 'SE'.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. Defaults to None.
    show : bool, optional
        Display the plot. Defaults to True.
    **kwargs : dict, optional
        Additional keyword arguments passed to the plotting functions.
        'title', 'xlabel', 'ylabel', 'xlim', 'ylim', 'xscale', 'yscale' are
        also accepted as keyword arguments.
    """
    if not ax:
        ax = plt.subplot()
    
    # default labels
    ax.set(xlabel='Energy [keV]',
           ylabel=r'Spectral Intensity [ph/s/0.1%bw/mm$^2$]')

    settings = {setting: kwargs.pop(setting)
                for setting in ['title','xlabel','ylabel','xlim','ylim',
                                'xscale','yscale']
                    if setting in kwargs}
    ax.set(**settings)

    energy = 0 # dummy value
    arrI, [rangee] = SR.calc_intensity(coords, energy, X, Y,
                                       polarization,intType)
    e = np.linspace(*rangee)

    ax.plot(e*1e-3,arrI)
    
    ax.grid()

    if show:
        plt.show()


#todo: testar comportamento de kwargs e nao apagar setting se ja foi feito antes
#?: deixar plotar so' caracteristicas espaciais, ou com energia tambem?
def plot_wfr_inten(SR: rs.SynchrotronRadiation,
        # intensity configuration arguments #
        coords:_Coordinate,
        energy:float,
        X:float,
        Y:float,
        polarization:_Polarization='total',
        intType:_IntensityI='SE',
        # data manipulation arguments #
        unit:_UnitXY='um',
        normalize=False,
        output=False,
        # plot configuration arguments #
        ax=None,
        legend=None,
        **kwargs
    ):
    """
    Plot the intensity of the synchrotron radiation wavefront.

    Parameters
    ----------
    SR : SynchrotronRadiation
        Instance of the SynchrotronRadiation class.
    coords : {'x', 'y', 'xy', 'ex', 'ey', 'exy'}
        Coordinates for plotting, either 'x', 'y', or 'xy'.
    energy : float
        Energy value for which the intensity is calculated.
    X,Y : float
        Horizontal and vertical position for calculation.
    polarization : str, optional
        The polarization of the radiation. Defaults to 'total'.
    intType : str, optional
        Intensity type. Defaults to 'SE'.
    unit : {'m', 'cm', 'mm', 'um'}, optional
        Unit of the transverse spatial coordinates. Defaults to 'um'.
    normalize : bool, optional
        Normalize the intensity. Defaults to False.
    output : bool, optional
        Return the intensity and intervals. Defaults to False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. Defaults to None.
    legend : str, optional
        Label for the plot legend. Defaults to None.
    **kwargs : dict, optional
        Additional keyword arguments passed to the plotting functions. `xlim`,
        `ylim`, `title`, `xlabel`, `ylabel` are also accepted as keyword
        arguments.

    Returns:
        If `output` is True, returns the intensity and intervals.
    """
    show = False if ax else True

    #?: assim ou plt.subplots ?
    if not ax:
        ax = plt.subplot()

    unit = {'m':1,'cm':1e2,'mm':1e3,'um':1e6}[unit]

    settings = {setting: kwargs.pop(setting)
                for setting in ['xlim','ylim','title','xlabel','ylabel']
                    if setting in kwargs}
    ax.set(**settings)

    arrI, intervals = SR.calc_intensity(coords,energy,X,Y,polarization,intType)

    if len(coords) == 1:
        
        rangei, = intervals
        xi = np.linspace(*rangei)

        normalization = np.max(arrI) if normalize else 1

        ax.plot(xi*unit,arrI/normalization,label=legend,**kwargs)

    elif len(coords) == 2:

        [rangex,rangey] = intervals
        xi, xf, nx, yi, yf, ny = *rangex, *rangey
        arrI = np.array(arrI).reshape(ny,nx)
        limits = np.array([xi,xf,yi,yf])*unit

        im = ax.imshow(arrI,extent=limits,origin='lower',**kwargs)
        if 'cmap' not in kwargs:
            im.set_cmap('gray')
    
    if legend:
        ax.legend()
    if show:
        plt.show()

    if output:
        return arrI, intervals



def plot_wfr_elec_field(SR: rs.SynchrotronRadiation,
                 coords: str, energy: float, X: float, Y: float, output = False,
                 polarization='total',part='re',
                 ax=None,xlim=None,ylim=None,xlabel='',ylabel='',show=True,
                 **kwargs):

    if not ax:
        _, ax = plt.subplots()

    if len(coords) == 1:
        
        arrE, intervals = SR.calc_electric_field(part, coords, energy, X, Y,
                                                polarization)
        rangei, = intervals
        xi = np.linspace(*rangei)

        ax.plot(xi*1e6,arrE)

    elif len(coords) == 2:

        arrE, intervals = SR.calc_electric_field(part, coords, energy, X, Y,
                                                       polarization)
        [rangex,rangey] = intervals
        xi, xf, nx, yi, yf, ny = *rangex, *rangey
        arrE = np.array(arrE).reshape(ny,nx)

        limits = np.array([xi,xf,yi,yf])*1e6
        ax.imshow(arrE,extent=limits,origin='lower',cmap='gray',**kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if show:
        plt.show()

    if output:
        return arrE, intervals




'''
def plot_vrule(ax, x, ymin, ymax, colors=None, linestyles='solid',
               label='', **kwargs):
    
    if not ax:
        ax = plt.subplot()

    ax.vlines(x, ymin, ymax, colors, linestyles,
              label, **kwargs)


def plot_hrule(ax, y, xmin, xmax, colors=None, linestyles='solid',
               label='', **kwargs):
    
    if not ax:
        ax = plt.subplot()

    ax.hlines(y, xmin, xmax, colors, linestyles,
               label, **kwargs)
'''

def add_colorbar(ax,**kwargs):
    """
    Add a colorbar to a given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the colorbar to.
    **kwargs : dict, optional
        Additional keyword arguments passed to the colorbar function.
    """
    im = ax.get_images()[0]
    plt.colorbar(im,**kwargs)











if __name__ == '__main__':

    # case 1 #

    w = np.linspace(-50,50,120)*1e-6
    bm = me.BendingMagnet(B=0.5642,L=2)
    fields = me.MagnetCnt(magnets=[bm],field_type='bending')
    SR = rs.SynchrotronRadiation(energy=20e3,d=10,x=w,y=w,fields=fields)
    SR.calc_wfr()

    arrI, [rangex, rangey] = plot_wfr_inten(
        SR,coords='xy',energy=20e3,X=0,Y=0, output=True, 
        title='Bending radiation - 0.5642 T, 20 keV',
        xlabel='x [um]', ylabel='y [um]', xlim=[-50,50], ylim=[-50,50],
        aspect='auto', show=True
    )


    # case 2 #

    energy = np.linspace(0.1,10e3,500)

    for i, B in enumerate([0.5642,1.4,3.2]):

        bm = me.BendingMagnet(B=B,L=2)
        fields = me.MagnetCnt(magnets=[bm],field_type='bending')
        SR = rs.SynchrotronRadiation(energy=energy,d=10,x=0,y=0,fields=fields)
        SR.calc_wfr()

        ax = plt.subplot(1,3,i)

        plot_spectrum(SR,coords='e',X=0,Y=0, output=True, 
            title='Bending radiation - 0.5642 T, 20 keV',
            xlabel='x [um]', ylabel='y [um]', xlim=[-50,50], ylim=[-50,50],
            aspect='auto', show=True
        )
