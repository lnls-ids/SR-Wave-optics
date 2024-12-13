
import numpy as np
import matplotlib.pyplot as plt

from .radiation_source import SynchrotronRadiation


#todo: funcao para simplesmente plotar linha horizontal e vertical em certa pos


#todo: tirar energy, nao faz sentido ter aqui
#todo: colocar coords permitidas apenas as que contem 'e'
def plot_spectrum(SR: SynchrotronRadiation, 
                  energy: float, X: float, Y: float,
                  coords: str = 'e',polarization='total',intType='SE',
                  ax=None,xlim=None,ylim=None,yscale='linear',xscale='linear',
                  title='',
                  xlabel='Energy [keV]',
                  ylabel=r'Spectral Intensity [ph/s/0.1%bw/mm$^2$]',
                  show=True):
    
    arrI, [rangee] = SR.calc_intensity(coords, energy, X, Y,
                                       polarization,intType)
    e = np.linspace(*rangee)

    if not ax:
        _, ax = plt.subplots()

    ax.plot(e*1e-3,arrI)
    
    ax.grid()

    ax.set(xlim=xlim,ylim=ylim,title=title,xlabel=xlabel,ylabel=ylabel,
           yscale=yscale,xscale=xscale)

    if show:
        plt.show()


#todo: testar comportamento de kwargs e nao apagar setting se ja foi feito antes
def plot_wfr_inten(SR: SynchrotronRadiation,
                   coords: str, energy: float, X: float, Y: float,
                   polarization='total',intType='SE',
                   normalize = False, output = False,
                   ax=None,legend=None,show=True,**kwargs):
    """
    Plot the intensity of the synchrotron radiation wavefront.

    Parameters
    ----------
    SR : SynchrotronRadiation
        Instance of the SynchrotronRadiation class.
    coords : str
        Coordinates for plotting, either 'x', 'y', or 'xy'.
    energy : float
        Energy value for which the intensity is calculated.
    X,Y : float
        Horizontal and vertical position for calculation.
    polarization : str, optional
        The polarization of the radiation. Defaults to 'total'.
    intType : str, optional
        Intensity type. Defaults to 'SE'.
    normalize : bool, optional
        Normalize the intensity. Defaults to False.
    output : bool, optional
        Return the intensity and intervals. Defaults to False.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. Defaults to None.
    legend : str, optional
        Label for the plot legend. Defaults to None.
    show : bool, optional
        Display the plot. Defaults to True.
    **kwargs : dict, optional
        Additional keyword arguments passed to the plotting functions. `xlim`,
        `ylim`, `title`, `xlabel`, `ylabel` are also accepted as keyword arguments.

    Returns:
        If `output` is True, returns the intensity and intervals.
    """

    #?: assim ou plt.subplots ?
    if not ax:
        ax = plt.subplot()

    settings = {setting: kwargs.pop(setting)
                for setting in ['xlim','ylim','title','xlabel','ylabel']
                    if setting in kwargs}
    ax.set(**settings)

    arrI, intervals = SR.calc_intensity(coords, energy, X, Y, polarization, intType)

    if len(coords) == 1:
        
        rangei, = intervals
        xi = np.linspace(*rangei)

        normalization = np.max(arrI) if normalize else 1

        ax.plot(xi*1e6,arrI/normalization,label=legend,**kwargs)

    elif len(coords) == 2:

        [rangex,rangey] = intervals
        xi, xf, nx, yi, yf, ny = *rangex, *rangey
        arrI = np.array(arrI).reshape(ny,nx)
        limits = np.array([xi,xf,yi,yf])*1e6

        im = ax.imshow(arrI,extent=limits,origin='lower',**kwargs)
        if 'cmap' not in kwargs:
            im.set_cmap('gray')
    
    if legend:
        ax.legend()
    if show:
        plt.show()

    if output:
        return arrI, intervals



def plot_wfr_elec_field(SR: SynchrotronRadiation,
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


