
import json
import copy
from array import array
from typing import Optional, Literal, Union

import numpy as np
import matplotlib.pyplot as plt

import srwpy.srwlib as srw
import srwpy.srwlpy as srwl

from . import magnets
from .beamtemp import Beam


# SynchrotronRadiation
_Num_Arr = Union[float, list[float], np.ndarray]
_Arr = Union[list[float], np.ndarray]
_Trajectory = Optional[srw.SRWLPrtTrj]
_MagCnt = Optional[magnets.MagnetCnt]
_Beam = Optional[Beam]
_Polarization = Literal[
    'linH',
    'sigma',
    'linV',
    'pi',
    'lin45',
    'lin135',
    'circR',
    'circL',
    'total',
]
_Intensity = Literal[
    'SE I',
    'ME I',
    'SE Fluence',
    'SE J',
    'SE F',
    'ME F',
    'SE P',
    'SE ReE',
    'SE ImE',
]
_IntensityI = Literal['SE', 'ME', 'SE Fluence', 'SE J']
_IntensityF = Literal['SE', 'ME']
_Part = Literal['phase', 're', 'im', 'all']
_Coordinate = Literal['e', 'x', 'y', 'xy', 'ex', 'ey', 'exy']
_CoordSpectrum = Literal['e', 'ex', 'ey', 'exy']
_Lim = Union[list[float], tuple[float], bool]
_Prefix = Literal['G', 'M', 'k', 'h', 'da', '', 'd', 'c', 'm', 'u', 'n']

invfieldtype_map = {
    0: 'manual',
    1: 'undulator',
    2: 'bending',
}

pol_map = {
    'linH': 0,
    'sigma': 0,
    0: 0,
    'linV': 1,
    'pi': 1,
    1: 1,
    'lin45': 2,
    2: 2,
    'lin135': 3,
    3: 3,
    'circR': 4,
    4: 4,
    'circL': 5,
    5: 5,
    'total': 6,
    6: 6,
}
calctype_map = {
    'SE I': 0,
    0: 0,
    'ME I': 1,
    1: 1,
    'SE F': 2,
    2: 2,
    'ME F': 3,
    3: 3,
    'SE P': 4,
    4: 4,
    'SE ReE': 5,
    5: 5,
    'SE ImE': 6,
    6: 6,
    'SE Fluence': 7,
    7: 7,
    'SE J': 8,
    8: 8,
}
part_map = {
    're': 'SE ReE',
    'im': 'SE ImE',
    'phase': 'SE P',
}
coords_map = {
    'e': 0,
    0: 0,
    'x': 1,
    1: 1,
    'y': 2,
    2: 2,
    'xy': 3,
    3: 3,
    'ex': 4,
    4: 4,
    'ey': 5,
    5: 5,
    'exy': 6,
    6: 6,
}

prefix_map = {
    'G': 1e-9,
    'M': 1e-6,
    'k': 1e-3,
    'h': 1e-2,
    'da': 1e-1,
    '': 1,
    'd': 1e1,
    'c': 1e2,
    'm': 1e3,
    'u': 1e6,
    'n': 1e9,
}
uprefix_map = {
    'u': r'$\mu$',
}
quantity_map = {
    'SE I': 'Flux Density',
    'ME I': 'Flux Density',
    'SE F': 'Flux',
    'ME F': 'Flux',
    'phase': 'Phase',
    're': 'Re E',
    'im': 'Im E',
    'SE Fluence': 'Unknown', #*
    'SE J': 'Unknown', #*
}
unit_map = {
    'SE I': r'ph/s/0.1%bw/mm$^2$',
    'ME I': r'ph/s/0.1%bw/mm$^2$',
    'SE F': r'ph/s/0.1%bw',
    'ME F': r'ph/s/0.1%bw',
    'phase': 'rad',
    're': r'sqrt(ph/s/0.1%bw/mm$^2$)',
    'im': r'sqrt(ph/s/0.1%bw/mm$^2$)',
    'SE Fluence': 'Unknown', #*
    'SE J': 'Unknown', #*
}


#todo: metodo para return todos os dados importantes da wavefront para ela poder ser inicializavel em outro lugar
#todo: metodo de str e repr
class RadiationSource:

    def __init__(self, energy: _Num_Arr, x: _Num_Arr, y: _Num_Arr, d: float):
        self.wfr = srw.SRWLWfr()
        self.set_mesh(energy, x, y, d)


    @staticmethod
    def _is_equally_spaced(arr):
        arr = np.atleast_1d(arr)
        if arr.size < 3:
            return True
        diffs = np.diff(arr)
        return np.allclose(diffs, diffs[0])

    #todo: pass docstring to numpy format
    def set_mesh(self, e: _Arr, x: _Arr, y: _Arr, d: float):
        """Initializes the wavefront mesh, i.e., the radiation sampling of the
        initial wavefront (before optical elements)

        Args:
            x (array): Horizontal positions [m].
            y (array): Vertical positions [m].
            e (array): Photon energies [eV].
            d (float): Longitudinal position for initial wavefront [m].
        """
        nTot = 2 # real and imag part for each point
        nMom = 11 # 11 moments for each energy point
        for coord, u in zip(['e', 'x', 'y'], [e, x, y]):
            u = np.atleast_1d(u)
            if not self._is_equally_spaced(u):
                raise ValueError(f'The {coord} array must be equally spaced!')
            ui, uf, nu = u[0], u[-1], len(u)
            setattr(self.wfr.mesh, f'{coord}Start', ui) # initial value
            setattr(self.wfr.mesh, f'{coord}Fin', uf) # final value
            setattr(self.wfr.mesh, f'n{coord}', nu) # number of points
            nTot *= nu
            if coord=='e':
                nMom *= nu
        self.wfr.mesh.zStart = d # longitudinal position for initial wfr [m]

        self.wfr.arEx = np.zeros(nTot, dtype=np.float32)
        self.wfr.arEy = np.zeros(nTot, dtype=np.float32)

        self.wfr.arMomX = np.zeros(nMom, dtype=np.float64)
        self.wfr.arMomY = np.zeros(nMom, dtype=np.float64)

    def window_limits(self, unitprefix: _Prefix = 'u'):
        mesh = self.wfr.mesh
        factor = prefix_map[unitprefix]
        return factor*np.array([mesh.xStart, mesh.xFin, mesh.yStart, mesh.yFin])

    def CalcElecField(self):
        raise NotImplementedError

    def calc_wfr(self):
        self.CalcElecField()

    #todo: mudar X, Y para x, y
    def IntFromElecField(self,
            pol: _Polarization,
            calctype: _Intensity,
            coords: _Coordinate,
            energy: float,
            X: float,
            Y: float
        ):
        """Wrapper of SRW's srwlpy.CalcIntFromElecField() function.

        Parameters
        ----------
        pol : {'linH' or 'sigma', 'linV' or 'pi', 'lin45', 'lin135', 'circR', 'circL', 'total'}
            Polarization of the radiation.
        intType : {'SE I', 'ME I', 'SE F', 'ME F', 'SE P', 'SE ReE', 'SE ImE', 
            'SE Fluence', 'SE J'}\n
            Intensity type.
        coords : {'e', 'x', 'y', 'xy', 'ex', 'ey', 'exy'}
            Coordinates for calculation.
        energy : float
            Energy value for which the intensity is calculated.
        X : float
            Horizontal position for calculation.
        Y : float
            Vertical position for calculation.

        Returns
        -------
        arrI : numpy.ndarray
            Intensity flat array.
        intervals : list
            Intervals of dependencies; list of mesh ranges. Each range follows
            the format [Start, Fin, n]. Example: [xi, xf, nx]
        """
        
        idx_pol = pol_map.get(pol)
        idx_type = calctype_map.get(calctype)
        idx_coords = coords_map.get(coords)

        if None in [idx_pol, idx_type, idx_coords]:
            raise ValueError('Invalid arguments! Check their writing.')

        N = np.prod([getattr(self.wfr.mesh, f'n{coord}') for coord in coords])
        intervals = [[getattr(self.wfr.mesh, f'{coord}Start'),
                    getattr(self.wfr.mesh, f'{coord}Fin'),
                    getattr(self.wfr.mesh, f'n{coord}')] for coord in coords]

        isPhase = calctype == 'SE P'
        floatXX = np.float64 if isPhase else np.float32

        arr = np.zeros(N, dtype=floatXX)
        arr = srwl.CalcIntFromElecField(
            arr, self.wfr, idx_pol, idx_type, idx_coords, energy, X, Y
        )

        return arr, intervals

    #todo: mudar X, Y para x, y
    def calc_intensity(self,
            coords: _Coordinate,
            energy: float = 0,
            X: float = 0,
            Y: float = 0,
            polarization: _Polarization = 'total',
            calctype: _IntensityI = 'SE'
        ):
        """Calculate the intensity of the radiation wavefront.

        Parameters
        ----------
        coords : {'e', 'x', 'y', 'xy', 'ex', 'ey', 'exy'}
            Coordinates for calculation, like 'x', 'y', or combinations of
            them, like 'x-y'.
        energy : float
            Energy value for which the intensity is calculated. Taken into
            account for coords `x`, `y` and `xy`.
        X : float
            Horizontal position for calculation. Taken into account for
            coords `e`, `y` and `ey`.
        Y : float
            Vertical position for calculation. Taken into account for
            coords `e`, `x` and `ex`.
        polarization : {'total', 'linH' or 'sigma', 'linV' or 'pi', 'lin45', 'lin135', 'circR',
            'circL'}, optional\n
            The polarization of the radiation. Defaults to 'total'.
        intType : {'SE', 'ME', 'SE Fluence', 'SE J'}, optional
            Intensity type. Defaults to 'SE'.

            * 'SE' -- Single-electron intensity.
            * 'ME' -- Multi-electron intensity.
            * 'SE Fluence' -- Single-electron fluence, i.e., intensity
            integrated over energy.
            * 'SE J' -- Single-electron mutual intensity. Not avaiable yet!

        Returns
        -------
        arrI : numpy.ndarray
            Intensity flat array.
        intervals : list
            Intervals of coordinates; list of ranges. Each range follows the
            format [coordsStart, coordsFin, nCoords]. Example: [xi, xf, nx]

        Notes
        -----
        The function also allows pass more than one coords to be calculated
        separately. Examples: coords='x-y', coords='x-xy', coords='e-xy'.
        For these cases, the returns are list of returns for each coords.

        """
        if calctype in ['SE', 'ME']:
            calctype += ' I'
        elif calctype == 'SE J':
            return "SRW C++ calculation apparently is not well finished, then \
                    the mutual intensity is not available yet."
        elif calctype != 'SE Fluence':
            raise ValueError('Invalid intensity type!')

        coords_lst = coords.split('-')

        results = [
            self.IntFromElecField(polarization, calctype, coords, energy, X, Y)
            for coords in coords_lst
        ]

        return results[0] if len(results)==1 else results

    def calc_flux(self,
            polarization: _Polarization = 'total',
            calctype: _IntensityF = 'SE'
        ):
        """Calculate the flux of the radiation wavefront.

        Parameters
        ----------
        polarization : {'total', 'linH' or 'sigma', 'linV' or 'pi', 'lin45', 'lin135', 'circR',
            'circL'}, optional\n
            Polarization of the radiation. Defaults to 'total'.
        intType : {'SE', 'ME'}, optional
            Flux type. Defaults to 'SE'.

        Notes
        -----
        The flux is calculated by integrating the intensity of the wavefront
        on its spatial window. Therefore, the only dependency allowed is
        coords='e'. Because of that, fix energy and transverse position X or Y
        is not necessary.

        """
        if calctype in ['SE', 'ME']:
            calctype += ' F'
        else:
            raise ValueError('Invalid flux type!')

        coords = 'e'
        energy = 0; X = 0; Y = 0  # dummy values; not used

        return self.IntFromElecField(polarization, calctype, coords, energy, X, Y)

    #todo: mudar X, Y para x, y
    def calc_electric_field(self,
            part: _Part,
            coords: _Coordinate,
            energy: float = 0,
            X: float = 0,
            Y: float = 0,
            polarization: _Polarization = 'total'
        ):
        """Calculate the electric field components of the radiation wavefront.

        Parameters
        ----------
        part : {'phase', 're', 'im', 'all'}
            Part of the electric field to be calculated.
            
            * 'phase' -- phase of the electric field.
            * 're' -- real part of the electric field.
            * 'im' -- imaginary part of the electric field.
            * 'all' -- all options of electric field.

        coords : {'e', 'x', 'y', 'xy', 'ex', 'ey', 'exy'}
            Coordinates for calculation, like 'x', 'y', or combinations of
            them, like 'x-y'.
        energy : float
            Energy value for which the intensity is calculated. Taken into
            account for coords `x`, `y` and `xy`.
        X : float
            Horizontal position for calculation. Taken into account for
            coords `e`, `y` and `ey`.
        Y : float
            Vertical position for calculation. Taken into account for
            coords `e`, `x` and `ex`.
        polarization : {'total', 'linH' or 'sigma', 'linV' or 'pi', 'lin45', 'lin135', 'circR',
            'circL'}, optional\n
            Polarization of the radiation. Defaults to 'total'.

        Returns
        -------
        arrI : numpy.ndarray
            Intensity flat array.
        intervals : list
            Intervals of coordinates; list of ranges. Each range follows the
            format [coordsStart, coordsFin, nCoords]. Example: [xi, xf, nx].

        Notes
        -----
        For part='all', the returned list of nested list will be nested with
        each part, i.e., a list as [re, im, phase].

        """
        part_lst = part.split('-')
        if 'all' in part: part_lst = ['re', 'im', 'phase']
        calctype_lst = [part_map[part] for part in part_lst]

        coords_lst = coords.split('-')
    
        results = []
        for coords in coords_lst:

            output = [
                self.IntFromElecField(polarization, calctype, coords, energy, X, Y)
                for calctype in calctype_lst
            ]

            arrsE = [arrE for arrE, _ in output]
            if len(arrsE)==1: arrsE = arrsE[0]
            intervals = output[0][1]

            results.append([arrsE, intervals])

        return results[0] if len(results)==1 else results

    



    @staticmethod
    def unwrap_phase(wrapped_phase: np.ndarray) -> np.ndarray:
        """Unwraps phase 1D or 2D array. Unwrapping is stack the intervals of 2pi
        of phase values.
        """
        if wrapped_phase.ndim == 1:
            unwrapped_phase = np.unwrap(wrapped_phase)
        elif wrapped_phase.ndim == 2:
            unwrapped_phase = np.unwrap(np.unwrap(wrapped_phase, axis=0), axis=1)
        return unwrapped_phase

    @staticmethod
    def wrap_phase(unwrapped_phase: np.ndarray) -> np.ndarray:
        """Wraps phase 1D or 2D array into the range [-pi,pi]."""
        wrapped_phase = ((unwrapped_phase-np.pi) % (2*np.pi)) - np.pi
        return wrapped_phase

    def count_points(self,
            xlim: _Lim = True,
            ylim: _Lim = True,
            elim: _Lim = True
        ) -> int:
        """Counts the number of points within specified limits across
        horizontal and vertical positions and energy of the wavefront.

        Parameters
        ----------
        xlim : array or bool, optional
            Horizontal range.
        ylim : array or bool, optional
            Vertical range.
        elim : array or bool, optional
            Energy range.

        Notes
        -----
        By default, True, the full range of points at some dimension are
        counted. If the dimension limits is False, it is not counted.
        """
        mesh = self.wfr.mesh

        counts = 1

        for u, ulim in zip(['e', 'x', 'y'], [elim, xlim, ylim]):
            ui = getattr(mesh,f'{u}Start')
            uf = getattr(mesh,f'{u}Fin')
            nu = getattr(mesh,f'n{u}')
            uarr = np.linspace(ui, uf, nu)

            if ulim:
                if ulim is True: ulim = (ui, uf)
                mask = (ulim[0] <= uarr) & (uarr <= ulim[1])
                counts *= np.sum(mask)

        return counts

    @staticmethod
    def _infer_units(coords, calctype, xunitprefix, yunitprefix, normalize):

        xunitprefix = uprefix_map.get(xunitprefix, xunitprefix)
        yunitprefix = uprefix_map.get(yunitprefix, yunitprefix)
        prefixes = [xunitprefix, yunitprefix]

        labels = []

        for coord, prefix in zip(coords, prefixes):

            if coord in ('x', 'y'):
                label = f'{coord} [{prefix}m]'
            else: # coord == 'e'
                label = f'Energy [{prefix}eV]'

            labels.append(label)

        if len(labels)==1:
            unit = unit_map[calctype] if not normalize else 'a.u.'
            quantity = quantity_map[calctype]
            label = f'{quantity} [{unit}]'
            labels.append(label)

        return labels

    # ainda fazer de plot #
    # X arg para output
    #todo: arg para proj integrada ou cut
    #todo: hintings de cada funcao
    #todo: docstrings de cada funcao
    #todo: not allow 'all' option in electric field
    # X ajeitar pseudo raw plot func
    #todo: checar plot de fluencia e flux e unidade
    # X create unitprefix shortcut for set x and y unit prefix together for spatial plot
    #todo: nao permitir coords ser list, bem como part para elecfield

    def plot_result(self,
            arr,
            intervals,
            coords: str,
            calctype: str,
            xunitprefix: _Prefix = 'u',
            yunitprefix: _Prefix = 'u',
            ax=None,
            **kwargs
        ):
        """
        Generic plot utility used by all plot_* methods.
        
        Kwargs contains
        X xlim, ylim, xscale, yscale, title, xlabel, ylabel,
        X colorbar, grid,
        X normalize, output, unitprefix,
        X the other kwargs of ax.plot and ax.imshow : lw , cmap, aspect , label
        """
        output = kwargs.pop('output', False)
        normalize = kwargs.pop('normalize', False)

        unitprefix = kwargs.pop('unitprefix', None)
        xunitprefix = unitprefix or xunitprefix
        yunitprefix = unitprefix or yunitprefix

        ulabel, vlabel = self._infer_units(coords, calctype, xunitprefix, yunitprefix, normalize)
        kwargs.setdefault('xlabel', ulabel)
        kwargs.setdefault('ylabel', vlabel)

        show = False if ax else True
        if not ax:
            ax = plt.subplot()

        settings = ['xlim', 'ylim', 'xscale', 'yscale', 'title', 'xlabel', 'ylabel']
        settings = {setting: kwargs.pop(setting) for setting in settings
                    if setting in kwargs}
        ax.set(**settings)

        if len(coords) == 1:
            rangeu, = intervals
            u = np.linspace(*rangeu)
            ufactor = prefix_map[xunitprefix]
            
            maximum = np.max(arr) if normalize else 1
            useGrid = kwargs.pop('grid', True)
            ax.plot(ufactor*u, arr/maximum, **kwargs)
            ax.grid(useGrid)

        elif len(coords) == 2:
            [rangeu, rangev] = intervals
            ui, uf, nu = rangeu
            vi, vf, nv = rangev
            arr = np.array(arr).reshape(nv, nu)
            ufactor = prefix_map[xunitprefix]
            vfactor = prefix_map[yunitprefix]
            limits = [ufactor*ui, ufactor*uf, vfactor*vi, vfactor*vf]
            cbar = kwargs.pop('colorbar', False)
            kwargs.setdefault('cmap', 'gray')
            im = ax.imshow(arr, extent=limits, origin='lower', **kwargs)
            if cbar:
                plt.colorbar(im, ax=ax)

        else:
            raise NotImplementedError("3D plotting not supported yet.")

        if 'label' in kwargs:
            ax.legend()

        if show:
            plt.show()

        if output:
            return arr, intervals

    def PlotFromElecField(self,
            # intensity configuration arguments #
            pol: _Polarization,
            calctype: _Intensity,
            coords: _Coordinate,
            energy: float,
            X: float,
            Y: float,
            # data manipulation arguments #
            xunitprefix: _Prefix = '',
            yunitprefix: _Prefix = '',
            # plot configuration arguments #
            ax=None,
            **kwargs
        ):
        """Plot of the radiation wavefront.

        Parameters
        ----------
        coords : {'e', 'x', 'y', 'xy', 'ex', 'ey', 'exy'}
            Coordinates for plotting.
        energy : float
            Energy value for which the intensity is calculated.
        X, Y : float
            Horizontal and vertical position for calculation. Taken in account
            when the plot of `coords` need to specify some position. For
            instance, the plot of x, or ex, must be at some coordinate Y.
        polarization : str, optional
            The polarization of the radiation. Defaults to 'total'.
        intType : {'SE I', 'ME', }, optional
            Intensity type.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Unit of the transverse spatial coordinates. Defaults to 'um'.
        normalize : bool
            Normalize the intensity.
        output : bool, optional
            Return the intensity and intervals.
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes to plot on.
        legend : str, optional
            Label for the plot legend.
        **kwargs : dict, optional
            Additional keyword arguments passed to the plotting functions,
            including `xlim`, `ylim`, `title`, `xlabel` and `ylabel`.

        Returns:
            If `output` is True, same return of the `calc_intensity` method.
        """

        arr, intervals = self.IntFromElecField(pol, calctype, coords, energy, X, Y)

        return self.plot_result(arr, intervals, coords, xunitprefix, yunitprefix, ax, **kwargs)

    def plot_intensity(self,
            coords: _Coordinate,
            energy: float = 0,
            X: float = 0,
            Y: float = 0,
            polarization: _Polarization = 'total',
            calctype: _IntensityI = 'SE',
            xunitprefix: _Prefix = 'u',
            yunitprefix: _Prefix = 'u',
            ax=None,
            **kwargs
        ):
        """Plot the intensity calculated by `calc_intensity()`."""
        arrI, intervals = self.calc_intensity(coords, energy, X, Y, polarization, calctype)

        calctype += ' I'

        return self.plot_result(arrI, intervals, coords, calctype, xunitprefix, yunitprefix,
                        ax, **kwargs)

    def plot_spectrum(self,
            # intensity configuration #
            coords: _CoordSpectrum = 'e',
            X: float = 0,
            Y: float = 0,
            polarization: _Polarization = 'total',
            calctype: _IntensityI = 'SE',
            # data manipulation #
            xunitprefix: _Prefix = 'k',
            yunitprefix: _Prefix = '',
            # plot configuration #
            ax=None,
            **kwargs
        ):
        """Plot the spectral intensity of the radiation wavefront.

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

        if coords not in ['e', 'ex', 'ey', 'exy']:
            raise ValueError('Not a spectrum plot!')

        energy = 0 # dummy value

        return self.plot_intensity(coords, energy, X, Y, polarization, calctype, xunitprefix, yunitprefix, ax, **kwargs)

    def plot_flux(self,
            polarization: _Polarization = 'total',
            calctype: _IntensityF = 'SE',
            xunitprefix: _Prefix = 'k',
            ax=None,
            **kwargs
        ):
        """Plot the flux calculated by `calc_flux()`."""
        arrI, intervals = self.calc_flux(polarization, calctype)

        coords = 'e'  # always energy-dependence for flux
        yunitprefix = '' # dummy value
        calctype += ' F'

        return self.plot_result(arrI, intervals, coords, calctype, xunitprefix, yunitprefix, ax, **kwargs)

    def plot_electric_field(self,
            # intensity configuration
            part: _Part,
            coords: _Coordinate,
            energy: float = 0,
            X: float = 0,
            Y: float = 0,
            polarization: _Polarization = 'total',
            # data manipulation
            xunitprefix: _Prefix = 'u',
            yunitprefix: _Prefix = 'u',
            # plot configuration #
            ax=None,
            **kwargs
        ):
        """Plot the electric field components of the radiation wavefront. calculated by `calc_electric_field()`.
        
        Parameters
        ----------
        part : {'re', 'im', 'phase'} (not 'all')
        """
        arrE, intervals = self.calc_electric_field(part, coords, energy, X, Y, polarization)

        return self.plot_result(arrE, intervals, coords, part, xunitprefix, yunitprefix, ax, **kwargs)



#todo: metodo de str e repr
#todo: tornar beam obrigatorio
class SynchrotronRadiation(RadiationSource):

    def __init__(self,
            energy: _Num_Arr,
            x: _Num_Arr,
            y: _Num_Arr,
            d: float,
            traj: _Trajectory = None,
            fields: _MagCnt = None,
            beam: _Beam = None
        ):
        """Basic Synchrotron Radiation class. Calculates radiation wavefront from
        source and propagates through optical elements.

        Parameters
        ----------
        energy : float or array
            Photon energy [eV].
        d : float
            Distance from source [m].
        x, y : float or array
            Horizontal and vertical position [m].
        traj : srw.SRWLPrtTrj or int
            Trajectory of the particle.
        field : srw.SRWLMagFldC or int
            Magnetic fields exerced over the particle.
        source : {'undulator', 'bending'}, optional
            Magnetic field type.
        beam : Beam, optional
            Particle beam.
        """
        super().__init__(energy, x, y, d)

        self.trajectory = traj
        self.fields = fields
        self.beam = beam
        self.wfr.partBeam = Beam() if beam is None else beam #todo: adjust

        self.precisions = {
            'method': fields.fieldtype,
            'prec': 0.005,
            'zstart': 0.0,
            'zend': 0.0,
            'npart': 50000,
            'usetermin': True,
            'sampling': 0,
        }

    # def __str__(self):

    #     titlestr = "Synchrotron Radiation\n"

    #     # mesh #
    #     mesh = self.wfr.mesh
    #     meshstr = (
    #         "- - Mesh Grid - -\n"
    #         f"Photon Energy [keV] : {mesh.eStart*1e-3} -- {mesh.eFin*1e-3} "
    #         f"({mesh.ne} points)\n"
    #         f"Horizontal [um]     : {mesh.xStart*1e6} -- {mesh.xFin*1e6} "
    #         f"({mesh.nx} points)\n"
    #         f"Vertical [um]       : {mesh.yStart*1e6} -- {mesh.yFin*1e6} "
    #         f"({mesh.ny} points)\n"
    #         f"Distance [m]        : {mesh.zStart}\n"
    #     )
        
    #     # beam #
    #     beam = self.beam
    #     env = self.wfr.partBeam.arStatMom2

    #     beamstr = (
    #         "- - - Beam - - -\n"
    #         f"Charge Energy [GeV]             : {beam.energy*1e-9}\n"
    #         f"Current [A]                     : {beam.current}\n"
    #         f"Size RMS (X x Y) [um]           : {env[]*1e6} x {*1e6}\n"
    #         f"Divergence RMS (X' x Y') [urad] : {*1e6} x {*1e6}\n"
    #         f"Length RMS (Z) [um]             : {}\n"
    #         f"Relative energy spread RMS [%]  : {}\n"
    #     )

    #     # traj #
    #     trajstr = "- - Trajectory - -\n"
    #     if self.trajectory:
    #         trajstr += "\n" # nothing yet
    #     else:
    #         trajstr += f"{None}\n"

    #     # mag #
    #     magstr = "- - Magnetic Field - -\n"
    #     if self.fields:
    #         magnet = self.fields.magnets[0]
    #         fieldtype = invfieldtype_map[self.fields.fieldtype]
    #         magstr += f"Source     : {fieldtype}\n"
    #         if fieldtype=='bending':
    #             magstr += f"Field  [T] : {magnet.B}"
    #         elif fieldtype=='undulator':
    #             magstr += (
    #                 f"Period [mm] : {magnet.period_length}\n"
    #                 f"Nº periods  : {magnet.nr_periods}\n"
    #                 f"Kx          : {magnet.K[0]}\n"
    #                 f"Ky          : {magnet.K[1]}"
    #             )
    #     else:
    #         magstr += f"{None}"

    #     return "\n".join([titlestr, meshstr, beamstr, trajstr, magstr])

    # def __repr__(self):
    #     mesh = self.wfr.mesh
    #     return (f"SynchrotronRadiation("
    #             "d={mesh.zStart}, "
    #             f"e_start={mesh.eStart}, e_fin={mesh.eFin}, "
    #             f"fields={self.fields!r}, beam={self.wfr.partBeam!r})")
    
    def setBeam(self, eqparams):
        self.wfr.partBeam = Beam(eqparams)

    def load_beam(self, beam="carcara", isTwiss=True):
        mode = 'twiss' if isTwiss else 'rms'
        beams_file = __file__.replace("radiation.py", "beams.json")
        with open(beams_file) as b:
            beams = json.load(b)
        eqparams = list(beams[beam][mode].values())
        self.wfr.partBeam = Beam(eqparams=eqparams)

    # def set_wfr_elec_field(self, arrEx, arrEy):
    #     self.arEx = copy.copy(arrEx)
    #     self.arEy = copy.copy(arrEy)

    def CalcElecField(self):
        precisions_lst = [
            self.precisions['method'],
            self.precisions['prec'],
            self.precisions['zstart'],
            self.precisions['zend'],
            self.precisions['npart'],
            self.precisions['usetermin'],
            self.precisions['sampling']
        ]
        srwl.CalcElecFieldSR(self.wfr, self.trajectory, self.fields.cnt,
                             precisions_lst)

    '''
    @classmethod
    def energy_loop_intensity(cls,energy,coords:str,x,y,*clsargs,**clskwargs):
        propa_sc = clskwargs.get('propa_sc')
        coords = coords.split('-')

        arrsI = []

        for E_ph in energy:
            print('e:',E_ph,'eV')
            SR = cls(energy=E_ph,*clsargs,**clskwargs)
            arrsIaux = []
            # rangesaux = []
            for coord in coords:
                # print(coord)
                arrI, ranges = SR.calc_intensity(coord,E_ph,X=x,Y=y)
                if propa_sc == 'From Waist' and (coord == 'x' or coord == 'y'):
                    r, = ranges
                    xold = np.linspace(*r)
                    arrI = uti.resize_1d(xold,arrI,xnew)
                if coord == 'xy':
                    rangex, rangey = ranges
                    arrI = np.array(arrI).reshape(rangey[-1],rangex[-1])
                arrsIaux.append(arrI)
                # rangesaux += ranges
            # print('old:',xold[0]*1e6,xold[-1]*1e6)
            # print('new:',xnew[0]*1e6,xnew[-1]*1e6)
            arrsI.append(arrsIaux)

        arrsI = np.array(arrsI)
        # arrsIx, arrsIy = arrsI[:,0,:], arrsI[:,1,:]
        
        return [arrsI[:,i,:] for i in range(len(coords))], ranges



    @classmethod
    def energy_loop_intensity(cls, energy, coords, energy, X, Y,
                              polarization='total',intType='SE'):
        
        arrsI = []
        ranges = []

        for Eph in energy:

            SR = 

            arrI, intervals = 

    '''


class GaussianBeam(srw.SRWLGsnBm):

    def __init__(self, energy, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.avgPhotEn = energy
        self.pulseEn = 0.001

    def set_waist(self, sigX, sigY, sigT, pos=None, ang=None):

        if not pos: pos = [0, 0, 0]
        if not ang: ang = [0, 0]

        self.x, self.y, self.z = pos
        self.xp, self.yp = ang
        self.sigX = sigX
        self.sigY = sigY
        self.sigT = sigT

    #todo: test total polarization
    def set_polarization(self, pol):
        # if pol=='total':
        #     raise ValueError("There is no ")
        idx_pol = pol_map[pol] + 1
        self.polar = idx_pol

    def set_mode(self, modex, modey):
        """Set the Hermite-Gauss transverse mode"""
        self.mx = modex
        self.my = modey


class GaussianRadiation(RadiationSource):

    def __init__(self, energy, x, y, d, gaussianbeam=None):  # , beam):
        super().__init__(energy, x, y, d)

        self.gaussianbeam = gaussianbeam if gaussianbeam else GaussianBeam(energy)

        # self.wfr.partBeam = Beam() if beam is None else beam

        self.precisions = {'sampling': 0}

    def __str__(self):
        mesh = self.wfr.mesh
        gb = self.gaussianbeam
        
        return (f"Gaussian Radiation Source\n"
                f"-------------------------\n"
                f"Distance:    {mesh.zStart:.4f} m\n"
                f"Energy:      {gb.avgPhotEn:.2f} eV\n"
                f"Waist Size:  {gb.sigX*1e6:.2f} x {gb.sigY*1e6:.2f} um\n"
                f"Waist Pos:   (x={gb.x*1e6:.2f}, y={gb.y*1e6:.2f}) um\n"
                f"Pulse En:    {gb.pulseEn} J")

    def __repr__(self):
        return (f"GaussianRadiation(energy={self.gaussianbeam.avgPhotEn}, "
                f"d={self.wfr.mesh.zStart}, gaussianbeam={self.gaussianbeam!r})")

    def CalcElecField(self):
        precisions = list(self.precisions.values())
        srwl.CalcElecFieldGaussian(self.wfr, self.gaussianbeam, precisions)


class PointSource(srw.SRWLPtSrc):

    def __init__(self, flux=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flux = flux


    def set_position(self, pos):
        if not pos: pos = [0, 0, 0]
        self.x, self.y, self.z = pos

    def set_polarization(self, pol):
        idx_pol = {'linH': 1, 'sigma': 1, 'linV': 2, 'pi': 2, 'lin45': 3, 'lin135': 4,
                   'circR': 5, 'circL': 6, 'radial':7}[pol]
        self.polar = idx_pol


class PointRadiation(RadiationSource):

    def __init__(self, energy, x, y, d, pointsource=None):  # , beam):
        super().__init__(energy, x, y, d)

        self.pointsource = pointsource if pointsource else PointSource()

        # self.wfr.partBeam = Beam() if beam is None else beam

        self.precisions = {'sampling': 0}

    def __str__(self):
        mesh = self.wfr.mesh
        ps = self.pointsource
        
        if mesh.ne > 1:
            e_str = f"[{mesh.eStart:.2f} - {mesh.eFin:.2f}] eV"
        else:
            e_str = f"{mesh.eStart:.2f} eV"

        return (f"Point Source Radiation\n"
                f"----------------------\n"
                f"Distance: {mesh.zStart:.4f} m\n"
                f"Energy:   {e_str}\n"
                f"Origin:   (x={ps.x*1e6:.2f}, y={ps.y*1e6:.2f}) um\n"
                f"Flux:     {ps.flux} ph/s/0.1%bw")

    def __repr__(self):
        return (f"PointRadiation(d={self.wfr.mesh.zStart}, "
                f"flux={self.pointsource.flux}, pointsource={self.pointsource!r})")

    def CalcElecField(self):
        precisions = list(self.precisions.values())
        srwl.CalcElecFieldPointSrc(self.wfr, self.pointsource, precisions)
