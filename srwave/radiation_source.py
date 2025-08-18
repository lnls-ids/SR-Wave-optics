
import json
import copy
from typing import Optional, Literal, Union, overload

import numpy as np
import matplotlib.pyplot as plt

from . import opt_elements as oe
from . import mag_elements as me
from .beamtemp import Beam

import srwpy.srwlib as srw
import srwpy.srwlpy as srwl




# SynchrotronRadiation
_Num_Arr = Union[float,list[float],np.ndarray]
_Arr = Union[list[float],np.ndarray]
_Trajectory = Optional[srw.SRWLPrtTrj]
_MagCnt = Optional[me.MagnetCnt]
_Beam = Optional[Beam]
_Polarization = Literal['linH', 'sigma', 'linV', 'pi',
                        'lin45', 'lin135',
                        'circR', 'circL',
                        'total']
_Intensity = Literal['SE I', 'ME I', 'SE Fluence', 'SE J',
                     'SE F', 'ME F',
                     'SE P', 'SE ReE', 'SE ImE']
_IntensityI = Literal['SE', 'ME', 'SE Fluence', 'SE J']
_IntensityF = Literal['SE', 'ME']
_Part = Literal['phase', 're', 'im', 'all']
_Coordinate = Literal['e', 'x', 'y', 'xy', 'ex', 'ey', 'exy']
_Lim = Union[list[float],tuple[float],None]
_UnitXY = Literal['m','cm','mm','um']



# possivelmente criar beamline sem nada vai dar erro ao acessar atributo
class Beamline:

    @overload
    def __init__(self,line:Optional[oe.OpticalElement]=...): ...
    @overload
    def __init__(self,line:Optional[list[oe.OpticalElement]]=...): ...

    def __init__(self,line=None):

        self.wfr = None 
        
        self.opt_elements = line


    @property
    def opt_elements(self) -> list[oe.OpticalElement]:
        """List of optical elements in the beamline"""
        return self._opt_elements
    
    @opt_elements.setter
    def opt_elements(self,elements):

        self._opt_elements = []

        if isinstance(elements, (list,tuple)):
            for element in elements:
                self.add_opt_element(element)
        elif elements:
            self.add_opt_element(elements)
            

    @property
    def srw_opts(self) -> list[srw.SRWLOpt]:
        """List of SRW optical elements in the beamline"""
        return [element.srw_opt for element in self.opt_elements]

    @property
    def props_params(self) -> list[list]:
        """List of propagation parameters of the beamline optical elements"""
        return [list(element.prop_params) for element in self.opt_elements]

    def add_opt_element(self,element:oe.OpticalElement):

        if not isinstance(element,oe.OpticalElement):
            raise TypeError("Optical Element type" + \
                            f" '{type(element).__name__}' " + \
                            "not supported")
        
        self._opt_elements.append(element)




#todo: metodo para return todos os dados importantes da wavefront para ela poder ser inicializavel em outro lugar
class SynchrotronRadiation:

    def __init__(self,
            energy:_Num_Arr,
            d:float,
            x:_Num_Arr,
            y:_Num_Arr,
            traj:_Trajectory=None,
            fields:_MagCnt=None,
            beam:_Beam=None,
            store_steps=False
        ):
        """
        Basic Synchrotron Radiation class. Calculates radiation wavefront from
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
        store_steps : bool, optional
            Store light source wavefront or not.
        """
        self.wfr = srw.SRWLWfr()
        self.wfr.partBeam = Beam() if beam is None else beam

        self.trajectory = traj

        self.fields = fields

        self.precisions = {'method':fields.fieldtype,'prec':0.005,'zstart':0.0,
                           'zend':0.0,'np':50000,'usetermin':True,'sampling':0}

        if np.isscalar(x): x = [x]
        if np.isscalar(y): y = [y]
        if np.isscalar(energy): energy = [energy]

        self.ExCnt = []
        self.EyCnt = []

        self.set_mesh(x,y,energy,d)
    



    def __str__(self):
        pass

    def __repr__(self):
        pass


    def setBeam(self,eqparams):
        self.wfr.partBeam = Beam(eqparams)

    def load_beam(self, beam="carcara", isTwiss=True):
        mode = 'twiss' if isTwiss else 'rms'
        beams_file = __file__.replace("radiation_source.py","beams.json")
        with open(beams_file) as b:
            beams = json.load(b)
        eqparams = list(beams[beam][mode].values())
        self.wfr.partBeam = Beam(eqparams=eqparams)



    def set_mesh(self,x:_Arr,y:_Arr,e:_Arr,d:float):
        """
        Initializes the wavefront mesh, i.e., the radiation sampling of the
        initial wavefront (before optical elements)

        Args:
            x (array): Horizontal positions [m].
            y (array): Vertical positions [m].
            e (array): Photon energies [eV].
            d (float): Longitudinal position for initial wavefront [m].
        """

        xi, xf, nx = x[0], x[-1], len(x)
        yi, yf, ny = y[0], y[-1], len(y)
        ei, ef, ne = e[0], e[-1], len(e)

        # number of points of photon energy, horizontal and vertical positions
        self.wfr.allocate(ne,nx,ny)

        # mesh
        self.wfr.mesh.eStart = ei #initial energy
        self.wfr.mesh.eFin   = ef #final energy
        self.wfr.mesh.xStart = xi #initial horizontal position [m]
        self.wfr.mesh.xFin   = xf #final horizontal position [m]
        self.wfr.mesh.yStart = yi #initial vertical position [m]
        self.wfr.mesh.yFin   = yf #final vertical position [m]
        self.wfr.mesh.zStart = d  #longitudinal position for initial wfr [m]

    def window_limits(self):
        mesh = self.wfr.mesh
        return np.array([mesh.xStart,mesh.xFin,mesh.yStart,mesh.yFin])
        

    def set_wfr_elec_field(self, arrEx, arrEy):
        self.arEx = copy.copy(arrEx)
        self.arEy = copy.copy(arrEy)



    def CalcElecFieldSR(self, store_steps=False):
        precisions = list(self.precisions.values())

        srwl.CalcElecFieldSR(self.wfr,self.trajectory,self.fields.cnt,
                             precisions)

        if store_steps:
            self.ExCnt = [copy.copy(self.arEx)]
            self.EyCnt = [copy.copy(self.arEy)]

    def calc_wfr(self, store_steps=False):
        self.CalcElecFieldSR(store_steps)

    # def calcWfr(self, partTraj, magFldCnt, precisions, store_steps=True):

    #     srwl.CalcElecFieldSR(self, partTraj, magFldCnt, precisions)

    #     if store_steps:
    #         self.ExCnt = [copy.copy(self.arEx)]
    #         self.EyCnt = [copy.copy(self.arEy)]

    def PropagElecField(self, oe_arr, pp_arr):
        optBl = srw.SRWLOptC(oe_arr,pp_arr)
        srwl.PropagElecField(self.wfr, optBl)

    def propagate_wfr(self, beamline:Beamline, store_steps=False):

        if not beamline.opt_elements:
            return False

        oe_arr, pp_arr = beamline.srw_opts, beamline.props_params

        if not store_steps:
            self.PropagElecField(oe_arr, pp_arr)

        else:
            for srwopt, prop_params in zip(oe_arr, pp_arr):
                
                self.PropagElecField([srwopt],[prop_params])

                self.ExCnt.append(self.arEx)
                self.EyCnt.append(self.arEy)

        return True


    def IntFromElecField(self,
            pol:_Polarization,
            intType:_Intensity,
            coords:_Coordinate,
            energy:float,
            X:float,
            Y:float
        ):
        """
        Wrapper of SRW's srwlpy.CalcIntFromElecField() function.

        Parameters
        ----------
        pol : {'linH', 'linV', 'lin45', 'lin135', 'circR', 'circL', 'total'}
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
        idx_pol = {'linH':0,'sigma':0,'linV':1,'pi':1,'lin45':2,'lin135':3,
                   'circR':4,'circL':5,
                   'total':6}.get(pol)
        idx_type = {'SE I':0,'ME I':1,'SE F':2,'ME F':3,
                    'SE P':4,'SE ReE':5,'SE ImE':6,
                    'SE Fluence':7,'SE J':8}.get(intType)
        idx_coord={'e':0,'x':1,'y':2,'xy':3,'ex':4,'ey':5,'exy':6}.get(coords)

        if None in [idx_pol,idx_type,idx_coord]:
            raise ValueError('Invalid arguments! Check their writing.')

        N = np.prod([getattr(self.wfr.mesh, f'n{coord}') for coord in coords])
        intervals = [[getattr(self.wfr.mesh,f'{coord}Start'),
                    getattr(self.wfr.mesh,f'{coord}Fin'),
                    getattr(self.wfr.mesh,f'n{coord}')] for coord in coords]

        isPhase = intType=='SE P'

        arrI = np.zeros(N, dtype=np.float64 if isPhase else np.float32)
        arrI: np.ndarray = srwl.CalcIntFromElecField(
            arrI, self.wfr, idx_pol, idx_type, idx_coord, energy, X, Y
        )

        return arrI, intervals

    #todo: aceitar polarization='sigma' e 'pi' tambem
    def calc_intensity(self,
            coords:_Coordinate,
            energy:float,
            X:float,
            Y:float,
            polarization:_Polarization='total',
            intType:_IntensityI='SE'
        ):
        """
        Calculate the intensity of the radiation wavefront.

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
        polarization : {'total', 'linH', 'linV', 'lin45', 'lin135', 'circR',
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
        if intType in ['SE','ME']:
            intType += ' I'
        elif intType == 'SE J':
            return "SRW C++ calculation apparently is not well finished, then \
                    the mutual intensity is not available yet."
        elif intType != 'SE Fluence':
            raise ValueError('Invalid intensity type!')

        coords_lst = coords.split('-')

        if len(coords_lst) == 1:
            return self.IntFromElecField(
                polarization, intType, coords, energy, X, Y
            )
        else:
            return [
                self.IntFromElecField(polarization,intType,coord,energy,X,Y)
                for coord in coords_lst
            ]

    def calc_flux(self,
            polarization:_Polarization='total',
            intType:_IntensityF='SE'
        ):
        """
        Calculate the flux of the radiation wavefront.

        Parameters
        ----------
        polarization : {'total', 'linH', 'linV', 'lin45', 'lin135', 'circR',
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
        if intType in ['SE','ME']:
            intType += ' F'
        else:
            raise ValueError('Invalid flux type!')

        coords = 'e'
        energy = 0; X = 0; Y = 0 # dummy values; not used
        
        return self.IntFromElecField(polarization, intType, coords, energy,X,Y)


    def calc_electric_field(self,
            part:_Part,
            coords:_Coordinate,
            energy:float,
            X:float,
            Y:float,
            polarization:_Polarization='total'
        ):
        """
        Calculate the electric field components of the radiation wavefront.

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
        polarization : {'total', 'linH', 'linV', 'lin45', 'lin135', 'circR',
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
        if part=='all':

            intType_lst = ['SE ReE','SE ImE','SE P']
            coords_lst = coords.split('-')

            if len(coords_lst) == 1:
                return [self.IntFromElecField(polarization, intType, coords, energy, X, Y)
                            for intType in intType_lst]
            else:
                return [[self.IntFromElecField(polarization, intType, coord, energy, X, Y)
                            for coord in coords_lst]
                            for intType in intType_lst]

        intType = {'phase':'SE P', 're':'SE ReE', 'im':'SE ImE'}[part]
        coords_lst = coords.split('-')

        if len(coords_lst) == 1:
            return self.IntFromElecField(polarization, intType, coords, energy, X, Y)
        else:
            return [self.IntFromElecField(polarization, intType, coord, energy, X, Y)
                        for coord in coords_lst]

    @staticmethod
    def unwrap_phase(wrapped_phase: np.ndarray) -> np.ndarray:
        """
        Unwraps phase 1D or 2D array. Unwrapping is stack the intervals of 2pi
        of phase values.
        """
        if wrapped_phase.ndim == 1:
            unwrapped_phase =  np.unwrap(wrapped_phase)
        elif wrapped_phase.ndim == 2:
            unwrapped_phase = np.unwrap(np.unwrap(wrapped_phase,axis=0),axis=1)
        return unwrapped_phase
    
    @staticmethod
    def wrap_phase(unwrapped_phase: np.ndarray) -> np.ndarray:
        """Wraps phase 1D or 2D array into the range [-pi,pi]."""
        wrapped_phase = ((unwrapped_phase-np.pi) % (2*np.pi)) - np.pi
        return wrapped_phase
    

    #todo: aceitar apenas numero, para poder contar so um plano ou segmento
    def count_points(self,
            xlim:_Lim=None,
            ylim:_Lim=None,
            elim:_Lim=None
        ) -> int:
        """
        Counts the number of points within specified limits across horizontal
        and vertical positions and energy for the wavefront.

        Parameters
        ----------
        xlim : array, optional
            Horizontal range. Defaults to the full range.
        ylim : array, optional
            Vertical range. Defaults to the full range.
        elim : array, optional
            Energy range. Defaults to the full range.

        """
        mesh = self.wfr.mesh
        xi, xf, nx = mesh.xStart, mesh.xFin, mesh.nx
        yi, yf, ny = mesh.yStart, mesh.yFin, mesh.ny
        ei, ef, ne = mesh.eStart, mesh.eFin, mesh.ne

        x = np.linspace(xi,xf,nx)
        y = np.linspace(yi,yf,ny)
        e = np.linspace(ei,ef,ne)

        if not xlim: xlim = (xi,xf)
        if not ylim: ylim = (yi,yf)
        if not elim: elim = (ei,ef)

        maskx = (xlim[0] <= x) & (x <= xlim[1])
        masky = (ylim[0] <= y) & (y <= ylim[1])
        maske = (elim[0] <= e) & (e <= elim[1])

        return np.sum(maskx) * np.sum(masky) * np.sum(maske)
    

    #todo: testar comportamento de kwargs e nao apagar setting se ja foi feito antes
    #?: deixar plotar so' caracteristicas espaciais, ou com energia tambem?
    #todo: energia, X e Y nao obrigatorios
    #todo: assumir que X e Y ja sao 0 e quem quiser deixar diferente vai e muda
    #todo: projx e projy somando cada eixo complementar
    def plot_intensity(self,
            # intensity configuration arguments #
            coords:_Coordinate,
            energy:float=0,
            X:float=0,
            Y:float=0,
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
            Additional keyword arguments passed to the plotting functions.
            `xlim`, `ylim`, `title`, `xlabel`, `ylabel` are also accepted
            as keyword arguments.

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

        arrI, intervals = self.calc_intensity(coords,energy,X,Y,
                                              polarization,intType)

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

    def plot_electric_field(self,
            coords: str,
            energy: float,
            X: float,
            Y: float,
            output = False,
            polarization='total',
            part='re',
            ax=None,xlim=None,ylim=None,xlabel='',ylabel='',show=True,**kwargs
        ):

        if not ax:
            _, ax = plt.subplots()

        arrE, intervals = self.calc_electric_field(part, coords, energy, X, Y,
                                                    polarization)

        if len(coords) == 1:
            
            rangei, = intervals
            xi = np.linspace(*rangei)

            ax.plot(xi*1e6,arrE)

        elif len(coords) == 2:

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

