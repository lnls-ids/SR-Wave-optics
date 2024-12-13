
import typing
import numpy as np
import srwpy.srwlib as srw
from . import opt_elements as oe


# types for calculations
_Propagator = typing.Literal['Standard', 'Quadratic', 'QuadraticSpecial', 'FromWaist', 'ToWaist']
_Polarization = typing.Literal['linH', 'linV', 'lin45', 'lin135', 'circR', 'circL', 'total']
_Intensity = typing.Literal['SE I', 'ME I', 'SE Fluence', 'SE J', 'SE F', 'ME F', 'SE P', 'SE ReE', 'SE ImE']
_IntensityI = typing.Literal['SE', 'ME', 'SE Fluence', 'SE J']
_IntensityF = typing.Literal['SE', 'ME']
_Part = typing.Literal['phase', 're', 'im', 'all']
_Coordinate = typing.Literal['e', 'x', 'y', 'xy', 'ex', 'ey', 'exy']


class Beam(srw.SRWLPartBeam):

    def __init__(self,I:float=...,E:float=...,isTwiss:bool=...,eqparams:typing.Optional[list]=...,*args,**kwargs):
        ...

    def __str__(self):
        ...

    def __repr__(self):
        ...





class Beamline:

    @typing.overload
    def __init__(self,line:typing.Optional[oe.OpticalElement]=...): ...
    @typing.overload
    def __init__(self,line:typing.Optional[list[oe.OpticalElement]]=...): ...
    
    def add_opt_element(self,element:oe.OpticalElement): ...

    def add_opt_elements(self,elements:list[oe.OpticalElement]): ...

    def opt_elements(self) -> list[oe.OpticalElement]: ...
    
    def props_params(self) -> list[list]: ...
        







class SynchrotronRadiation(srw.SRWLWfr):

    # tem wavefront inicial (ja calculou electric field) e so' quer propagar de outros jeitos, por exemplo
    # @typing.overload
    # def __init__(self, wfr: srw.SRWLWfr):
    #     ...
    
    # calcular a wfr inicial, mas ja tem a trajetoria da particula, entao nao precisa do campo magnetico
    # @typing.overload
    # def __init__(self,energy,d,beam):
    #     ...

    def __init__(self, energy, d, x, y, field:dict,
                 beam:typing.Optional[srw.SRWLPartBeam],**fieldKwargs): ...
    def __str__(self):
        ...

    def __repr__(self):
        ...


    def setBeam(self, eqparams: list): ...
    def load_beam(self, beam="carcara", isTwiss=True): ...

    def setBendingMagnet(self,B,L): ...
    @staticmethod
    def setHarmonicField(B,plane='v',phase0=0,symmetry=1,transverse_coeff=1): ...
    def setUndulator(self,period_length,nr_periods,B,Bsettings=['v',0,'',1]): ...

    @typing.overload
    def setTrajectory(self, element: str, field: srw.SRWLMagFld, relPrec: float): ...
    @typing.overload
    def setTrajectory(self, element: str, field: list, relPrec: float): ...
    @typing.overload
    def setTrajectory(self, element: srw.SRWLPrtTrj): ...
    
    #todo: overload: cls.setWfr(arrReE,arrImE)
    def setWfr(self,x,y,e,d,unit=1): ...

    #todo: overload calcWfr partraj: SRWLPrtTrj
    def calcWfr(self, partTraj:int, magFldCnt:srw.SRWLMagFld, precisions:list, store_steps:bool=...): ...

    def propagateWfr(self, beamline:Beamline, store_steps:bool=...): ...

    def IntFromElecField(self, pol:_Polarization, intType:_Intensity, coords:_Coordinate,
                         energy:float, X:float, Y:float): ...

    def calc_intensity(self, coords:str, energy:float, X:float, Y:float,
                       polarization:_Polarization=..., intType:_IntensityI=...): ...
    
    def calc_flux(self, coords:str, energy:float, X:float, Y:float,
                       polarization:_Polarization=..., intType:_IntensityF=...): ...

    def calc_electric_field(self, part:_Part, coords:str, energy:float, X:float, Y:float,
                            polarization:_Polarization=...): ...
    
    @staticmethod
    def unwrap_phase(wrapped_phase: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def wrap_phase(unwrapped_phase: np.ndarray) -> np.ndarray: ...

    def resize_wfr_srw(self,xrangenew,yrangenew,erangenew=[],method='regular',rsType='pos/ang'):
        ...
    def resize_wfr(self,newxlim,newylim):
        ...
    def slice_wfr(self,xlim,ylim):
        ...

    @classmethod
    def energy_loop_intensity(cls,energy,coords:str,x,y,xnew=None,*clsargs,**clskwargs):
        ...