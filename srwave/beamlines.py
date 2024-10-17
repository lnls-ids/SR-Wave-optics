


import typing

import numpy as np
import scipy.constants as cte

from . import radiation_source as radsrc
from ..naoajustado_lnls_srw.SRctes import get_beam




# srw propagators dict
_propagators = {'Standard':0,
                'Quadratic':1, 'Quadratic Special':2,
                'From Waist':3, 'To Waist':4}
_type_propas = typing.Literal['Standard',
                              'Quadratic','Quadratic Special',
                              'From Waist','To Waist']





class PinholeLine(radsrc.SynchrotronRadiation):

    def __init__(self,energy,d,D,x,y,B,L,
                 d_Al,apertx,aperty,
                 propa_sc:_type_propas,rs_sc,ra_sc,re_sc):
        
        # beam = list(get_beam('carcara').values())

        super().__init__(energy, d, x, y,
                         field = {'BM':[B,L]})
        
        self.load_beam("carcara")
        
        optBL = radsrc.BeamLine()
        optBL.add(*optBL.Filter(energy=energy,
                                thickness=d_Al,density=2.7,material='Al'))
        optBL.add(*optBL.Aperture(a=apertx/2,b=aperty/2))
        optBL.add(*optBL.Screen(dist=D,propagator=propa_sc,
                                rs=rs_sc,ra=ra_sc,re=re_sc))

        self.propagateWfr(optBL)


class ToroidalMirrorLine(radsrc.SynchrotronRadiation):

    def __init__(self,energy,d,D,x,y,
                 B,L,relPrec,
                 apertx,aperty,
                 error:typing.Literal["Zeiss","meas"],
                 propa_sc:_type_propas,ra_sc,re_sc,rs=0):
        
        # beam = list(get_beam('carcara').values())

        super().__init__(energy, d, x, y,
                         field = {'BM':[B,L]}, relPrec=relPrec)

        self.load_beam("carcara")

        optBL = radsrc.BeamLine()
        optBL.add(*optBL.Aperture(a=apertx/2,b=aperty/2))
        optBL.add(*optBL.ToroidalMirror(ang=18.78e-3,
                                        tang_len=0.22,R_tang=905.271,
                                        sag_len=0.008,R_sag=0.3192))
        
        if error:
            errorname = {'Zeiss':'CAX_M1_Zeiss_height_error_sh.dat',
                         'meas': 'CAX_height_error_FZI_220mm_sh.dat'}.get(error)
            errorpath = 'data_opt_elements/'+errorname
            errorfile = __file__.replace("beamlines.py",errorpath)
            unit = {'Zeiss':1,'meas':1e-3}.get(error)

            optBL.add(*optBL.ErrorMirror(errorfile,unit,18.78e-3,'x',L=0.22,W=0.008))
        # if D > 0:
        optBL.add(*optBL.Screen(dist=D,propagator=propa_sc,
                                rs=rs,ra=ra_sc,re=re_sc))

        self.propagateWfr(optBL)




#todo: metodos setters para redefinir prop params dos elementos
#todo: ajustar o propagate wfr para isso
class CarcaraPinhole(PinholeLine):

    #! _type_propas aparece no docstring
    def __init__(self,energy,d,D,apertx,aperty,
                 B=0.5642,L=2,d_Al=1e-3,
                 propa_sc:_type_propas='Standard',rs_sc=0,ra_sc=2,re_sc=8):

        x = [-50e-6,50e-6,120]
        y = [-50e-6,50e-6,120]

        super().__init__(energy,d,D,x,y,B,L,
                         d_Al,apertx,aperty,
                         propa_sc,rs_sc,ra_sc,re_sc)


class CarcaraMirror(ToroidalMirrorLine):

    def __init__(self,energy=11e3,d=17,D=17,
                 error:typing.Literal["Zeiss","meas"]='',
                 apertx=4e-3,aperty=4e-3,
                 propa_sc='Quadratic',ra_sc=3.0,re_sc=3.0):

        super().__init__(energy,d,D,
                         x=[-3e-3,3e-3,200],y=[-3e-3,3e-3,200],
                         B=0.5642,L=0.853,relPrec=0.01,
                         apertx=apertx,aperty=aperty,
                         error=error,
                         propa_sc=propa_sc,ra_sc=ra_sc,re_sc=re_sc)

