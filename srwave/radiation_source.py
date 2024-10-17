
import os
import sys
import json
import typing
from array import array
from copy import deepcopy

import numpy as np
import scipy.constants as cte

from . import utils as uti

from optlnls.mirror import transmission
from optlnls.surface import SRW_figure_error

import srwpy.srwlib as srw
import srwpy.srwlpy as srwl


# fundamental constants
_c = cte.c # speed of light [m/s]
_e = cte.e # fundamental charge [C]
_me = cte.electron_mass # electron rest mass [kg]
_E0 = _me*(_c**2)/_e # electron rest energy [eV]

# accelerator constants
_I = 0.1 # current [A]
_E = 3e9 # energy [eV]
_gamma = _E/_E0 # lorentz factor [adim]


# srw propagators dict
_propagators = {'Standard':0,
                'Quadratic':1, 'Quadratic Special':2,
                'From Waist':3, 'To Waist':4}
_type_propas = typing.Literal['Standard',
                              'Quadratic','Quadratic Special',
                              'From Waist','To Waist']

# types for calculations
_beam_type = typing.Union[srw.SRWLPartBeam,None]
_part_type = typing.Literal['re','im']
_pol_type = typing.Literal['linH','linV','lin45','lin135','circR','circL','total']
_intens_type = typing.Literal['SE','ME','SE Fluence']
# _coords = typing.Literal['e','x','y','xy','ex','ey','exy']


#todo: mudar E_ph para energy para fazer sentido aceitar um range de energias
#todo: testar se classes e propagacoes e coisas do srw funcionam sem array module, so' list python ou numpy array



class Beam(srw.SRWLPartBeam):

    def __init__(self,I=_I,E=_E,isTwiss=True,eqparams=None,*args,**kwargs):
        """
        Electron Beam.

        Args:
            isTwiss (bool): twiss parameters are given or rms are given.
            eqparams (list): beam equilibrium parameters:
                isTwiss=True:
                    [0]:  sigEperE; relative RMS energy spread
                    [1]:  emitx; horizontal emittance [m.rad]
                    [2]:  betax; horizontal beta [m]
                    [3]:  alphax; horizontal alpha [rad]
                    [4]:  etax; horizontal dispersion [rad]
                    [5]:  etapx; horizontal dispersion derivative [rad/m]
                    [6]:  emity; vertical emittance [m.rad]
                    [7]:  betay; vertical beta [m]
                    [8]:  alphay; vertical alpha [rad]
                    [9]:  etay; vertical dispersion [rad]
                    [10]: etapy; vertical dispersion derivative [rad/m]
                isTwiss=False:
                    [0]: sigEperE; relative RMS energy spread
                    [1]: sig_rx; horizontal RMS size of e-beam [m]
                    [2]: sig_px; horizontal RMS angular divergence [rad]
                    [3]: rxpx; <(rx-<rx>)(px-<px>)>; horizontal crossed second moment [m.rad]
                    [4]: sig_ry; vertical RMS size of e-beam [m]
                    [5]: sig_py; vertical RMS angular divergence [rad]
                    [6]: rypy; <(ry-<ry>)(py-<py>)>; vertical crossed second moment [m.rad]
        """
        
        super().__init__(*args,**kwargs)

        if eqparams is None:
            self.Iavg = I
            self.partStatMom1.gamma = _gamma
        elif isTwiss:
            self.from_Twiss(I,E,*eqparams)
        else:
            self.from_RMS(I,E,*eqparams)

    def __str__(self):
        print('Beam')
        print('Iavg =',f'{self.Iavg*1e3} mA')
        print('E =',f'{self.partStatMom1.gamma*_E0*1e-9} GeV')
        print(r'$\delta =$',f'{self.arStatMom2[10]}')
        print('')

    def __repr__(self):
        pass


#todo: tirar elementos da beamline
# class OptElements:


class BeamLine(srw.SRWLOptC):

    propagators = _propagators

    #todo: definicao alternativa que insere os opt elements direto no init
    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)



    def Drift(self,dist,propagator:_type_propas,rs,ra,re):
        p = self.propagators[propagator]

        opt_element = srw.SRWLOptD(dist)

        prop_params = [rs,   #"Auto Resize Before Propagation" 
                       0,    #"Auto Resize After Propagation" 
                       1.0,  #"Relative precision for propagation with autoresizing"
                       p, #"Propagator" (check propagators dict)
                       0,    #"Do any resizing..."? (0: False, 1: True)
                       ra, #"H range modification factor at resizing"
                       re, #"H resolution modification factor at resizing"
                       ra, #"V range modification factor at resizing"
                       re, #"V resolution modification factor at resizing"
                       0,    # mysterios parameter 1
                       0.0,  # mysterios parameter 2
                       0.0]  # mysterios parameter 3

        return opt_element, prop_params

    Screen = Drift

    #todo: redefinir classe do filtro para aceitar nx=ny=1
    @staticmethod
    def Filter(energy,thickness,density,material):

        if not isinstance(energy, (list,np.ndarray)): energy = [energy]

        transmE = np.sqrt(transmission(energy, thickness, density, material))
        arr_transm = np.array(120*120*[[t,0] for t in transmE]).reshape(-1)
        # arr_transm =  array('d',120*120*[np.sqrt(transm),0])

        #todo: comment meaning of params
        opt_element = srw.SRWLOptT(
            _x = 0.0, _rx = 200e-6, _nx = 120, # x center and range
            _y = 0.0, _ry = 200e-6, _ny = 120, # y center and range
            _arTr = arr_transm, # transmission array
            _extTr = 1,
            _eStart=energy[0], _eFin=energy[-1], _ne=len(energy) # energy range
        )
        
        prop_params = [0,   #"Auto Resize Before Propagation" 
                       0,   #"Auto Resize After Propagation" 
                       1.0, #"Relative precision for propagation with autoresizing"
                       0,   #"Propagator" (check propagators dict)
                       0,   #"Do any resizing..."? (0: False, 1: True)
                       1.0, #"H range modification factor at resizing"
                       1.0, #"H resolution modification factor at resizing"
                       1.0, #"V range modification factor at resizing"
                       1.0, #"V resolution modification factor at resizing"
                       0,   # mysterios parameter 1
                       0.0, # mysterios parameter 2
                       0.0] # mysterios parameter 3
                            # optional parameters 1 to 5 (not included, all are 0)

        return opt_element, prop_params

    @staticmethod
    def Aperture(a,b,xc=0.0,yc=0.0,ra_a=1.0,re_a=1.0):

        opt_element = srw.SRWLOptA(_shape = 'r', #'r': rectangle
                                   _ap_or_ob = 'a', #'a': aperture
                                   _Dx = 2*a, #"Width [m]"
                                   _Dy = 2*b, #"Height [m]"
                                   _x = xc, # horizontal center [m]
                                   _y = yc) # vertical center [m]

        prop_params = [0,   #"Auto Resize Before Propagation" 
                       0,   #"Auto Resize After Propagation" 
                       1.0, #"Relative precision for propagation with autoresizing"
                       0,   #"Propagator" (check propagators dict)
                       0,   #"Do any resizing..."? (0: False, 1: True)
                       ra_a, #"H range modification factor at resizing"
                       re_a, #"H resolution modification factor at resizing"
                       ra_a, #"V range modification factor at resizing"
                       re_a, #"V resolution modification factor at resizing"
                       0,   # mysterios parameter 1
                       0.0, # mysterios parameter 2
                       0.0] # mysterios parameter 3
                            # optional parameters 1 to 5 (not included, all are 0)

        # carcara mirror: 'Standard', ra=8, re=2

        return opt_element, prop_params
    
    @staticmethod
    def PlaneMirror(ang,tang_len,sag_len):

        opt_element = srw.SRWLOptMirPl()
        opt_element.set_dim_sim_meth(
            _size_tang = tang_len, #"Tangential Size [m]"
            _size_sag = sag_len, #"Sagittal Size [m]"
            _ap_shape = 'r', # shape of aperture ('r': rectangular)
            _sim_meth = 2, # simulation method (2: "thick" approximation)
            _treat_in_out = 1 # 1: input and output wfr at center of mirror
        )
        opt_element.set_orient(
            _nvx = -np.sqrt(1-ang**2), # horizontal coordinate of central normal vector
            _nvy = 0, # vertical coordinate of central normal vector
            _nvz = -ang, # longitudinal coordinate of central normal vector
            _tvx = ang, # horizontal coordinate of central tangential vector
            _tvy = 0, # vertical coordinate of central tangential vector
            _x = 0, # horizontal position of mirror center [m]
            _y = 0 # vertical position of mirror center [m]
        )

        prop_params = [0,   #"Auto Resize Before Propagation" 
                       0,   #"Auto Resize After Propagation" 
                       1.0, #"Relative precision for propagation with autoresizing"
                       0,   #"Propagator" (check propagators dict)
                       0,   #"Do any resizing..."? (0: False, 1: True)
                       1.0, #"H range modification factor at resizing"
                       1.0, #"H resolution modification factor at resizing"
                       1.0, #"V range modification factor at resizing"
                       1.0, #"V resolution modification factor at resizing"
                       0,   # mysterios parameter 1
                       0.0, # mysterios parameter 2
                       0.0] # mysterios parameter 3
                            # optional parameters 1 to 5 (not included, all are 0)

        return opt_element, prop_params


    @staticmethod
    def ToroidalMirror(ang,tang_len,R_tang,sag_len,R_sag):

        opt_element = srw.SRWLOptMirTor(_rt = R_tang, #"Tangential Radius [m]"
                                        _rs = R_sag) #"Sagittal Radius [m]"
        opt_element.set_dim_sim_meth(
            _size_tang = tang_len, #"Tangential Size [m]"
            _size_sag = sag_len, #"Sagittal Size [m]"
            _ap_shape = 'r', # shape of aperture ('r': rectangular)
            _sim_meth = 2, # simulation method (2: "thick" approximation)
            _treat_in_out = 1 # 1: input and output wfr at center of mirror
        )
        opt_element.set_orient(
            _nvx = -np.sqrt(1-ang**2), # horizontal coordinate of central normal vector
            _nvy = 0, # vertical coordinate of central normal vector
            _nvz = -ang, # longitudinal coordinate of central normal vector
            _tvx = ang, # horizontal coordinate of central tangential vector
            _tvy = 0, # vertical coordinate of central tangential vector
            _x = 0, # horizontal position of mirror center [m]
            _y = 0 # vertical position of mirror center [m]
        )

        prop_params = [0,   #"Auto Resize Before Propagation" 
                       0,   #"Auto Resize After Propagation" 
                       1.0, #"Relative precision for propagation with autoresizing"
                       0,   #"Propagator" (check propagators dict)
                       0,   #"Do any resizing..."? (0: False, 1: True)
                       1.0, #"H range modification factor at resizing"
                       1.0, #"H resolution modification factor at resizing"
                       1.0, #"V range modification factor at resizing"
                       1.0, #"V resolution modification factor at resizing"
                       0,   # mysterios parameter 1
                       0.0, # mysterios parameter 2
                       0.0] # mysterios parameter 3
                            # optional parameters 1 to 5 (not included, all are 0)

        return opt_element, prop_params

    @staticmethod
    def ErrorMirror(filename,unit,ang,orientation,L,W):

        opt_element = SRW_figure_error(filename,unit,ang,ang,orientation,L=L,W=W)

        prop_params = [0,   #"Auto Resize Before Propagation" 
                       0,   #"Auto Resize After Propagation" 
                       1.0, #"Relative precision for propagation with autoresizing"
                       0,   #"Propagator" (check propagators dict)
                       0,   #"Do any resizing..."? (0: False, 1: True)
                       1.0, #"H range modification factor at resizing"
                       1.0, #"H resolution modification factor at resizing"
                       1.0, #"V range modification factor at resizing"
                       1.0, #"V resolution modification factor at resizing"
                       0,   # mysterios parameter 1
                       0.0, # mysterios parameter 2
                       0.0] # mysterios parameter 3
                            # optional parameters 1 to 5 (not included, all are 0)

        return opt_element, prop_params

    def add(self,opt_element,prop_params):
        self.arOpt.append(opt_element)
        self.arProp.append(prop_params)



class SynchrotronRadiation(srw.SRWLWfr):

    # tem wavefront inicial (ja calculou electric field) e so' quer propagar de outros jeitos, por exemplo
    # @typing.overload
    # def __init__(self, wfr: srw.SRWLWfr):
    #     ...
    
    # calcular a wfr inicial, mas ja tem a trajetoria da particula, entao nao precisa do campo magnetico
    # @typing.overload
    # def __init__(self,energy,d,beam):
    #     ...

    # calcular wfr inicial a partir do beam e do campo
    #* nao aceita field=None, quando seria passado so' trajetoria
    def __init__(self, energy, d, x, y,
                 field:dict,
                 beam:_beam_type=None,**fieldKwargs):
        """
        Basic Synchrotron Radiation class. Calculates radiation wavefront from
        source.

        Args:
            energy (float or list): photon energy(ies) [eV]; format: [ei,ef,ni]
            d (float): distance from source [m]
            x (float or list): horizontal position(s) [m]; format: [xi,xf,nx]
            y (float or list): vertical position(s) [m]; format: [yi,yf,ny]
            field (dict):
                - format: {fieldType: field}
                - Bending source: fieldType = 'BM', field = [B,L]
                - Undulator source: fieldType = 'Und', field = [period_length, nr_periods, B] 
            beam:
                - from RMS: beam = [sigX,sigXp,XXP,sigY,sigYp,YYP]
                - from twiss: beam = [emitX,betaX,alphaX,etaX,etaXp,
                                      emitY,betaY,alphaY,etaY,etaYp]
        """

        super().__init__()


        self.partBeam = Beam() if beam is None else beam
        

        fieldType = list(field.keys())[0]

        partTraj, magFldCnt, precisions = self.setTrajectory(
            element=fieldType, field=field[fieldType], **fieldKwargs
        )
        
        
        if isinstance(x,(int,float)): x = [x,x,1]
        elif isinstance(x,np.ndarray): x = [x[0],x[-1],len(x)]
        if isinstance(y,(int,float)): y = [y,y,1]
        elif isinstance(y,np.ndarray): y = [y[0],y[-1],len(y)]
        if isinstance(energy,(int,float)): energy = [energy,energy,1]
        elif isinstance(energy,np.ndarray): energy = [energy[0],energy[-1],len(energy)]

        self.setWfr(x,y,energy,d)


        srwl.CalcElecFieldSR(self, partTraj, magFldCnt, precisions)
    
    # def calc_wfr(wfr,partTraj):
    #     srwl.CalcElecFieldSR(wfr,partTraj)

    def __str__(self):
        #todo
        pass

    def __repr__(self):
        #todo
        pass


    def setBeam(self, isTwiss: bool, eqparams: list):
        self.partBeam = Beam(isTwiss,eqparams)

    def load_beam(self, beam="carcara", isTwiss=True):
        mode = 'twiss' if isTwiss else 'rms'
        beams_file = __file__.replace("radiation_source.py","beams.json")
        with open(beams_file) as b:
            beams = json.load(b)
        eqparams = list(beams[beam][mode].values())
        self.partBeam = Beam(isTwiss=isTwiss,eqparams=eqparams)



    def setBendingMagnet(self,B,L):
        """B [T], L [m]"""
        
        nr_poles = 1
        BM = srw.SRWLMagFldM(B, nr_poles, 'n', L)

        return BM
    
    @staticmethod
    def setHarmonicField(B,plane='v',phase0=0,symmetry=1,transverse_coeff=1):
        """harmonic magnetic field.
        Args:
            plane: magnetic field plane: horzontal ('h') or vertical ('v').
            symmetry: longitudinal symmetry: symmetric ('') or anti-symmetric ('anti').
        """
        n_harm = 1 #harmonic number ; todos os exemplos usam isso #?: o que e'?
        idx_symm = {'':1,'anti':-1}.get(symmetry)
        return srw.SRWLMagFldH(n_harm,plane,B,phase0,idx_symm,transverse_coeff)

    def setUndulator(self,period_length,nr_periods,B,Bsettings=['v',0,'',1]):
        """
        Args:
            period_length [m].
            nr_periods: number of periods.
            B (float or list of floats): magnetic field amplitude [T].
            Bsettings: list of params to configurate harmonic magnetic fields:
                [0]: plane ('h' or 'v')
                [1]: initial phase
                [2]: symmetry ('' or 'anti')
                [3]: coefficient of field transverse dependence
        """

        if isinstance(B,(int,float)): B, Bsettings = [B], [Bsettings]
        arrH = []
        for b, settings in zip(B,Bsettings):
            harm = self.setHarmonicField(b,*settings)
            arrH.append(harm)
        
        und = srw.SRWLMagFldU(arrH,period_length,nr_periods)

        return und

    #todo: overload de List[MagFld]
    @typing.overload
    def setTrajectory(self, element: str, field: srw.SRWLMagFld, relPrec=0.005): ...
    @typing.overload
    def setTrajectory(self, element: str, field: list, relPrec=0.005): ...
    @typing.overload
    def setTrajectory(self, element: srw.SRWLPrtTrj): ...
    def setTrajectory(self, element, field=None, relPrec=0.005):
        #!: nao aceita return caso seja dado um SRWLPrtTrj
        #*: nao aceita lista de fieldType

        if isinstance(element,srw.SRWLPrtTrj):
            partTraj = element
            return partTraj
        
        else:
            partTraj = 0 # traj arrays not defined, calculate them using _inMagFldC

            if isinstance(field,srw.SRWLMagFld):
                arrB = [field]
            elif isinstance(field,list):
                if element == 'BM':
                    B, L = field
                    arrB = [self.setBendingMagnet(B,L)]
                #todo: elliptical undulator
                elif element == 'Und':
                    period_length, nr_periods, B = field
                    arrB = [self.setUndulator(period_length, nr_periods, B)]
            else:
                raise TypeError("Field type not alowed.")
            
            # center of magnet: origin
            Bcenters = [array('d', [0.0]), array('d', [0.0]), array('d', [0.0])]
            #Container of magnetic field elements and their positions in 3D:
            magFldCnt = srw.SRWLMagFldC(_arMagFld=arrB, 
                                        _arXc=Bcenters[0], 
                                        _arYc=Bcenters[1], 
                                        _arZc=Bcenters[2])
            
            method = {'Und':1, 'BM':2}.get(element)
            precisions = [method,
                          relPrec, #method=2 => relative precision
                          0,     #longitudinal position [m] to start integration
                          0,     #longitudinal position [m] to finish integration
                          50000, #number of points to use for trajectory calculation 
                          1,     #do calculate terminating terms
                          0.0]   #sampling factor

            return partTraj, magFldCnt, precisions



    #todo: overload: cls.setWfr(arrReE,arrImE)
    def setWfr(self,xrange,yrange,erange,d,unit=1):
        xi, xf, nx = xrange
        yi, yf, ny = yrange
        ei, ef, ne = erange

        #Radiation Sampling for the Initial Wavefront (before optical elements)

        #Numbers of points of photon energy, horizontal and vertical positions
        self.allocate(ne,nx,ny) 
        #Mesh
        self.mesh.eStart = ei #initial energy
        self.mesh.eFin   = ef #final energy
        self.mesh.xStart = xi #initial horizontal position [m]
        self.mesh.xFin   = xf #final horizontal position [m]
        self.mesh.yStart = yi #initial vertical position [m]
        self.mesh.yFin   = yf #final vertical position [m]
        self.mesh.zStart = d #Longitudinal position for initial wavefront [m]
        #Electric field unit:
        # 0: arbitrary, 1: sqrt(Phot/s/0.1%bw/mm^2)
        # 2: sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on represent (freq. or time)
        self.unitElFld = unit 



    #todo: inserir calc Electric field em algum lugar
    #?: recalcular depois de alguma redefinicao?

    def propagateWfr(self,optBL): srwl.PropagElecField(self, optBL)


    #todo: aceitar lista de coords para calcular multiplas intensidades de uma vez
    def calc_intensity(self, coords: str, energy: float, X: float, Y: float,
                       polarization:_pol_type='total',intType:_intens_type='SE'):

        idx_pol = {'linH':0,'linV':1,'lin45':2,'lin135':3,
                   'circR':4,'circL':5,
                   'total':6}.get(polarization)
        idx_type = {'SE':0,'ME':1,'SE Fluence':7}.get(intType)
        idx_coord = {'e':0,'x':1,'y':2,'xy':3,'ex':4,'ey':5,'exy':6}.get(coords)

        if None not in [idx_pol,idx_type,idx_coord]:

            N = np.prod([getattr(self.mesh, f'n{coord}') for coord in coords])
            intervals = [[getattr(self.mesh,f'{coord}Start'),
                          getattr(self.mesh,f'{coord}Fin'),
                          getattr(self.mesh,f'n{coord}')] for coord in coords]

            arrI = array('f', N*[0])
            srwl.CalcIntFromElecField(arrI, self,
                                        idx_pol, idx_type, idx_coord,
                                        energy, X, Y)

            return arrI, intervals

        else:
            raise ValueError('Invalid arguments! Check their writing.')

    def calc_electric_field(self,part:_part_type,coords,energy,X,Y,polarization:_pol_type='total'):

        idx_pol = {'linH':0,'linV':1,'lin45':2,'lin135':3,
                   'circR':4,'circL':5,
                   'total':6}.get(polarization)
        idx_part = {'re':5,'im':6}.get(part)
        idx_coord = {'e':0,'x':1,'y':2,'xy':3,'ex':4,'ey':5,'exy':6}.get(coords)

        if None not in [idx_pol,idx_part,idx_coord]:

            N = np.prod([getattr(self.mesh, f'n{coord}') for coord in coords])
            intervals = [[getattr(self.mesh,f'{coord}Start'),
                          getattr(self.mesh,f'{coord}Fin'),
                          getattr(self.mesh,f'n{coord}')] for coord in coords]
            
            arrI = array('f', N*[0])
            srwl.CalcIntFromElecField(arrI, self,
                                        idx_pol, idx_part, idx_coord,
                                        energy, X, Y)

            return arrI, intervals

        else:
            print('Invalid arguments! Check their writing.')

    #!: ainda com problemas de acertar o novo range pedido
    def resize_wfr_srw(self,xrangenew,yrangenew,erangenew=[],method='regular',rsType='pos/ang'):
        """
        Resize wavefront with respect to horizontal and vertical ranges or energy range.

        Args:
            xnew (list): new x range values; [xStart, xFin, nx]
            ynew (list): new y range values; [yStart, yFin, ny]
        """

        idx_meth = {'regular':0, 'special':1}.get(method) # without or with FFT
        idx_type = {'pos/ang':'c', 'e/t':'f'}.get(rsType)

        if idx_type=='c':

            xi_old, xf_old, nx_old = self.mesh.xStart, self.mesh.xFin, self.mesh.nx
            yi_old, yf_old, ny_old = self.mesh.yStart, self.mesh.yFin, self.mesh.ny
            xi_new, xf_new, nx_new = xrangenew
            yi_new, yf_new, ny_new = yrangenew

            xamp_old = xf_old - xi_old
            xamp_new = xf_new - xi_new
            f_ra_x = xamp_new/xamp_old
            # xstep_old = xamp_old/(nx_old-1)
            # xstep_new = xamp_new/(nx_new-1)
            # f_re_x = xstep_old/xstep_new # srw way
            # f_re_x = nx_new/nx_old       # my way
            f_re_x = 1.0                   # oasys way
            xc_new = (xi_new + xf_new)/2
            f_xc = (xc_new - xi_old)/xamp_old
            #todo: ajustar xc para sempre deslocar para depois de xi_new, o mais
            #todo: proximo e sempre antes de xf_new, o mais proximo

            yamp_old = yf_old - yi_old
            yamp_new = yf_new - yi_new
            f_ra_y = yamp_new/yamp_old
            # ystep_old = yamp_old/(ny_old-1)
            # ystep_new = yamp_new/(ny_new-1)
            # f_re_y = ystep_old/ystep_new # srw way
            # f_re_y = ny_new/ny_old       # my way
            f_re_y = 1.0                   # oasys way
            yc_new = (yi_new + yf_new)/2
            f_yc = (yc_new - yi_old)/yamp_old

            params = [idx_meth,f_ra_x,f_re_x,f_ra_y,f_re_y,f_xc,f_yc]
            # print(params)

        elif idx_type=='f':

            ei_old, ef_old, ne_old = self.mesh.eStart, self.mesh.eFin, self.mesh.ne
            ei_new, ef_new, ne_new = erangenew

            eamp_old = ef_old - ei_old
            eamp_new = ef_new - ei_new
            f_ra = eamp_new/eamp_old
            # estep_old = eamp_old/(ne_old-1)
            # estep_new = eamp_new/(ne_new-1)
            # f_re = estep_old/estep_new # srw way
            # f_re = ne_new/ne_old       # my way
            f_re = 1.0                   # oasys way
            ec_new = (ei_new + ef_new)/2
            f_c = (ec_new - ei_old)/eamp_old

            params = [idx_meth,f_ra,f_re,f_c]
            # print(params)

        else:
            return False

        srwl.ResizeElecField(self,idx_type,params)

        return True

    def resize_wfr(self,newxlim,newylim):
        nx_old, ny_old = self.mesh.nx, self.mesh.ny
        x_start, x_fin = newxlim
        y_start, y_fin = newylim
        nx_new = int((x_fin - x_start)/(self.mesh.xFin - self.mesh.xStart)*(nx_old-1) + 1)
        ny_new = int((y_fin - y_start)/(self.mesh.yFin - self.mesh.yStart)*(ny_old-1) + 1)
        idx_x = slice(int((x_start - self.mesh.xStart)/(self.mesh.xFin - self.mesh.xStart)*(nx_old-1)),int((x_fin - self.mesh.xStart)/(self.mesh.xFin - self.mesh.xStart)*(nx_old-1) + 1))
        idx_y = slice(int((y_start - self.mesh.yStart)/(self.mesh.yFin - self.mesh.yStart)*(ny_old-1)),int((y_fin - self.mesh.yStart)/(self.mesh.yFin - self.mesh.yStart)*(ny_old-1) + 1))
        self.arEx = self.arEx[idx_x,idx_y]
        self.arEy = self.arEy[idx_x,idx_y]
        self.mesh.nx = nx_new
        self.mesh.ny = ny_new
        self.mesh.xStart = x_start
        self.mesh.xFin = x_fin
        self.mesh.yStart = y_start
        self.mesh.yFin = y_fin
    
    def slice_wfr(self,xlim,ylim):

        print('initial self.Ex[:10] array:',self.arEx[:10])
        print('initial self.xi:',self.mesh.xStart,'initial self.xf:',self.mesh.xFin)
        print('initial self.nx:',self.mesh.nx)
        print('initial self.yi:',self.mesh.yStart,'initial self.yf:',self.mesh.yFin)
        print('initial self.ny:',self.mesh.ny)

        xi,xf,nx = self.mesh.xStart, self.mesh.xFin, self.mesh.nx
        yi,yf,ny = self.mesh.yStart, self.mesh.yFin, self.mesh.ny
        ne = self.mesh.ne

        arEx = np.array(self.arEx).reshape(ny,nx,ne,2)

        x0, x1 = xlim
        y0, y1 = ylim

        x = np.linspace(xi,xf,nx)
        y = np.linspace(yi,yf,ny)
        maskx = (x0 <= x) & (x <= x1)
        masky = (y0 <= y) & (y <= y1)

        newxi, newxf, newnx = x[maskx][0], x[maskx][-1], sum(maskx)
        newyi, newyf, newny = y[masky][0], y[masky][-1], sum(masky)

        self.allocate(ne,newnx,newny)

        sliced_arEx = arEx[masky,maskx,:,:].reshape(-1)

        self.arEx = array('f',sliced_arEx)
        

        self.mesh.xStart, self.mesh.xFin, self.mesh.nx = newxi, newxf, newnx
        self.mesh.yStart, self.mesh.yFin, self.mesh.ny = newyi, newyf, newny

        


        print('final self.Ex[:10] array:',self.arEx[:10])
        print('final self.xi:',self.mesh.xStart,'final self.xf:',self.mesh.xFin)
        print('final self.nx:',self.mesh.nx)
        print('final self.yi:',self.mesh.yStart,'final self.yf:',self.mesh.yFin)
        print('final self.ny:',self.mesh.ny)

        
    # def resize_wfr(self,)
    
    
    #todo: override calc_stokes de SRWLWfr

    @classmethod
    def energy_loop_intensity(cls,energy,coords:str,x,y,xnew=None,*clsargs,**clskwargs):
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



