
import json
import typing
import copy

import numpy as np


from . import utils as uti
from . import opt_elements as oe
from .beamtemp import Beam

import srwpy.srwlib as srw
import srwpy.srwlpy as srwl




class Beamline:

    def __init__(self,line=None):

        self.wfr = None 
        
        self.opt_elements = line


    @property
    def opt_elements(self):
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
    def srw_opts(self):
        """List of SRW optical elements in the beamline"""
        return [element.srw_opt for element in self.opt_elements]

    @property
    def props_params(self):
        """List of propagation parameters of the beamline optical elements"""
        return [list(element.prop_params) for element in self.opt_elements]

    def add_opt_element(self,element):
        if isinstance(element,oe.OpticalElement):
            self._opt_elements.append(element)
        else:
            raise TypeError(f"Optical Element type '{type(element).__name__}' not supported")



class SynchrotronRadiation(srw.SRWLWfr):

    # calcular wfr inicial a partir do beam e do campo
    #* nao aceita field=None, quando seria passado so' trajetoria
    def __init__(self, energy, d, x, y,
                 field,
                 beam=None,store_steps=False,**fieldKwargs):
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
        
        if np.isscalar(x): x = [x]
        if np.isscalar(y): y = [y]
        if np.isscalar(energy): energy = [energy]

        self.ExCnt = []
        self.EyCnt = []

        self.setWfr(x,y,energy,d)

        self.calcWfr(partTraj, magFldCnt, precisions, store_steps)

    def __str__(self):
        pass

    def __repr__(self):
        pass


    def setBeam(self,eqparams):
        self.beam = Beam(eqparams)

    def load_beam(self, beam="carcara", isTwiss=True):
        mode = 'twiss' if isTwiss else 'rms'
        beams_file = __file__.replace("radiation_source.py","beams.json")
        with open(beams_file) as b:
            beams = json.load(b)
        eqparams = list(beams[beam][mode].values())
        self.partBeam = Beam(eqparams=eqparams)


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

    def setTrajectory(self, element, field=None, relPrec=0.005):

        if isinstance(element,srw.SRWLPrtTrj):
            partTraj = element
            return partTraj
        
        else: # traj arrays not defined, calculate them using _inMagFldC
            partTraj = 0 

            if isinstance(field,srw.SRWLMagFld):
                arrB = [field]
            elif isinstance(field,list):
                if element == 'BM':
                    B, L = field
                    arrB = [self.setBendingMagnet(B,L)]
                elif element == 'Und':
                    period_length, nr_periods, B = field
                    arrB = [self.setUndulator(period_length, nr_periods, B)]
            else:
                raise TypeError("Field type not alowed.")
            
            # Center of magnet: origin
            # Bcenters = [array('d', [0.0]), array('d', [0.0]), array('d', [0.0])]
            # Container of magnetic field elements and their positions in 3D:
            magFldCnt = srw.SRWLMagFldC(_arMagFld=arrB, 
                                        _arXc=np.array([0.0],dtype=np.float64), 
                                        _arYc=np.array([0.0],dtype=np.float64), 
                                        _arZc=np.array([0.0],dtype=np.float64))
            
            method = {'Und':1, 'BM':2}.get(element)
            precisions = [method,
                          relPrec, #method=2 => relative precision
                          0,     #longitudinal position [m] to start integration
                          0,     #longitudinal position [m] to finish integration
                          50000, #number of points to use for trajectory calculation 
                          1,     #do calculate terminating terms
                          0.0]   #sampling factor

            return partTraj, magFldCnt, precisions


    def setWfr(self,x,y,e,d,unit=1):
        """
        Initializes the wavefront mesh.

        Args:
            x (array): Horizontal positions [m].
            y (array): Vertical positions [m].
            e (array): Photon energies [eV].
            d (float): Longitudinal position for initial wavefront [m].
            unit (int, optional): Electric field unit. Defaults to 1.
                - 0: arbitrary
                - 1: sqrt(Phot/s/0.1%bw/mm^2)
                - 2: sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on frequency or time domain.
        """

        xi, xf, nx = x[0], x[-1], len(x)
        yi, yf, ny = y[0], y[-1], len(y)
        ei, ef, ne = e[0], e[-1], len(e)

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
        self.unitElFld = unit 

    def setWfrElecField(self, arrEx, arrEy):
        self.arEx = copy.copy(arrEx)
        self.arEy = copy.copy(arrEy)

    def calcWfr(self, partTraj, magFldCnt, precisions, store_steps=True):

        srwl.CalcElecFieldSR(self, partTraj, magFldCnt, precisions)

        if store_steps:
            self.Ex = [copy.copy(self.arEx)]
            self.Ey = [copy.copy(self.arEy)]

    def propagateWfr(self, beamline: Beamline, store_steps=False):

        if not beamline.opt_elements:
            return False
        
        optBl = srw.SRWLOptC()

        oe_arr, pp_arr = beamline.srw_opts, beamline.props_params

        if not store_steps:
            optBl.arOpt, optBl.arProp = oe_arr, pp_arr
            srwl.PropagElecField(self, optBl)

        else:
            for srwopt, prop_params in zip(oe_arr, pp_arr):

                optBl.arOpt, optBl.arProp = [srwopt], [prop_params]
                srwl.PropagElecField(self, optBl)
                self.Ex.append(self.arEx)
                self.Ey.append(self.arEy)

        return True


    def IntFromElecField(self, pol, intType, coords, energy, X, Y):

        idx_pol = {'linH':0,'linV':1,'lin45':2,'lin135':3,
                   'circR':4,'circL':5,
                   'total':6}.get(pol)
        idx_type = {'SE I':0,'ME I':1,'SE F':2,'ME F':3,
                    'SE P':4,'SE ReE':5,'SE ImE':6,
                    'SE Fluence':7,'SE J':8}.get(intType)
        idx_coord = {'e':0,'x':1,'y':2,'xy':3,'ex':4,'ey':5,'exy':6}.get(coords)

        if None not in [idx_pol,idx_type,idx_coord]:

            N = np.prod([getattr(self.mesh, f'n{coord}') for coord in coords])
            intervals = [[getattr(self.mesh,f'{coord}Start'),
                        getattr(self.mesh,f'{coord}Fin'),
                        getattr(self.mesh,f'n{coord}')] for coord in coords]

            isPhase = intType=='SE P'
            arrI = np.zeros(N, dtype=np.float64 if isPhase else np.float32)

            srwl.CalcIntFromElecField(arrI, self,
                                      idx_pol, idx_type, idx_coord,
                                      energy, X, Y)

            return arrI, intervals

        else:
            raise ValueError('Invalid arguments! Check their writing.')

    #todo: nao funciona 'SE J'
    def calc_intensity(self, coords, energy, X, Y,
                       polarization='total',intType='SE'):

        if intType in ['SE','ME']:
            intType += ' I'
        else:
            q, inten = intType.split()
            if (q=='SE') and (inten not in ['Fluence','J']):
                raise ValueError('Invalid intensity type!')
            
        coords_lst = coords.split('-')

        if len(coords_lst) == 1:
            return self.IntFromElecField(polarization, intType, coords, energy, X, Y)
        else:
            return [self.IntFromElecField(polarization, intType, coord, energy, X, Y)
                        for coord in coords_lst]

    #todo: nao funciona xy
    def calc_flux(self, coords, energy, X, Y,
                  polarization='total',intType='SE'):
        
        if intType in ['SE','ME']:
            intType += ' F'
        else:
            raise ValueError('Invalid flux type!')

        coords_lst = coords.split('-')
        
        if len(coords_lst) == 1:
            return self.IntFromElecField(polarization, intType, coords, energy, X, Y)
        else:
            return [self.IntFromElecField(polarization, intType, coord, energy, X, Y)
                        for coord in coords_lst]

    def calc_electric_field(self, part, coords, energy, X, Y, polarization='total'):
        
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

        intType = {'phase':'SE P','re':'SE ReE','im':'SE ImE'}[part]
        coords_lst = coords.split('-')

        if len(coords_lst) == 1:
            return self.IntFromElecField(polarization, intType, coords, energy, X, Y)
        else:
            return [self.IntFromElecField(polarization, intType, coord, energy, X, Y)
                        for coord in coords_lst]

    @staticmethod
    def unwrap_phase(wrapped_phase: np.ndarray) -> np.ndarray:

        if wrapped_phase is None:
            return False
        
        if wrapped_phase.ndim == 1:
            unwrapped_phase =  np.unwrap(wrapped_phase)
        elif wrapped_phase.ndim == 2:
            unwrapped_phase =  np.unwrap(np.unwrap(wrapped_phase, axis=0), axis=1)

        return unwrapped_phase
    
    @staticmethod
    def wrap_phase(unwrapped_phase: np.ndarray) -> np.ndarray:

        if unwrapped_phase is None:
            return False
        
        wrapped_phase = ((unwrapped_phase-np.pi) % (2*np.pi)) - np.pi

        return wrapped_phase
        

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



