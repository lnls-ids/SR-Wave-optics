

import scipy.constants as cte

import srwpy.srwlib as srw


# fundamental constants
_c = cte.c # speed of light [m/s]
_e = cte.e # fundamental charge [C]
_me = cte.electron_mass # electron rest mass [kg]
_E0 = _me*(_c**2)/_e # electron rest energy [eV]


# accelerator constants
_I = 100e-3 # current [A]
_E = 3e9 # energy [eV]
_gamma = _E/_E0 # lorentz factor [adim]


class Beam(srw.SRWLPartBeam):

    def __init__(self,I=_I,E=_E,eqparams=None,*args,**kwargs):
        """
        Electron Beam.

        Args:
            I (float): current [A]
            E (float): energy [eV]
            isTwiss (bool): twiss parameters are given or rms are given.
            eqparams (list): beam equilibrium parameters:
                if twiss parameters:
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
                if rms parameters:
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
            self.partStatMom1.gamma = E/_E0
        elif len(eqparams) == 11:
            self.from_Twiss(I,E,*eqparams)
        elif len(eqparams) == 7:
            self.from_RMS(I,E,*eqparams)
        else:
            raise ValueError('eqparams must have 7 or 11 elements')
    
    @property
    def current(self):
        """Beam Current [A]"""
        return self.Iavg

    @current.setter
    def current(self,I):
        self.Iavg = I

    @property
    def energy(self):
        """Particle Energy [eV]"""
        return self.partStatMom1.gamma*_E0
    
    @energy.setter
    def energy(self,E):
        self.partStatMom1.gamma = E/_E0

    

    #todo: transformar params em properties com seus getters e setters

    #todo: transformar array de momentos em algo acessavel por strings
    # apos mudar algum valor(es), o seguinte metodo sera' chamada
    # apenas para atualizar da maneira normal o arStatMom2
    
    # def _update_arrMom2(self,string,value):
    #     idx = {...}[string]
    #     self.arStatMom2[idx] = value

    def __str__(self):
        print('Beam')
        print('Current [mA] =',f'{self.Iavg*1e3} mA')
        print('Beam energy [] =',f'{self.partStatMom1.gamma*_E0*1e-9} GeV')
        print('Energy spread',f'{self.arStatMom2[10]}')
        print('')

    # def __repr__(self):
    #     pass
    
    # size = moment() # teria que ser decorator pra passar logo o getter, que vai
    # receber direcao e ainda tirar sqrt
    # isso faz sentido? porque nos descriptors ja implementa-se getters e setters

    # length = moment('s')
    # energy_spread = moment('e')

    # size = property(fget=get_2nd_moment)

    def get_2nd_moment(self,mom2):
        idx = {'xx':0,'xxp':1,'xpxp':2,'yy':3,'yyp':4,'ypyp':5,'xy':6,
               'xpy':7,'xyp':8,'xpyp':9,'ee':10,'ss':11,'se':12,'xe':13,
               'xpe':14,'ye':15,'ype':16,'xs':17,'xps':18,'ys':19,'yps':20}[mom2]
        return self.arStatMom2[idx]


    def get_size(self,size):
        """Transversal beam size [m]"""
        idx = {'x':0,'y':3}[size]
        return self.arStatMom2[idx]
    
    def get_divergence(self,divergence):
        """Tranversal beam divergence [rad]"""
        idx = {'xp':2,'yp':5}[divergence]
        return self.arStatMom2[idx]

    def get_length(self):
        """Longitudinal bunch length [m]"""
        return self.arStatMom2[11]

    def get_energy_spread(self):
        """Relative energy spread [adimensional]"""
        return self.arStatMom2[10]

    # mixmom2
    def get_mixed_moments(self,moment2): 
        """Mixed moments of x, xp, y, yp, e and s"""
        idx = {'xxp':1,'yyp':4,'xy':6,'xpy':7,'xyp':8,
               'xpyp':9,'se':12,'xe':13,'xpe':14,'ye':15,
               'ype':16,'xs':17,'xps':18,'ys':19,'yps':20}[moment2]
        return self.arStatMom2[idx]

    # def get_twiss(self,twiss,direction,dispersion=False):
    #     """Beam Twiss parameters: alpha, beta, gamma and emittance"""

    #     d = {'x':0,'y':3}.get(direction)
        
    #     # axis 2nd moments; can be xx, xxp, xpxp or yy, yyp, ypyp
    #     aa, aap, apap = self.arStatMom2[0+d], self.arStatMom2[1+d], self.arStatMom2[2+d]

    #     # without dispersion:

    #     emittance = np.sqrt(aa*apap-aap**2)
    #     alpha = -apap/emittance
    #     beta = aa/emittance
    #     gamma = (1+alpha**2)/beta

    # def get_dispersion(self,dispersion):
    #     """Beam dispersion: eta and etap"""


    # def set_mixed