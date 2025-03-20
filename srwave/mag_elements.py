


from typing import Union, Optional, Literal

import numpy as np
import scipy.constants as cte

import srwpy.srwlib as srw
import srwpy.srwlpy as srwl





# MagnetCnt
_Method = Literal['manual','undulator','bending']
_CT = Union[list[float],tuple[float],Literal['length'],None]




class Magnet:

    @property
    def length(self):
        """Magnetic field length"""
        raise NotImplementedError
    
    @staticmethod
    def calc_gamma(E:float) -> float:
        """
        Calculates the gamma factor for the accelerator.
        
        Parameters
        ----------
        E : float
            Accelerator energy [eV]
        """
        E0 = cte.m_e*cte.c**2/cte.e # eV
        gamma = E/E0
        return gamma
    
    @staticmethod
    def calc_beta(E:float) -> float:
        """
        Calculates the velocity parameter or beta factor for the accelerator.
        
        Parameters
        ----------
        E : float
            Accelerator energy [eV]
        """
        E0 = cte.m_e*cte.c**2/cte.e # eV
        gamma = E/E0
        beta = 1-(1/(2*(gamma**2)))
        return beta







# poder armazenar bending normal, FldM para poles=2, ou mapa de campo de bending
# class BendingMagnet(srw.SRWLMagFldM):
#todo: conferir se argumento _R faz sentido pra bending, caso nao, retirar args, kwargs
class BendingMagnet(Magnet):

    def __init__(self, B:float, L:float, Ledge:float=0, *args, **kwargs):
        """
        Bending magnet field.

        Parameters
        ----------
        B : float
            Magnetic field amplitude [T]
        L : float
            Effective magnet length [m]
        Ledge : float, optional
            Edge length for field variation from 10 to 90% [m]
        """
        super().__init__()

        self.B = B
        
        self.srw_mag = srw.SRWLMagFldM(_G=B, _m=1, _n_or_s='n', _Leff=L,
                                       _Ledge=Ledge, *args, **kwargs)

    @property
    def length(self) -> float:
        return self.srw_mag.Leff
    
    @length.setter
    def length(self, value):
        self.srw_mag.Leff = value

    
    def critical_energy(self,E:float) -> float:
        """
        Bending critical energy.
        
        Parameters
        ----------
        E : float
            Accelerator energy [eV]
        """
        gamma = self.calc_gamma(E)
        rho = (gamma*cte.m_e)*cte.c/(cte.e*self.B)
        wc = (3/2)*(cte.c/rho)*(gamma**3)
        return cte.hbar*wc/cte.e
    
    


#todo: permitir configurar campos harmonicos
class Undulator(Magnet):

    def __init__(self, period_length:float, nr_periods:int, Kv:float, Kh:float, settings:Optional[list]=None):
        """
        Undulator magnetic field.

        Parameters
        ----------
        period_length [m] : float
        nr_periods : int
            Number of periods.
        Kh, Kv : float
            Deflection parameters for vertical and horizontal planes, respectively.
        settings: list or list of lists, optional
            Parameters to configurate harmonic magnetic fields:
            [0] plane ('h' or 'v')
            [1] initial phase
            [2] symmetric boolean
            [3] coefficient of field transverse dependence
        """
        super().__init__()

        self._period_length = period_length
        self._nr_periods = nr_periods
        self._K = np.array([Kh,Kv])

        Bv = self.K_to_B(period_length, Kv)
        Bh = self.K_to_B(period_length, Kh)
        
        self.fieldV = srw.SRWLMagFldH(_n=1,_h_or_v='v',_B=Bv)
        self.fieldH = srw.SRWLMagFldH(_n=1,_h_or_v='h',_B=Bh)

        self.srw_mag = srw.SRWLMagFldU(_arHarm=[self.fieldV, self.fieldH],
                                       _per=period_length, _nPer=nr_periods)


    @property
    def period_length(self) -> float:
        """Undulator period length [m]"""
        return self._period_length
    
    @property
    def nr_periods(self) -> int:
        """Undulator number of periods"""
        return self._nr_periods
    
    @property
    def K(self):
        """Undulator deflection parameters [Kh,Kv]; [Kx,Ky]"""
        return self._K
    
    @property
    def length(self) -> float:
        return (8+self.srw_mag.nPer)*self.srw_mag.per
    


    @staticmethod
    def B_to_K(period_length:float, B:float) -> float:
        """
        Convert undulator field amplitude B to deflection parameter K.

        Parameters
        ----------
        period_length : float
            Undulator period length; lambda_u [m]
        B : float
            Undulator harmonic field amplitude [T]
        """
        K = cte.e*B*period_length/(2*np.pi*cte.m_e*cte.c)
        return K
    
    @staticmethod
    def K_to_B(period_length:float, K:float) -> float:
        """
        Convert undulator deflection parameter K to field amplitude B.

        Parameters
        ----------
        period_length : float
            Undulator period length; lambda_u [m]
        K : float
            Undulator harmonic field deflection parameter [adim]
        """
        B = 2*np.pi*cte.m_e*cte.c*K/(cte.e*period_length)
        return B
    
    #todo: typing de harmn int or array like
    def harmonic_wavelength(self, harmn:int, E:float, theta:float) -> float:
        """
        Radiation harmonic wavelength. Application of the undulator equation
        for a elliptical one.

        Parameters
        ----------
        harmn : float | array_like
            Harmonic number
        E : float
            Accelerator energy [eV]
        theta : float
            Observation angle with respect to undulator axis [rad]

        Returns
        -------
        lambda : float | array_like
            Undulator harmonic wavelength [m]
        """
        if np.any(harmn==0):
            raise ValueError("Energy or harmonic cannot be zero!")
    
        gamma = self.calc_gamma(E)
        lambda_n = self.period_length/( 2*harmn*(gamma**2) )
        lambda_n *= 1 + np.sum(self.K**2)/2 + (gamma*theta)**2

        return lambda_n

    #todo: typing de harmn int or array like
    def harmonic_energy(self, harmn:int, E:float, theta:float) -> float:
        """
        Radiation harmonic energy.

        Parameters
        ----------
        n : float | array_like
            Harmonic number
        E : float
            Accelerator energy [eV]
        theta : float
            Observation angle with respect to undulator axis [rad]

        Returns
        -------
        Eph : float | array_like
            Undulator energy harmonic [eV]
        """
        lambda_n = self.harmonic_wavelength(harmn,E,theta)
        energy_n = cte.h*cte.c/(lambda_n*cte.e)
        return energy_n
    
    #todo: typing de harmn int or array like
    def energy_to_harmonic(self, energy, E, theta):
        """
        Calculate the harmonic number given the energy.

        Parameters
        ----------
        energy : float
            Energy of the radiation [eV]
        E : float
            Accelerator energy [eV]
        theta : float
            Observation angle with respect to undulator axis [rad]
        """
        harmn1 = 1
        energy1 = self.harmonic_energy(harmn1, E, theta)
        return energy/energy1
    
    def adjust_energy(self,energy,E,theta):

        harmn = self.energy_to_harmonic(energy,E,theta)

        step0 = harmn[1]-harmn[0] #todo: media em todos os espacamentos
        n = np.ceil(1/step0) # number of steps between consecutive harmonics
        step = 1/n

        harmn_i, harmn_f = np.ceil(harmn[0]), np.floor(harmn[-1])
        k_left  = np.floor( n*(harmn_i-harmn[0])  )
        k_right = np.floor( n*(harmn[-1]-harmn_f) )
        harmn_i, harm_f = harmn_i-k_left*step, harmn_f+k_right*step

        harmn = np.arange(harmn_i,harm_f,step)

        energy = self.harmonic_energy(harmn,E,theta)

        return energy
    




# OBS: usando allocate para centros iguais a zero nao e' necessario escrever tal parte
#todo: aceitar 0, 1 e 2 como field type tambem
class MagnetCnt:

    def __init__(self,
            magnets:list[Magnet],
            field_type:_Method,
            centers=None,
            axes=None,
            angs=None
        ):
        """
        Magnetic Field Container.

        Parameters
        ----------
        magnets : list[Magnet]
            List of magnetic elements.
        centers : list[list|array], optional
            List of center coordinates of each magnet in `magnets` [m]; format
            is [xc,yc,zc].
        axes : list[list|array], optional
            List of axes unit vectors for each magnet in `magnets`; format
            is [vx,vy,vz].
        angs : list[float], optional
            List of rotation angles with respect to the axes [rad].
        field_type : {'manual', 'undulator', 'bending'}, optional
            Synchrotron Radiation calculation method. Defaults to 'bending'.
        """
        self.magnets = magnets if magnets else []

        N = len(magnets)
        self.centers = centers if centers else 3*[N*[0]]
        if axes:
            axes = np.asarray(axes)
            if not np.isclose(axes[0]**2+axes[1]**2+axes[2]**2,1):
                raise ValueError("Axes must be unit vectors!")
            else:
                self.axes = axes
        else:
            self.axes = [N*[0],N*[0],N*[1]]
        self.angs = angs if angs else np.zeros(N)

        self.cnt = srw.SRWLMagFldC(
            _arMagFld=self.srw_mags,
            _arXc=self.centers[0],_arYc=self.centers[1],_arZc=self.centers[2],
            _arVx=self.axes[0], _arVy=self.axes[1], _arVz=self.axes[2],
            _arAng=self.angs
        )

        self.fieldtype = None

        # SRW Synchrotron Radiation calculation method
        idx_method = {'manual':0, 'undulator':1, 'bending':2}.get(field_type)
        if idx_method:
            self.fieldtype = idx_method
        else:
            raise ValueError("Invalid field type!")

    @property
    def srw_mags(self) -> list[srw.SRWLMagFld]:
        """List of SRW magnetic fields in the container"""
        return [magnet.srw_mag for magnet in self.magnets]



    def calc_trajectory(self,
            np=50000,
            ctlim:_CT=None,
            particle:Optional[srw.SRWLParticle]=None,
            precisions=0
        ) -> srw.SRWLPrtTrj:

        ptraj = srw.SRWLPrtTrj(_np=np)

        if particle:
            ptraj.partInitCond = particle
        else:
            ptraj.partInitCond.gamma = 3/0.51099890221e-03

        ptraj: srw.SRWLPrtTrj = srwl.CalcPartTraj(ptraj, self.cnt, precisions)

        if isinstance(ctlim,(tuple,list)):
            ptraj.ctStart, ptraj.ctEnd = ctlim
        elif ctlim=='length':
            mag = self.magnets[0]
            ptraj.ctStart, ptraj.ctEnd = -mag.length/2, mag.length/2
        else:
            ptraj.ctStart, ptraj.ctEnd = ptraj.arZ[0], ptraj.arZ[-1]
        
        return ptraj



#*: usar meshgrid 3D de arrays flat x, y, z para checar organizacao do return,
#*: daí calcular campo talvez com radia nesses pontos e fazer o mesmo, mas sem
#*: meshgrid, usando loop no padrao srw, e comparar organizacao

#todo: aceitar Bs no padrao srw por compatibilidade, mas criar classmethod para
#todo: padrao meshgrid e numpy

#*: padrao srw :*#
#*: for iz in range nz:
#*:     for iy in range ny:
#*:         for ix in range nx:
#*:             b.append

class FieldMap(Magnet):

    def __init__(self, Bx, By, Bz, x, y, z, reps=1,interpolation='bilinear'):
        """
        Field map magnetic field.

        Parameters
        ----------
        Bx, By, Bz : array_like
            Flat arrays of magnetic field components, composed from
            nested loops through the spatial points x, y, z;
            [Bn for rangez for rangey for rangex], z outmost loop and
            x the innermost loop.
        x, y, z : array_like
            Arrays defining the 3D grid on which the magnetic field is defined.
        reps : int, optional
            Number of periods or repetitions of the field in z direction.
            Default is 1.
        interpolation : {'bilinear','biquadratic','bicubic','1D+2D'}, optional
            Interpolation method to use for calculating the magnetic field in
            the interior of the grid. '1D+2D' means 1D cubic spline on the
            longitudinal direction and 2D bicubic on the transverse plane.
            Default is 'bilinear'.
        """
        nx, ny, nz = len(x), len(y), len(z)
        # rx, ry, rz = x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]
        arZ, arY, arX = np.meshgrid(z,y,x,indexing='ij')
        arX, arY, arZ = arX.ravel(), arY.ravel(), arZ.ravel()

        idx_interp = {'bilinear':1,'biquadratic':2,
                      'bicubic':3,'1D+2D':4}.get(interpolation)
        if not idx_interp: raise ValueError("Invalid interpolation method!")
        
        self.srw_mag = srw.SRWLMagFld3D(_arBx=Bx,_arBy=By,_arBz=Bz,
                                        _nx=nx,_ny=ny,_nz=nz,
                                        _nRep=reps,_interp=idx_interp,
                                        _arX=arX,_arY=arY,_arZ=arZ)
        
        
        self._functional_field = None #todo: funcao de interpolacao do campo


    @classmethod
    def from_srw_file(cls, filename, reps=1, interpolation='bilinear'):

        with open(filename,'r') as f:

            f.readline() # skip header

            xStart = float(f.readline().split()[0].lstrip('#'))
            xStep = float(f.readline().split()[0].lstrip('#'))
            xNp = int(f.readline().split()[0].lstrip('#'))
            xRange = (xNp-1)*xStep if xNp>1 else xStep
            xEnd = xStart+xRange

            yStart = float(f.readline().split()[0].lstrip('#'))
            yStep = float(f.readline().split()[0].lstrip('#'))
            yNp = int(f.readline().split()[0].lstrip('#'))
            yRange = (yNp-1)*yStep if yNp>1 else yStep
            yEnd = yStart+yRange

            zStart = float(f.readline().split()[0].lstrip('#'))
            zStep = float(f.readline().split()[0].lstrip('#'))
            zNp = int(f.readline().split()[0].lstrip('#'))
            zRange = (zNp-1)*zStep if zNp>1 else zStep
            zEnd = zStart+zRange

            Bx, By, Bz = np.loadtxt(f)[:xNp*yNp*zNp].T.copy()

        x = np.linspace(xStart,xEnd,xNp)
        y = np.linspace(yStart,yEnd,yNp)
        z = np.linspace(zStart,zEnd,zNp)

        return cls(Bx, By, Bz, x, y, z, reps, interpolation)


    def get_field(self, x, y, z):
        # usar self._functional_field para avaliar campo nos pontos x, y, z
        pass
