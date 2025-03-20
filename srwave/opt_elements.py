import typing

import numpy as np
import scipy.constants as cte

import xraylib
import srwpy.srwlib as srw
from optlnls.surface import SRW_figure_error


_PropParams = typing.Literal['auto_rs_before', 'auto_rs_after', 'auto_rs_prec',
                             'propagator', 'rs', 'ra', 're', 'ra_h', 're_h',
                             'ra_v', 're_v', 'p1', 'p2', 'p3'] #, 'p4', 'p5', 'p6', 'p7', 'p8']


class Propagation:

    propagators = {'Standard': 0, 'Quadratic': 1, 'QuadraticSpecial': 2,
                   'FromWaist': 3, 'ToWaist': 4}
    params = ['auto_rs_before', 'auto_rs_after', 'auto_rs_prec',
              'propagator',
              'rs',
              'ra_h', 're_h', 'ra_v', 're_v',
              'p1', 'p2', 'p3']   #, 'p4', 'p5', 'p6', 'p7', 'p8']

    #[ 9]: Type of wavefront Shift before Resizing (not yet implemented); default is 0
    #[10]: New Horizontal wavefront Center position after Shift (not yet implemented); default is 0
    #[11]: New Vertical wavefront Center position after Shift (not yet implemented); default is 0
    #[12]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Horizontal Coordinate
    #[13]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Vertical Coordinate
    #[14]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Longitudinal Coordinate
    #[15]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Horizontal Coordinate
    #[16]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Vertical Coordinate

    def __init__(self):

        dft_values = [False, False, 1.0,
                      self.propagators['Standard'],
                      False,
                      1.0, 1.0, 1.0, 1.0,
                      0, 0, 0]

        self._prop_params = {
            key: value for key, value in zip(self.params,dft_values)
        }

    # def __repr__(self):
    #     pass

    def __str__(self):
        rstr = ''
        for key, value in self._prop_params.items():
            rstr += f'{key:<14} : {value}\n'
        rstr.rstrip('\n')
        return rstr

    def __getitem__(self, key:_PropParams):
        if key in self.params:
            return self._prop_params[key]
        elif key=='ra':
            return self._prop_params['ra_h'], self._prop_params['ra_v']
        elif key=='re':
            return self._prop_params['re_h'], self._prop_params['re_v']
        else:
            raise KeyError(f"Param '{key}' not found")

    def __setitem__(self, key, value):
        if key in self.params:
            self._prop_params[key] = value if key != 'propagator' else self.propagators[value]
        elif key=='ra':
            self._prop_params['ra_h'] = value
            self._prop_params['ra_v'] = value
        elif key=='re':
            self._prop_params['re_h'] = value
            self._prop_params['re_v'] = value
        else:
            raise KeyError(f"Param '{key}' not found")
        
    def __iter__(self):
        for key in self.params:
            yield self._prop_params[key]
    


class OpticalElement:

    def __init__(self):

        self.srw_opt = srw.SRWLOpt()
        self.prop_params = Propagation()

    @property
    def propagators(self) -> list[str]:
        """List of available diffraction propagators for the optical element."""
        return list(self.prop_params.propagators)


class Drift(OpticalElement):

    def __init__(self,dist,*args,**kwargs):
        super().__init__()
        
        self.srw_opt = srw.SRWLOptD(dist,*args,**kwargs)


class DissipationFilter(OpticalElement):

    def __init__(self,x,y,energy,thickness,material,density=None,*args,**kwargs):
        super().__init__()
        
        xi, xf, nx = x[0], x[-1], len(x)
        yi, yf, ny = y[0], y[-1], len(y)
        if np.isscalar(energy): energy = np.array([energy])
        ei, ef, ne = energy[0], energy[-1], len(energy)

        t = self.transmission(energy, thickness, material, density)
        transm = np.zeros(2*ne*nx*ny)
        transm[::2] = np.tile(np.abs(t),reps=nx*ny) # desconsiderando imag
        
        self.srw_opt = srw.SRWLOptT(
            _x = (xi+xf)/2, _rx = xf-xi, _nx = nx, # x center and range
            _y = (yi+yf)/2, _ry = yf-yi, _ny = ny, # y center and range
            _arTr = transm, # transmission array
            _extTr = 1,
            _eStart = ei, _eFin = ef, _ne = ne, # energy range
            *args, **kwargs
        )

    @classmethod
    def transmission(cls, energy, thickness, material, density=None) -> complex:
        """
        Calculates the transmission phasor t through a given material. To convert to \
        transmittance T calculate T=abs(t)**2

        Parameters
        ----------
        energy : float or array
            Energy of the radiation [eV]
        thickness : float
            Thickness of the material [m]
        material : str
            Material of the filter. Supported default materials are 'Al', 'Cu', 'Mo'
        density : float, optional
            Material density in g/cm^3. If None, the density of the material is used

        Returns
        -------
        transmission : complex array
            Transmission through the material
        """

        if density is None:
            density = {'Al': 2.7, 'Cu': 8.935, 'Mo': 10.223}[material]

        if np.isscalar(energy):
            n = xraylib.Refractive_Index(material, energy*1e-3, density)
        else:
            n = np.array([xraylib.Refractive_Index(material, e*1e-3, density) for e in energy])

        wl = cte.h*cte.c/(energy*cte.e)
        k = 2*np.pi/wl
        t = np.exp(1j*n*k*thickness)

        return t


class GaussianFilter(OpticalElement):

    def __init__(self,x,y,energy,e_center,e_sigma,*args,**kwargs):
        super().__init__()

        xi, xf, nx = x[0], x[-1], len(x)
        yi, yf, ny = y[0], y[-1], len(y)
        if np.isscalar(energy): energy = [energy]
        ei, ef, ne = energy[0], energy[-1], len(energy)
        
        t = self.transmission(energy, e_center, e_sigma) # array
        transm = np.array([t.real, t.imag]).T.reshape(-1)
        transm = ny*nx*list(transm)

        t = self.transmission(energy, e_center, e_sigma)
        transm = np.zeros(2*ne*nx*ny)
        transm[::2] = np.tile(np.abs(t),reps=nx*ny) # desconsiderando imag

        self.srw_opt = srw.SRWLOptT(
            _x = (xi+xf)/2, _rx = xf-xi, _nx = nx, # x center and range
            _y = (yi+yf)/2, _ry = yf-yi, _ny = ny, # y center and range
            _arTr = transm, # transmission array
            _extTr = 1,
            _eStart = ei, _eFin = ef, _ne = ne, # energy range
            *args, **kwargs
        )


    @classmethod
    def transmission(cls, energy, e_center: float, e_sigma: float) -> complex:
        """
        Calculates transmission phasor t through a Gaussian filter. to convert to \
        transmittance T calculate T=abs(t)**2

        Parameters
        ----------
        energy : float or array
            Photon energy(ies) [eV]
        e_center : float
            Central energy [eV]
        e_sigma : float
            Standard deviation of transmission curve [eV]

        Returns
        -------
        transmission : complex array
            Transmission through the filter
        """

        t = np.exp(-(energy-e_center)**2/(2*e_sigma**2)/2) + 0j

        return t


class Aperture(OpticalElement):

    def __init__(self,Dx,Dy,xc=0.0,yc=0.0,shape='r'):
        super().__init__()

        self.srw_opt = srw.SRWLOptA(_shape = shape, #'r': rectangle
                                        _ap_or_ob = 'a', #'a': aperture
                                        _Dx = Dx, #"Width [m]"
                                        _Dy = Dy, #"Height [m]"
                                        _x = xc, # horizontal center [m]
                                        _y = yc) # vertical center [m]


class PlaneMirror(OpticalElement):

    def __init__(self,ang,tang_len,sag_len,*args,**kwargs):
        super().__init__()

        self.srw_opt = srw.SRWLOptMirPl(*args,**kwargs)

        self.srw_opt.set_dim_sim_meth(
            _size_tang = tang_len, #"Tangential Size [m]"
            _size_sag = sag_len, #"Sagittal Size [m]"
            _ap_shape = 'r', # shape of aperture ('r': rectangular)
            _sim_meth = 2, # simulation method (2: "thick" approximation)
            _treat_in_out = 1 # 1: input and output wfr at center of mirror
        )
        self.srw_opt.set_orient(
            _nvx = -np.sqrt(1-ang**2), # horizontal coordinate of central normal vector
            _nvy = 0, # vertical coordinate of central normal vector
            _nvz = -ang, # longitudinal coordinate of central normal vector
            _tvx = ang, # horizontal coordinate of central tangential vector
            _tvy = 0, # vertical coordinate of central tangential vector
            _x = 0, # horizontal position of mirror center [m]
            _y = 0 # vertical position of mirror center [m]
        )


class ToroidalMirror(OpticalElement):

    def __init__(self,ang,tang_len,sag_len,R_tang,R_sag,*args,**kwargs):
        super().__init__()
        
        self.srw_opt = srw.SRWLOptMirTor(_rt = R_tang, #"Tangential Radius [m]"
                                             _rs = R_sag, #"Sagittal Radius [m]"
                                             *args,**kwargs)
        
        self.srw_opt.set_dim_sim_meth(
            _size_tang = tang_len, #"Tangential Size [m]"
            _size_sag = sag_len, #"Sagittal Size [m]"
            _ap_shape = 'r', # shape of aperture ('r': rectangular)
            _sim_meth = 2, # simulation method (2: "thick" approximation)
            _treat_in_out = 1 # 1: input and output wfr at center of mirror
        )
        self.srw_opt.set_orient(
            _nvx = -np.sqrt(1-ang**2), # horizontal coordinate of central normal vector
            _nvy = 0, # vertical coordinate of central normal vector
            _nvz = -ang, # longitudinal coordinate of central normal vector
            _tvx = ang, # horizontal coordinate of central tangential vector
            _tvy = 0, # vertical coordinate of central tangential vector
            _x = 0, # horizontal position of mirror center [m]
            _y = 0 # vertical position of mirror center [m]
        )


class MirrorError(OpticalElement):

    def __init__(self,filename,unit,ang,orientation,L,W):
        super().__init__()

        self.srw_opt = SRW_figure_error(filename,unit,ang,ang,orientation,L=L,W=W)


class Lens(OpticalElement):

    def __init__(self,fx,fy,xc=0,yc=0):
        super().__init__()

        self.srw_opt = srw.SRWLOptL(_Fx=fx,_Fy=fy,
                                    _x=xc,_y=yc)
    
    @classmethod
    def lens_from_dists(cls,dist_src_lens,dist_lens_screen):
        f = dist_src_lens*dist_lens_screen/(dist_src_lens+dist_lens_screen)
        return cls(fx=f,fy=f)


class FresnelZonePlate(OpticalElement):

    def __init__(self,nr_zones,radius_n,zthickness=10e-6,
                 delta1=1e-6,atlen1=0.1,delta2=0,atlen2=1e-6,
                 xc=0,yc=0,energy_avg=0):
        super().__init__()

        self.srw_opt = srw.SRWLOptZP(
            _nZones = nr_zones, _rn = radius_n,
            _thick = zthickness,
            _delta1 = delta1, _atLen1 = atlen1,
            _delta2 = delta2, _atLen2 = atlen2,
            _x = xc, _y = yc,
            _e = energy_avg
        )


if __name__ == "__main__":

    pp = Propagation()

    print(pp)