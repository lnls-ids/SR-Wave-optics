
import typing

import numpy as np

from . import radiation_source as rs
from . import opt_elements as oe
from . import mag_elements as me



class Line:

    def __init__(self):

        self.sourceWfr = rs.SynchrotronRadiation()
        self.arrWfr = []

        self.optBl = rs.Beamline()


class PinholeLine(rs.SynchrotronRadiation):

    def __init__(self,energy,d,D,x,y,apertx,aperty,
                 B=0.5642,L=3,
                 material='Al',thickness=1e-3):
        """
        Pinhole line using bending magnet and dissipation filter, without
        monocromator or lens.

        Parameters
        ----------
        energy : float
            Photon energy [eV].
        d : float
            Distance from source to pinhole [m].
        D : float
            Distance from pinhole to screen [m].
        x, y : array
            Source wavefront horizontal and vertical positions [m]. Suggestion:
            np.linspace(apert/2-10e-6,apert/2+10e-6,120)
        apertx, aperty : float
            Horizontal and vertical pinhole aperture width [m]. Tipically order
            of um.
        B : float, optional
            Bending magnet field [T]. Default SIRIUS B1: 0.5642 T.
        L : float, optional
            Bending magnet effective length [m]. Default 3 m to avoid edge
            radiation effects. For reference, SIRIUS B1 is 0.853 m.
        material : string, optional
            Dissipation filter material. Default to Al, Aluminum.
        thickness : float, optional
            Dissipation filter thickness [m]. Default to 1e-3 m.
        """
        bm = me.BendingMagnet(B,L)
        fields = me.MagnetCnt(magnets=[bm],field_type='bending')
        super().__init__(energy, d, x, y, fields = fields)
        self.calc_wfr()

        self.load_beam("carcara")

        self.filter = oe.DissipationFilter(x,y,energy,thickness,material)
        self.aperture = oe.Aperture(apertx,aperty)
        self.aperture.prop_params['ra'] = 10.0
        self.screen = oe.Drift(D)

        self.beamline = rs.Beamline(line=[self.filter,self.aperture,self.screen])

        self.propagate_wfr(self.beamline)




class ToroidalMirrorLine(rs.SynchrotronRadiation):

    def __init__(self, energy, d, D, x, y, apertx, aperty,
                 ang, tang_len, sag_len, R_tang, R_sag,
                 B=0.5642, L=3):
        """
        Generic beamline with focusing toroidal mirror.

        Parameters
        ----------
        energy : float
            Photon energy [eV].
        d : float
            Distance from source to mirror [m].
        D : float
            Distance from mirror to screen [m].
        x, y : array
            Source wavefront horizontal and vertical positions [m]. Suggestion:
            np.linspace(-3,3,200)*1e-3
        apertx, aperty : float
            Horizontal and vertical widths of mask before mirror [m].
        ang : float
            Incidence angle with respect to mirror's plane [rad].
        tang_len : float
            Tangential length of the toroidal mirror [m].
        sag_len : float
            Sagittal length of the toroidal mirror [m].
        R_tang : float
            Tangential radius of curvature of the mirror [m].
        R_sag : float
            Sagittal radius of curvature of the mirror [m].
        B : float, optional
            Bending magnet field [T]. Default SIRIUS B1: 0.5642 T.
        L : float, optional
            Bending magnet effective length [m]. Default: 3 m, large value to
            avoid edge radiation effects. For reference, SIRIUS B1 is 0.853 m.

        """
        bm = me.BendingMagnet(B,L)
        fields = me.MagnetCnt(magnets=[bm],field_type='bending')
        super().__init__(energy, d, x, y, fields = fields)
        self.calc_wfr()

        self.load_beam("carcara")

        self.mask = oe.Aperture(apertx,aperty)
        self.mirror = oe.ToroidalMirror(ang,tang_len,sag_len,R_tang,R_sag)
        self.screen = oe.Drift(D)
        self.screen.prop_params['re'] = 2.0
        self.screen.prop_params['propagator'] = 'Quadratic'

        bl = rs.Beamline(line=[self.mask,self.mirror,self.screen])

        self.propagate_wfr(beamline=bl)




class Carcara(rs.SynchrotronRadiation):

    def __init__(self,apertx,aperty,xc=0,yc=0,energy=11e3,d=17,D=17,
                 mirror_error=''):
        """
        Carcara beamline.

        Parameters
        ----------
        apertx, aperty : float
            Horizontal and vertical widths of first aperture after mirror [m].
        xc, yc : float, optional
            Center of the first aperture after the mirror [m]. Default: 0 m.
        energy : float, optional
            Photon energy [eV]. Default SIRIUS CARCARA energy: 11 keV.
        d : float, optional
            Distance from source to mirror [m]. Default: 17 m.
        D : float, optional
            Distance from mirror to screen [m]. Default: 17 m.
        mirror_error : string, optional
            Type of mirror error. Options: 'zeiss', 'meas'. Default: no error.
        """
        w = np.linspace(-3,3,200)*1e-3
        bm = me.BendingMagnet(B=0.5642,L=0.853)
        fields = me.MagnetCnt(magnets=[bm],field_type='bending')
        super().__init__(energy=energy, d=d, x=w, y=w, fields=fields)
        self.calc_wfr()

        self.load_beam("carcara")

        self.mask = oe.Aperture(4e-3,4e-3)
        self.mask.prop_params['ra'] = 4.0
        self.mirror = oe.ToroidalMirror(
            ang=18.78e-3,tang_len=0.22,sag_len=0.008,
            R_tang=905.271,R_sag=0.3192
        )

        line = [self.mask,self.mirror]

        if mirror_error:

            directory = __file__.replace('beamlines.py','data_opt_elements/carcara/')
            errorfile = {'zeiss': 'CAX_M1_Zeiss_height_error_sh.dat',
                         'meas': 'CAX_height_error_FZI_220mm_sh.dat'}[mirror_error]
            
            self.error = oe.MirrorError(
                filename=directory+errorfile, unit=1e-3, ang=18.78e-3,
                orientation='x', L=0.22, W=0.008
            )

            line.append(self.error)

        self.space = oe.Drift(dist=4.4)
        self.space.prop_params['propagator'] = 'Quadratic'
        self.aperture1 = oe.Aperture(apertx,aperty,xc=xc,yc=yc)
        self.screen = oe.Drift(dist=D-4.4)
        self.screen.prop_params['ra'] = 2.0
        self.screen.prop_params['re'] = 2.0
        self.screen.prop_params['propagator'] = 'Quadratic'

        line.extend([self.space,self.aperture1,self.screen])

        bl = rs.Beamline(line=line)

        self.propagate_wfr(beamline=bl)




#* changes the state of SR, propagating its wavefront
def caustic(SR: rs.SynchrotronRadiation, dists,
            coord: typing.Literal['x','y'], energy: float, X: float, Y: float):
    
    dl = dists[1:] - dists[:-1]
    lengths = np.insert(dl,0,dists[0])

    arrsI = []
    ranges = []

    for d in lengths:
        
        screen = oe.Drift(dist=d)
        bl = rs.Beamline(line=screen)
        SR.propagate_wfr(beamline=bl, store_steps=False)

        arrIxn, [rangexn] = SR.calc_intensity(coord,energy,X,Y)

        arrsI.append(arrIxn)
        ranges.append(rangexn)

    arrsI = np.array(arrsI).T

    return arrsI, ranges

'''
def energy_loop_intensity(SR)
'''