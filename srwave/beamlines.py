
from typing import Optional, Literal, Union

import numpy as np

import srwpy.srwlib as srw
import srwpy.srwlpy as srwl

from . import radiation as _radiation, optics, magnets

_OptElements = Union[optics.OpticalElement, list[optics.OpticalElement]]


#?: trocar evaluate por outro nome ?

#todo: tornar Beamline que nem Accelerator, da fac. funcionar que nem lista

#todo: cada elemento optico da lista poderia vir com wavefront, dai teriamos naturalmente o store de cada propagacao

class Beamline:

    def __init__(self,
        opt_elements: _OptElements,
        radiation: Optional[_radiation.RadiationSource] = None,
        evaluate: bool = False,
    ):

        self.opt_elements = opt_elements
        self.radiation = radiation

        if evaluate: self.calc_propagate_wfr()

    @property
    def opt_elements(self) -> list[optics.OpticalElement]:
        """List of optical elements in the beamline"""
        return self._opt_elements

    @opt_elements.setter
    def opt_elements(self, elements):

        self._opt_elements = []

        if isinstance(elements, (list, tuple)):
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

    def add_opt_element(self, element: optics.OpticalElement):

        if not isinstance(element, optics.OpticalElement):
            raise TypeError("Optical Element type" +
                            f" '{type(element).__name__}' " +
                            "not supported")

        self._opt_elements.append(element)

    
    def PropagElecField(self, oe_arr, pp_arr):
        optBl = srw.SRWLOptC(oe_arr, pp_arr)
        srwl.PropagElecField(self.radiation.wfr, optBl)

    def propagate_wfr(self):

        if not self.opt_elements:
            return False

        self.PropagElecField(self.srw_opts, self.props_params)

        return True


    def calc_propagate_wfr(self):
        self.radiation.calc_wfr()
        self.propagate_wfr()



#todo: cada line e' algo muito especifico, entao nomear de maneira especifica tambem

#todo: tirar evaluate das lines para sempre ter que executar

class PinholeLine(Beamline):

    def __init__(self, energy, x, y, d, D, apertx, aperty,
                 B=0.5642, L=3,
                 material='Al', thickness=1e-3,
                 evaluate=True):
        """Pinhole line using bending magnet and dissipation filter, without
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
        bm = magnets.BendingMagnet(B, L)
        fields = magnets.MagnetCnt(magnets=[bm], fieldtype='bending')

        SR = _radiation.SynchrotronRadiation(energy, x, y, d, fields=fields)

        self.filter = optics.AbsorptionFilter(x, y, energy, thickness, material)
        self.aperture = optics.Slit(apertx, aperty)
        self.aperture.prop_params['ra'] = 10.0
        self.screen = optics.Drift(D)

        super().__init__(radiation=SR,
                         opt_elements=[self.filter, self.aperture, self.screen],
                         evaluate=evaluate)

        SR.load_beam("carcara")





class ToroidalMirrorLine(_radiation.SynchrotronRadiation):

    def __init__(self, energy, d, D, x, y, apertx, aperty,
                 ang, tang_len, sag_len, R_tang, R_sag,
                 B=0.5642, L=3):
        """Generic beamline with focusing toroidal mirror.

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
        bm = magnets.BendingMagnet(B, L)
        fields = magnets.MagnetCnt(magnets=[bm], field_type='bending')
        super().__init__(energy, d, x, y, fields=fields)
        self.calc_wfr()

        self.load_beam("carcara")

        self.mask = optics.Slit(apertx, aperty)
        self.mirror = optics.ToroidalMirror(ang, tang_len, sag_len, R_tang, R_sag)
        self.screen = optics.Drift(D)
        self.screen.prop_params['re'] = 2.0
        self.screen.prop_params['propagator'] = 'Quadratic'

        bl = _radiation.Beamline(line=[self.mask, self.mirror, self.screen])

        self.propagate_wfr(beamline=bl)


class Carcara(_radiation.SynchrotronRadiation):

    def __init__(self, apertx, aperty, xc=0, yc=0, energy=11e3, d=17, D=17,
                 mirror_error=''):
        """Carcara beamline.

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
        w = np.linspace(-3, 3, 200)*1e-3
        bm = magnets.BendingMagnet(B=0.5642, L=0.853)
        fields = magnets.MagnetCnt(magnets=[bm], field_type='bending')
        super().__init__(energy=energy, d=d, x=w, y=w, fields=fields)
        self.calc_wfr()

        self.load_beam("carcara")

        self.mask = optics.Slit(4e-3, 4e-3)
        self.mask.prop_params['ra'] = 4.0
        self.mirror = optics.ToroidalMirror(
            ang=18.78e-3, tang_len=0.22, sag_len=0.008,
            R_tang=905.271, R_sag=0.3192
        )

        line = [self.mask, self.mirror]

        if mirror_error:

            directory = __file__.replace('beamlines.py', 'data_opt_elements/carcara/')
            errorfile = {'zeiss': 'CAX_M1_Zeiss_height_error_sh.dat',
                         'meas': 'CAX_height_error_FZI_220mm_sh.dat'}[mirror_error]

            self.error = optics.MirrorError(
                filename=directory+errorfile, unit=1e-3, ang=18.78e-3,
                orientation='x', L=0.22, W=0.008
            )

            line.append(self.error)

        self.space = optics.Drift(distance=4.4)
        self.space.prop_params['propagator'] = 'Quadratic'
        self.aperture1 = optics.Slit(apertx, aperty, xc=xc, yc=yc)
        self.screen = optics.Drift(distance=D-4.4)
        self.screen.prop_params['ra'] = 2.0
        self.screen.prop_params['re'] = 2.0
        self.screen.prop_params['propagator'] = 'Quadratic'

        line.extend([self.space, self.aperture1, self.screen])

        bl = _radiation.Beamline(line=line)

        self.propagate_wfr(beamline=bl)


# * changes the state of SR, propagating its wavefront
def caustic(SR: _radiation.SynchrotronRadiation, dists,
            coord: Literal['x', 'y'], energy: float, X: float, Y: float):

    dl = dists[1:] - dists[:-1]
    lengths = np.insert(dl, 0, dists[0])

    arrsI = []
    ranges = []

    for d in lengths:

        screen = optics.Drift(distance=d)
        bl = Beamline(radiation=SR, line=screen)
        bl.propagate_wfr()

        arrIxn, [rangexn] = SR.calc_intensity(coord, energy, X, Y)

        arrsI.append(arrIxn)
        ranges.append(rangexn)

    arrsI = np.array(arrsI).T

    return arrsI, ranges


'''
def energy_loop_intensity(SR)
'''
