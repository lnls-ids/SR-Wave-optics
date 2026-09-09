from .beamlines import Beamline, PinholeLine
from . import light_sources
from .magnets import BendingMagnet, Undulator, MagnetCnt, FieldMap
from .optics import Drift, AbsorptionFilter, GaussianFilter, Slit, Obstacle, PlaneMirror, ToroidalMirror, MirrorError, Lens, FresnelZonePlate
from .radiation import SynchrotronRadiation, GaussianRadiation, PointRadiation
from . import utils


from importlib.metadata import version

__version__ = version("srwave")

__all__ = [
    "__version__",
    "Beamline",
    "PinholeLine",
    "light_sources",
    "BendingMagnet",
    "Undulator",
    "MagnetCnt",
    "FieldMap",
    "Drift",
    "AbsorptionFilter",
    "GaussianFilter",
    "Slit",
    "Obstacle",
    "PlaneMirror",
    "ToroidalMirror",
    "MirrorError",
    "Lens",
    "FresnelZonePlate",
    "SynchrotronRadiation",
    "GaussianRadiation",
    "PointRadiation",
    "utils",
]
