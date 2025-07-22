
from . import light_sources
from . import radiation_source
from . import opt_elements
from . import mag_elements
from . import rad_plots
from . import beamlines
from . import utils


from importlib.metadata import version

__version__ = version("srwave")

__all__ = [
    "__version__",
    "beamlines",
    "light_sources",
    "opt_elements",
    "rad_plots",
    "radiation_source",
    "utils",
]
