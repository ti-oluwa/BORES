"""
**BORES**

3D 3-Phase Black-Oil Reservoir Modelling and Simulation Framework.
"""

import logging
import os

from .analyses import *
from .boundary_conditions import *
from .config import *
from .constants import *
from .errors import *
from .factories import *
from .fluids import *
from .fractures import *
from .grids import *
from .models import *
from .monitoring import *
from .precision import *
from .rock_fluid import *
from .serialization import *
from .serialization import register_ndarray_serializers
from .simulate import *
from .solvers import *
from .states import *
from .stores import *
from .streams import *
from .tables import *
from .timing import *
from .types import *
from .utils import *
from .visualization import *
from .wells import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Use custom ndarray serializer if `BORES_SAVE_RAW_NDARRAY != True`
if os.getenv("BORES_SAVE_RAW_NDARRAY", "f").lower() not in (
    "t",
    "y",
    "yes",
    "true",
    "1",
):
    register_ndarray_serializers()

use_32bit_precision()

__version__ = "0.1.0"
