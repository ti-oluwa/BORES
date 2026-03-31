"""
**BORES**

3D 3-Phase Black-Oil Reservoir Modelling and Simulation Framework.
"""

import logging
import os

import numba

# Set numba thread count to match available cores (or from environment variable)
cpu_count = os.cpu_count() or 1
numba.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", cpu_count)))

from ._precision import *
from .analyses import *
from .boundary_conditions import *
from .capillary_pressures import *
from .config import *
from .constants import *
from .errors import *
from .factories import *
from .fluids import *
from .fractures import *
from .grids import *
from .models import *
from .monitoring import *
from .relperm import *
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
