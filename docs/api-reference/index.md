# API Reference

## Overview

This section documents the programmatic interfaces of the BORES framework. The reference is organized into two areas: the PVT correlations, which work on single values and on numpy grids alike, and the full public API surface exported from the `bores` package.

The correlations module is the computational foundation of the simulator. Every time BORES needs to compute an oil formation volume factor, gas compressibility, water viscosity, or any other pressure-volume-temperature property, it calls one of these functions. You can also call them directly in your own scripts for standalone PVT calculations, unit conversions, or validation against laboratory data.

---

## Module Organization

| Module | Description |
| --- | --- |
| `bores.correlations` | PVT correlation functions for oil, gas and water, for numbers and arrays |
| `bores` | Top-level package re-exporting all public classes, functions, and constants |

---

## How to Use This Reference

**Correlations** compute PVT properties from conditions. The same function takes a single set of conditions or whole grids.

```python
import numpy as np
from bores import correlations

Bo = correlations.compute_oil_formation_volume_factor_standing(
    temperature=200.0,  # degrees F
    oil_specific_gravity=0.85,
    gas_gravity=0.7,
    gas_to_oil_ratio=500.0,  # SCF/STB
)

Bo_grid = correlations.compute_oil_formation_volume_factor_standing(
    temperature=temperature_grid,
    oil_specific_gravity=0.85,
    gas_gravity=0.7,
    gas_to_oil_ratio=Rs_grid,
)
```

**Full API** lists every class, function, and constant exported from the `bores` package. Use it as a lookup when you know the name of something but need to confirm the import path or check the signature.

---

## Pages

- [Correlations](correlations.md) - PVT functions for oil, gas and water, for single values and arrays
- [Full API](full-api.md) - Complete listing of the public API surface
