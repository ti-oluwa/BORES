---
title: Installation Guide - BORES Documentation
description: Install BORES reservoir simulator with uv or pip. Includes platform-specific instructions for Linux, macOS, and Windows, dependency setup, and troubleshooting common issues.
tags:
  - guide
  - beginner
  - python
  - reservoir-simulation
keywords: install BORES, Python reservoir simulator setup, uv package manager, pip install, BORES dependencies, NumPy SciPy installation
author: ti-oluwa
---

# Installation

This page covers how to install BORES, verify your setup, and troubleshoot common issues. BORES supports Python 3.10 and above on Linux, macOS, and Windows.

---

## Prerequisites

Before installing BORES, make sure you have the following:

- **Python 3.10 or later** - BORES uses modern typing features and requires Python 3.10 as a minimum. You can check your version with `python --version`.
- **A C compiler** - Numba (used for JIT-compiled numerical kernels) occasionally needs to compile extensions. On Linux, `gcc` is typically available by default. On macOS, install Xcode command line tools with `xcode-select --install`. On Windows, the Microsoft Visual C++ Build Tools are sufficient.
- **A working internet connection** - Required to download BORES and its dependencies from PyPI.

BORES depends on several scientific Python packages that will be installed automatically:

| Package | Purpose |
|---------|---------|
| numpy | Array operations and linear algebra |
| scipy | Sparse matrix solvers and numerical methods |
| numba | JIT compilation for performance-critical functions |
| attrs / cattrs | Data models and serialization |
| h5py / zarr / orjson | HDF5, Zarr, and JSON storage backends |
| plotly | Visualization (series, maps, 3D volumes) |

---

## Install BORES

=== "uv (recommended)"

    [uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependency resolution efficiently. If you do not have `uv` installed yet, you can install it with:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Then install BORES:

    ```bash
    uv add bores-framework
    ```

    If you are working from a cloned copy of the BORES repository, you can install it in development mode:

    ```bash
    uv sync
    ```

    This reads the `pyproject.toml` and installs all dependencies, including optional dev tools.

=== "pip"

    You can install BORES with standard pip:

    ```bash
    pip install bores-framework
    ```

    Or to ensure that pip uses an index for MKL-optimized numpy/scipy for x86_64 Linux/Windows:

    ```bash
    pip install bores-framework --extra-index-url https://urob.github.io/numpy-mkl
    ```

    For development mode from a local clone:

    ```bash
    pip install -e ".[dev]"
    ```

!!! tip "Virtual Environments"

    Always install BORES inside a virtual environment to avoid conflicts with other packages. With `uv`, virtual environments are managed automatically. With `pip`, create one using `python -m venv .venv` and activate it before installing.

---

## Verify Your Installation

Run the following script to confirm that BORES is installed correctly and its core dependencies are available:

```python
import bores

print(f"BORES version: {bores.__version__}")
print(f"Default precision: {bores.get_dtype()}")
```

Expected output:

```
BORES version: 0.1.0
Default precision: <class 'numpy.float32'>
```

If you see an `ImportError`, double-check that you installed BORES in the correct Python environment and that all dependencies resolved successfully.

---

## Optional Dependencies

BORES includes several optional packages that extend its capabilities. These are installed by default with the standard installation, but are worth knowing about if you are working in a minimal environment.

### Thermodynamics - CoolProp and thermo

BORES uses [CoolProp](http://www.coolprop.org/) and [thermo](https://github.com/CalebBell/thermo) for advanced fluid property calculations, particularly when computing gas properties from first principles. These packages enable accurate PVT calculations for non-standard gases like CO2 or nitrogen.

```bash
pip install coolprop thermo
```

### Visualization - Plotly and Trame

The visualization module relies on [Plotly](https://plotly.com/python/) for generating charts and 3D renders, along with [Kaleido](https://github.com/nicholasgasior/kaleido) for static image export and [Trame](https://kitware.github.io/trame/) for interactive 3D volume visualization in web browsers.

```bash
pip install plotly kaleido trame
```

### Storage - HDF5 and Zarr

BORES supports multiple storage backends for saving and loading simulation results. [h5py](https://www.h5py.org/) provides HDF5 support, while [zarr](https://zarr.dev/) enables chunked, compressed array storage suitable for large simulations.

```bash
pip install h5py zarr
```

### Data frames - pandas and polars

Results can be turned into data frames with the `as_dataframe` function of `bores.schedule.summary`, `bores.reservoir.state`, `bores.wells.state` and `bores.blackoil.state`. Neither library is required by BORES. Install the one you use with the `backend` argument, or both.

```bash
pip install 'bores-framework[pandas]'      # backend="pandas"
pip install 'bores-framework[polars]'      # backend="polars"
pip install 'bores-framework[dataframes]'  # both
```

---

## Troubleshooting

### Numba compilation errors

Numba compiles Python functions to machine code on first use. If you encounter compilation errors, try the following:

1. **Update Numba**: Ensure you have numba >= 0.63.0 installed.
2. **Clear the cache**: Delete Numba's cache directory, typically found at `__pycache__` folders containing `.nbi` and `.nbc` files.
3. **Check your compiler**: Run `numba -s` to print system information and verify that a compatible C compiler is detected.

!!! note "First-run latency"

    The first time you import BORES or run a simulation, Numba compiles several cached functions. This may take 10-30 seconds. Subsequent runs will be much faster because compiled code is cached to disk.

### Missing BLAS/LAPACK libraries

scipy and numpy depend on optimized linear algebra libraries. On Linux, if you see errors about missing BLAS or LAPACK:

=== "Ubuntu / Debian"

    ```bash
    sudo apt-get install libopenblas-dev liblapack-dev
    ```

=== "Fedora / RHEL"

    ```bash
    sudo dnf install openblas-devel lapack-devel
    ```

=== "macOS"

    macOS ships with the Accelerate framework, which provides BLAS/LAPACK. No additional installation is needed.

=== "Windows"

    The numpy and scipy wheels on PyPI bundle their own BLAS/LAPACK (OpenBLAS). No additional installation is typically required. If you installed via `uv` on x86_64, MKL-optimized versions are used automatically.

### HDF5 build failures

If `h5py` fails to install, it may be because the HDF5 C library is not available on your system:

=== "Ubuntu / Debian"

    ```bash
    sudo apt-get install libhdf5-dev
    ```

=== "macOS"

    ```bash
    brew install hdf5
    ```

=== "Windows"

    Pre-built wheels for h5py are available on PyPI and should install without issues. If you encounter problems, try upgrading pip: `pip install --upgrade pip`.

---

## Numba Threading Layer (Optional Performance Tuning)

BORES uses [Numba](https://numba.readthedocs.io/) for JIT-compiled numerical kernels. By default, Numba uses multiple threads for parallelization across your CPU cores. The threading backend can be configured for optimal performance.

### Threading Backends

Numba supports three threading layers in order of preference:

1. **TBB (Intel Threading Building Blocks)** - Best performance, cross-platform support
2. **OMP (OpenMP)** - Good performance, widely available
3. **Workqueue** - Fallback, available on all platforms

**Important**: BORES does not require TBB and will work correctly with any available threading layer. By default, Numba automatically selects the best available option: `tbb -> omp -> workqueue`.

### Check Your Current Threading Layer

To see which threading layer Numba is currently using, run:

```python
import numba


def get_threading_backend():
    try:
        return numba.threading_layer()
    except Exception:
        return "unknown"


print(f"Numba threading layer: {get_threading_backend()}")
```

Expected output (one of):

    Numba threading layer: tbb
    Numba threading layer: omp
    Numba threading layer: workqueue

!!! tip "Thread Pool Size (Optional)"

    BORES automatically configures the thread pool size for optimal performance. You only need to adjust this if you want to override the defaults (e.g., on shared systems to avoid oversubscription).

    To limit the number of threads, set:

    ```bash
    export NUMBA_NUM_THREADS=4
    ```

    Or programmatically:

    ```python
    import numba

    numba.set_num_threads(4)
    ```

### Optional: Install TBB for Better Performance

If you want to ensure **TBB** (the fastest threading layer) is used, install it on your platform:

=== "Linux / WSL"

    First, install the system TBB library:

    ```bash
    sudo apt-get install libtbb-dev
    ```

    Then install the Python packages:

    ```bash
    pip install tbb tbb4py
    uv add tbb tbb4py  # or with uv
    ```

    Verify TBB is enabled:

    ```python
    import numba

    print(numba.threading_layer())  # Should print: tbb
    ```

=== "macOS"

    Install TBB via Homebrew:

    ```bash
    brew install tbb
    ```

    Then install the Python packages:

    ```bash
    pip install tbb tbb4py
    uv add tbb tbb4py  # or with uv
    ```

    Verify TBB is enabled:

    ```python
    import numba

    print(numba.threading_layer())  # Should print: tbb
    ```

=== "Windows"

    Install the Python packages directly (Windows wheels include TBB binaries):

    ```bash
    pip install tbb tbb4py
    uv add tbb tbb4py  # or with uv
    ```

    Verify TBB is enabled:

    ```python
    import numba

    print(numba.threading_layer())  # Should print: tbb
    ```

!!! warning "TBB is Optional"

    - If TBB installation fails, BORES will automatically fall back to OMP or workqueue—no crashes, no errors.
    - You do **not** need to force the threading layer. Numba handles fallback automatically.
    - Do **not** set `NUMBA_THREADING_LAYER=tbb` manually, as this will cause crashes if TBB is unavailable.

### Graceful Fallback Behavior

BORES is designed to work reliably with any threading layer. If you want to explicitly ensure safe fallback behavior at startup:

    import os
    import numba

    # Ensure safe fallback: tbb -> omp -> workqueue
    os.environ.setdefault("NUMBA_THREADING_LAYER", "default")

    # Print what's available
    print(f"Numba threading layer: {numba.threading_layer()}")

    import bores  # Now import BORES

---

## Platform Notes

### Linux

Linux is the primary development and testing platform for BORES. All features, including MKL-optimized linear algebra on x86_64, work out of the box with the standard installation.

### macOS

BORES works on both Intel and Apple Silicon Macs. On Apple Silicon (M1/M2/M3/M4), Numba uses the ARM64 backend, which is fully supported. The Accelerate framework provides optimized BLAS/LAPACK.

### Windows

BORES is supported on Windows 10 and later. If you are using Windows Subsystem for Linux (WSL), the Linux installation instructions apply. Native Windows installation works with both `uv` and `pip`.

!!! info "MKL Optimization"

    On x86_64 Linux and Windows, BORES is configured to use Intel MKL-optimized versions of numpy and scipy when installed via `uv`. This provides significant performance improvements for linear algebra operations, which are central to reservoir simulation. macOS and non-x86 architectures use standard OpenBLAS builds.

---

## Next Steps

With BORES installed, head to the [Quickstart](quickstart.md) to build and run your first simulation.
