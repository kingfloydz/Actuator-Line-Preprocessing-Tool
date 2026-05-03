# XFOIL Core Library

This directory contains the minimal Fortran wrapper that exposes the XFOIL routines needed by `xfoil_analyzer.py`.

## Building the Shared Library

1. Install a Fortran compiler (e.g. `gfortran`) and NumPy.
2. From the repository root, run:

   ```powershell
   python -m numpy.f2py -c xfoil_core\xfoil_core_wrapper.f90 xfoil\src\*.f -m xfoil_core --opt="-O2"
   ```

   The command will create `xfoil_core.pyd` (Windows) or `xfoil_core.so` (Linux/macOS) in the current directory.

3. Copy the generated module into `xfoil_core/lib/` as `xfoil_core.dll` (or the platform-specific equivalent) so that `xfoil_core.api.XFoilCore` can locate it.

You can override the library location by setting the environment variable `XFOIL_CORE_LIBRARY` or by passing `library_path` directly when instantiating `XFoilCore`.
