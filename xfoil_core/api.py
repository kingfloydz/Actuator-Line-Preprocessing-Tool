import ctypes
import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np


class XFoilCoreError(RuntimeError):
    """Raised when the low-level XFOIL core cannot be used."""


class XFoilCore:
    """Thin ctypes wrapper around the minimal XFOIL Fortran core."""

    _ENV_KEY = "XFOIL_CORE_LIBRARY"

    def __init__(self, library_path: Optional[os.PathLike] = None):
        self._lib_path = self._resolve_library_path(library_path)
        self._lib = ctypes.CDLL(str(self._lib_path))
        self._fn = self._lib.xfoil_compute_cl_cd
        self._fn.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),  # x
            ctypes.POINTER(ctypes.c_double),  # y
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),  # alpha_pos
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),  # alpha_neg
            ctypes.c_double,  # Reynolds
            ctypes.c_double,  # Mach
            ctypes.c_int,     # iteration limit
            ctypes.c_int,     # n_panels
            ctypes.c_double,  # cv_par
            ctypes.c_double,  # cte_ratio
            ctypes.c_double,  # ctr_ratio
            ctypes.POINTER(ctypes.c_double),  # CL pos
            ctypes.POINTER(ctypes.c_double),  # CD pos
            ctypes.POINTER(ctypes.c_int),     # status pos
            ctypes.POINTER(ctypes.c_double),  # CL neg
            ctypes.POINTER(ctypes.c_double),  # CD neg
            ctypes.POINTER(ctypes.c_int),     # status neg
        ]
        self._fn.restype = None

    @classmethod
    def _default_candidates(cls) -> Sequence[Path]:
        candidates: list[Path] = []
        env = os.getenv(cls._ENV_KEY)
        if env:
            candidates.append(Path(env))
        module_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                module_dir / "lib" / "xfoil_core.dll",
                module_dir / "lib" / "libxfoil_core.dll",
                module_dir / "xfoil_core.dll",
                module_dir / "libxfoil_core.dll",
            ]
        )
        return candidates

    def _resolve_library_path(self, explicit: Optional[os.PathLike]) -> Path:
        if explicit:
            candidate = Path(explicit)
            if not candidate.exists():
                raise XFoilCoreError(f"Specified XFOIL core library not found: {candidate}")
            return candidate

        for candidate in self._default_candidates():
            if candidate.exists():
                return candidate
        raise XFoilCoreError(
            "Unable to locate XFOIL core shared library. "
            "Set XFOIL_CORE_LIBRARY or pass library_path explicitly."
        )

    def compute_coefficients(
        self,
        x: Iterable[float],
        y: Iterable[float],
        alphas_pos: Iterable[float],
        alphas_neg: Iterable[float],
        reynolds: float,
        mach: float = 0.0,
        iter_limit: int = 100,
        n_panels: int = 160,
        cv_par: float = 1.0,
        cte_ratio: float = 0.15,
        ctr_ratio: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y coordinate arrays must have the same shape")
        if x_arr.ndim != 1:
            raise ValueError("x and y must be one-dimensional sequences")

        alpha_pos_arr = np.ascontiguousarray(list(alphas_pos), dtype=np.float64)
        alpha_neg_arr = np.ascontiguousarray(list(alphas_neg), dtype=np.float64)

        cl_pos = np.zeros_like(alpha_pos_arr)
        cd_pos = np.zeros_like(alpha_pos_arr)
        status_pos = np.zeros(alpha_pos_arr.size, dtype=np.int32)

        cl_neg = np.zeros_like(alpha_neg_arr)
        cd_neg = np.zeros_like(alpha_neg_arr)
        status_neg = np.zeros(alpha_neg_arr.size, dtype=np.int32)

        self._fn(
            ctypes.c_int(x_arr.size),
            x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(alpha_pos_arr.size),
            alpha_pos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(alpha_neg_arr.size),
            alpha_neg_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(reynolds),
            ctypes.c_double(mach),
            ctypes.c_int(iter_limit),
            ctypes.c_int(n_panels),
            ctypes.c_double(cv_par),
            ctypes.c_double(cte_ratio),
            ctypes.c_double(ctr_ratio),
            cl_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cd_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            status_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            cl_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cd_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            status_neg.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return cl_pos, cd_pos, status_pos, cl_neg, cd_neg, status_neg


__all__ = ["XFoilCore", "XFoilCoreError"]
