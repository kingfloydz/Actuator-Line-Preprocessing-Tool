"""High-level aerodynamic analysis using the embedded XFOIL core."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from xfoil_core import XFoilCore, XFoilCoreError


def setup_logging(workdir: str) -> logging.Logger:
    """Configure file-only logging for XFOIL processing."""
    log_file = os.path.join(workdir, "xfoil_analysis.log")
    logger = logging.getLogger("bladeprocess.xfoil")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


@dataclass(frozen=True)
class SectionResult:
    """Structured result for a single polar evaluation."""

    section: int
    angle_deg: float
    cl: float | None
    cd: float | None
    converged: bool


@dataclass(frozen=True)
class SectionTask:
    section: int
    airfoil_path: str
    reynolds: float
    angles_pos: tuple[float, ...]
    angles_neg: tuple[float, ...]
    iter_limit: int
    mach: float
    lib_path: str
    n_panels: int = 160
    cv_par: float = 1.0
    cte_ratio: float = 0.15
    ctr_ratio: float = 0.2


def _process_section_task(task: SectionTask) -> tuple[int, List[SectionResult], str | None]:
    try:
        core = XFoilCore(task.lib_path)
    except XFoilCoreError as exc:
        return task.section, [], f"XFOIL core unavailable: {exc}"

    try:
        x_vals, y_vals = XfoilAnalyzer._read_airfoil_coordinates(task.airfoil_path)
    except Exception as exc:  # pragma: no cover - defensive
        return task.section, [], str(exc)

    try:
        with suppress_xfoil_output():
            cl_pos, cd_pos, status_pos, cl_neg, cd_neg, status_neg = core.compute_coefficients(
                x_vals,
                y_vals,
                task.angles_pos,
                task.angles_neg,
                reynolds=task.reynolds,
                mach=task.mach,
                iter_limit=task.iter_limit,
                n_panels=task.n_panels,
                cv_par=task.cv_par,
                cte_ratio=task.cte_ratio,
                ctr_ratio=task.ctr_ratio,
            )
    except Exception as exc:  # pragma: no cover - defensive
        return task.section, [], str(exc)

    results_pos = XfoilAnalyzer._build_section_results(
        task.section,
        task.angles_pos,
        cl_pos,
        cd_pos,
        status_pos,
    )
    results_neg = XfoilAnalyzer._build_section_results(
        task.section,
        task.angles_neg,
        cl_neg,
        cd_neg,
        status_neg,
    )
    return task.section, results_pos + results_neg, None


def _worker_wrapper(task: SectionTask, queue: multiprocessing.Queue):
    """Wrapper to run task and put result in queue."""
    try:
        result = _process_section_task(task)
        queue.put(result)
    except Exception as exc:
        queue.put((task.section, [], str(exc)))


@contextlib.contextmanager
def suppress_xfoil_output():
    """Redirect low-level stdout/stderr to suppress Fortran noise."""
    try:
        # Flush Python-level buffers
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Save original FDs
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        
        # Open null device
        devnull = os.open(os.devnull, os.O_RDWR)
        
        try:
            # Redirect stdout/stderr to null
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            yield
        finally:
            # Restore FDs
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            
            # Close temporary FDs
            os.close(devnull)
            os.close(saved_stdout)
            os.close(saved_stderr)
            
    except Exception:
        # Fallback: just redirect Python objects (won't stop Fortran output)
        with open(os.devnull, "w") as f:
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = f, f
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err


import sys  # Needed for the context manager above


class XfoilAnalyzer:
    """Run aerodynamic polars by calling the minimal XFOIL core directly."""

    RESULTS_EXCEL_FILENAME = "xfoil_cl_cd_results.xlsx"
    AIRFOIL_FILE_TEMPLATE = "airfoil_blade{blade_number}_section{section_num}.dat"
    DEFAULT_ITER_LIMIT = 300
    DEFAULT_MACH_NUMBER = 0.0

    DEFAULT_ANGLES_NON_NEG: Sequence[float] = (
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
        2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75,
        5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25,
        7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75,
        10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75,
        12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0,
        14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5,
        19.0, 19.5, 20.0,
        20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
        25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0,
        29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5,
        34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0,
    )

    DEFAULT_ANGLES_NEG: Sequence[float] = (
        -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0, -2.25,
        -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0, -4.25,
        -4.5, -4.75, -5.0, -5.25, -5.5, -5.75, -6.0, -6.25,
        -6.5, -6.75, -7.0, -7.25, -7.5, -7.75, -8.0,
        -8.5, -9.0, -9.5, -10.0, -10.5, -11.0, -11.5, -12.0,
        -12.5, -13.0, -13.5, -14.0, -14.5, -15.0, -15.5, -16.0,
        -16.5, -17.0, -17.5, -18.0, -18.5, -19.0, -19.5, -20.0,
    )

    def __init__(
        self,
        param_manager,
        geometry,
        *,
        angles_non_neg: Iterable[float] | None = None,
        angles_neg: Iterable[float] | None = None,
        iter_limit: int | None = None,
        mach: float | None = None,
        n_panels: int = 160,
        cv_par: float = 1.0,
        cte_ratio: float = 0.15,
        ctr_ratio: float = 0.2,
    ) -> None:
        self.param_manager = param_manager
        self.geometry = geometry
        self.workdir = param_manager.xfoil_workdir
        self.logger = setup_logging(self.workdir)

        lib_hint = getattr(param_manager, "xfoil_core_lib_path", None)
        try:
            self.core = XFoilCore(lib_hint)
        except XFoilCoreError as exc:  # pragma: no cover - defensive
            raise RuntimeError("XFOIL core library is unavailable") from exc

        self.iter_limit = iter_limit or self.DEFAULT_ITER_LIMIT
        self.mach = self.DEFAULT_MACH_NUMBER if mach is None else mach
        self.n_panels = n_panels
        self.cv_par = cv_par
        self.cte_ratio = cte_ratio
        self.ctr_ratio = ctr_ratio
        self.angles_non_neg = list(angles_non_neg or self.DEFAULT_ANGLES_NON_NEG)
        self.angles_neg = list(angles_neg or self.DEFAULT_ANGLES_NEG)
        self.section_reynolds = self._load_section_reynolds()
        self.results: List[SectionResult] = []

    def _load_section_reynolds(self) -> dict[int, float]:
        reynolds_data = {}
        for blade_idx, blade in enumerate(self.geometry.blade_sections):
            for section_idx, section in enumerate(blade):
                section_num = section_idx + 1
                reynolds_data[section_num] = float(section.get('reynolds', 1e6))
        return reynolds_data

    @staticmethod
    def _read_airfoil_coordinates(filepath: str) -> tuple[List[float], List[float]]:
        x_vals, y_vals = [], []
        with open(filepath, "r") as f:
            lines = f.readlines()
            # Skip header if present (simple heuristic: skip first line if it contains text)
            start_idx = 0
            if lines and any(c.isalpha() for c in lines[0]):
                start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x_vals.append(float(parts[0]))
                        y_vals.append(float(parts[1]))
                    except ValueError:
                        continue
        return x_vals, y_vals

    @staticmethod
    def _build_section_results(
        section: int,
        angles: Sequence[float],
        cl_arr: np.ndarray,
        cd_arr: np.ndarray,
        status_arr: np.ndarray,
    ) -> List[SectionResult]:
        results = []
        for i, angle in enumerate(angles):
            converged = bool(status_arr[i])
            results.append(
                SectionResult(
                    section=section,
                    angle_deg=angle,
                    cl=float(cl_arr[i]) if converged else None,
                    cd=float(cd_arr[i]) if converged else None,
                    converged=converged,
                )
            )
        return results

    def _execute_task_locally(self, task: SectionTask) -> tuple[int, List[SectionResult], str | None]:
        return _process_section_task(task)

    def _log_section_completion(self, results: List[SectionResult]):
        converged_count = sum(1 for r in results if r.converged)
        total = len(results)
        if results:
            self.logger.info(
                "Section %d: %d/%d converged.",
                results[0].section,
                converged_count,
                total,
            )

    def _execute_tasks_parallel(self, tasks: List[SectionTask], max_workers: int, timeout: float, results_dict: dict):
        """Execute tasks in parallel with timeout protection."""
        task_queue = list(tasks)  # Copy
        running_procs = {}  # {pid: (process, start_time, task_section)}
        result_queue = multiprocessing.Queue()

        while task_queue or running_procs:
            # Start new processes if slots available
            while task_queue and len(running_procs) < max_workers:
                task = task_queue.pop(0)
                p = multiprocessing.Process(target=_worker_wrapper, args=(task, result_queue))
                p.start()
                running_procs[p.pid] = (p, time.time(), task.section)
                self.logger.info(f"Started processing section {task.section}")

            # Check for results
            while not result_queue.empty():
                try:
                    section, res, error = result_queue.get_nowait()
                    if error:
                        self.logger.error(f"Section {section} skipped: {error}")
                    else:
                        results_dict[section] = res
                        self._log_section_completion(res)
                except Exception:
                    break

            # Check for timeouts and finished processes
            pids_to_remove = []
            for pid, (p, start_time, section) in running_procs.items():
                if not p.is_alive():
                    pids_to_remove.append(pid)
                elif time.time() - start_time > timeout:
                    self.logger.error(f"Section {section} timed out after {timeout}s. Killing process.")
                    p.terminate()
                    p.join(timeout=1)
                    if p.is_alive():
                        p.kill()
                    pids_to_remove.append(pid)
            
            for pid in pids_to_remove:
                del running_procs[pid]
            
            time.sleep(0.1)

    def run_xfoil_analysis(self) -> List[List[object]]:
        """Evaluate all configured sections sequentially or in parallel with timeout protection."""
        blade_index = 0
        total_sections = getattr(self.param_manager, "num_sections", len(self.section_reynolds))
        lib_path = str(self.core._lib_path)
        tasks: List[SectionTask] = []

        for section_num in range(1, total_sections + 1):
            airfoil_file = self.AIRFOIL_FILE_TEMPLATE.format(
                blade_number=blade_index + 1,
                section_num=section_num,
            )
            airfoil_path = os.path.join(self.workdir, airfoil_file)
            reynolds = float(self.section_reynolds.get(section_num, 1e6))
            tasks.append(
                SectionTask(
                    section=section_num,
                    airfoil_path=airfoil_path,
                    reynolds=reynolds,
                    angles_pos=tuple(self.angles_non_neg),
                    angles_neg=tuple(self.angles_neg),
                    iter_limit=self.iter_limit,
                    mach=self.mach,
                    lib_path=lib_path,
                    n_panels=self.n_panels,
                    cv_par=self.cv_par,
                    cte_ratio=self.cte_ratio,
                    ctr_ratio=self.ctr_ratio,
                )
            )

        per_section_results: dict[int, List[SectionResult]] = {}
        max_workers = max(1, min(getattr(self.param_manager, "max_processes", 1), os.cpu_count() or 1))

        # Calculate timeout based on number of angles
        # Reference from exe version: BASE_TIMEOUT + len(angles) * ANGLE_TIMEOUT
        # Let's be generous: 30s base + 5s per angle
        sample_task = tasks[0] if tasks else None
        if sample_task:
            num_angles = len(sample_task.angles_pos) + len(sample_task.angles_neg)
            task_timeout = 30 + num_angles * 2.2
        else:
            task_timeout = 300

        if max_workers == 1:
            for task in tasks:
                section, section_results, error = self._execute_task_locally(task)
                if error:
                    self.logger.error("Section %s skipped: %s", section, error)
                    continue
                per_section_results[section] = section_results
                self._log_section_completion(section_results)
        else:
            self._execute_tasks_parallel(tasks, max_workers, task_timeout, per_section_results)

        # Flatten results for export
        flat_results = []
        for section_num in sorted(per_section_results.keys()):
            for res in per_section_results[section_num]:
                flat_results.append([
                    res.section,
                    res.angle_deg,
                    res.cl,
                    res.cd,
                    res.converged
                ])
        return flat_results
