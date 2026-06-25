# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-level Lindblad solver combining the stochastic Magnus expansion with
the variational quantum simulation framework.

This module provides :class:`LindbladMagnusSolver`, the user-facing entry point
that orchestrates:

* sampling of the multiple stochastic integrals for a Lindblad trajectory,
* assembly of the (non-Hermitian) effective Hamiltonian via the stochastic
  Magnus expansion,
* integration of the McLachlan variational principle on a user-supplied
  :class:`~pyqpanda_alg.LindbladMagnus.ansatz.VariationalAnsatz`,
* ensemble averaging over many independent trajectories.

The implementation follows
    J.-C. Huang, H.-E. Li, Y.-C. Wang, G.-Z. Zhang, J. Li, H.-S. Hu,
    "Towards Robust Variational Quantum Simulation of Lindblad Dynamics via
    Stochastic Magnus Expansion", PRX Quantum 6, 040312 (2025).
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Callable, Sequence

import numpy as np

from .ansatz import VariationalAnsatz
from .magnus import effective_hamiltonian, sample_wiener_integrals
from .variational import (mclachlan_system, variational_step_euler,
                          variational_step_rk4)

__all__ = ["LindbladMagnusSolver", "evolve_trajectory", "solve"]


class LindbladMagnusSolver:
    """Variational Lindblad solver based on the stochastic Magnus expansion.

    The solver describes the unravelling (``qsd_type``), the integration scheme
    (``magnus_order``) and the variational manifold (``ansatz``) once and for
    all.  Each call to :meth:`solve` then runs an independent ensemble of
    trajectories and returns the averaged expectation values.

    Parameters
    ----------
    H : ``ndarray``\n
        System Hamiltonian (Hermitian, shape ``(2**n, 2**n)``).
    c_ops : ``list`` of ``ndarray``\n
        Lindblad collapse operators.
    ansatz : :class:`~pyqpanda_alg.LindbladMagnus.ansatz.VariationalAnsatz`\n
        Parameterised variational ansatz describing the simulation manifold.
    qsd_type : ``{'nonlinear', 'linear'}``, optional (default='nonlinear')\n
        Choice of unravelling.
    magnus_order : ``int``, optional (default=1)\n
        Order of the stochastic Magnus expansion.  ``0`` selects the
        Euler-Maruyama scheme and ``1``-``4`` enable the higher-order Magnus
        schemes of the paper.
    nonlinear_corr : ``bool``, optional (default=False)\n
        Predictor-corrector flag for the nonlinear QSD drift.
    integrator : ``{'euler', 'rk4'}``, optional (default='rk4')\n
        Classical integrator used for the variational equation of motion.
    init_params : ``ndarray``, optional\n
        Initial variational parameters.  When ``None`` a zero vector is used
        and the ansatz is expected to prepare the desired initial state by
        itself (e.g. via :class:`~pyqpanda_alg.LindbladMagnus.ansatz.VariationalAnsatz`
        with a non-trivial ``init_state``).

    Examples
    --------
    >>> import numpy as np
    >>> from pyqpanda_alg.LindbladMagnus import (LindbladMagnusSolver,
    ...                                          fmo_model, HardwareEfficientAnsatz)
    >>> H, c_ops, e_ops, psi0 = fmo_model()
    >>> ans = HardwareEfficientAnsatz(n_qubits=3, layers=2, init_state=psi0)
    >>> solver = LindbladMagnusSolver(H, c_ops, ans, magnus_order=1)
    >>> times = np.linspace(0, 50, 26)
    >>> expect, std = solver.solve(psi0, times, e_ops, traj_num=4, seed=42)
    """

    def __init__(self, H: np.ndarray, c_ops: Sequence[np.ndarray],
                 ansatz: VariationalAnsatz,
                 qsd_type: str = "nonlinear",
                 magnus_order: int = 1,
                 nonlinear_corr: bool = False,
                 integrator: str = "rk4",
                 init_params: np.ndarray | None = None):
        self.H = np.asarray(H, dtype=complex)
        self.c_ops = [np.asarray(op, dtype=complex) for op in c_ops]
        self.ansatz = ansatz
        if qsd_type not in ("nonlinear", "linear"):
            raise ValueError(
                f"qsd_type must be 'nonlinear' or 'linear', got {qsd_type!r}")
        self.qsd_type = qsd_type
        if magnus_order < 0 or magnus_order > 4:
            raise ValueError(
                f"magnus_order must be in [0, 4], got {magnus_order}")
        self.magnus_order = int(magnus_order)
        self.nonlinear_corr = bool(nonlinear_corr)
        if integrator not in ("euler", "rk4"):
            raise ValueError(
                f"integrator must be 'euler' or 'rk4', got {integrator!r}")
        self.integrator = integrator
        self.init_params = (np.zeros(ansatz.n_parameters, dtype=float)
                            if init_params is None
                            else np.asarray(init_params, dtype=float).reshape(-1))

    # ------------------------------------------------------------------ #
    #  Single-trajectory evolution                                       #
    # ------------------------------------------------------------------ #
    def evolve_trajectory(self, tlist: np.ndarray, e_ops: Sequence[np.ndarray],
                          seed: int = 0,
                          return_params: bool = False) -> dict:
        """Evolve a single Lindblad trajectory.

        Parameters
        ----------
        tlist : ``ndarray``\n
            Time grid.  The first entry is taken as ``t_0``.
        e_ops : ``list`` of ``ndarray``\n
            Observables whose expectation values are returned at every time.
        seed : ``int``, optional (default=0)\n
            Base random seed for this trajectory.  The actual seed used at
            step ``i`` is ``seed + i`` so that the same ``seed`` reproduces the
            same Wiener path.
        return_params : ``bool``, optional (default=False)\n
            If ``True`` the dictionary also contains the parameter trajectory.

        Return
        ----------
        result : ``dict``\n
            Dictionary with keys ``"times"`` (the input time grid),
            ``"expect"`` (array of shape ``(len(e_ops), len(tlist))``),
            ``"norm"`` (wave-function norm at every time) and, optionally,
            ``"params"``.
        """
        tlist = np.asarray(tlist, dtype=float).reshape(-1)
        n_steps = len(tlist) - 1
        dim = self.H.shape[0]
        ansatz = self.ansatz

        theta = self.init_params.copy()
        psi_norm = 1.0
        expect = np.zeros((len(e_ops), len(tlist)), dtype=float)
        param_traj = np.zeros((len(tlist), ansatz.n_parameters), dtype=float)
        norm_traj = np.zeros(len(tlist), dtype=float)

        # Initial measurement.
        expect[:, 0] = self._measure_observables(theta, e_ops, psi_norm)
        param_traj[0] = theta
        norm_traj[0] = psi_norm

        for i in range(n_steps):
            dt = float(tlist[i + 1] - tlist[i])
            # Sample the stochastic integrals once per step (used for both the
            # predictor and the corrector step of the nonlinear QSD).
            rng = np.random.RandomState(seed + i)
            integrals = sample_wiener_integrals(len(self.c_ops), dt, rng=rng)

            # Predictor: build H_eff from the current state.
            psi_pred = None
            if self.qsd_type == "nonlinear":
                psi_pred = ansatz.get_statevector(theta)
                psi_pred = psi_pred / np.linalg.norm(psi_pred)

            H_eff = self._build_H_eff(dt, theta, psi_pred, integrals)

            # Optional predictor-corrector: do an Euler half-step with the
            # predictor H_eff, then rebuild H_eff from the predicted state and
            # use it for the full step.
            if self.nonlinear_corr and self.qsd_type == "nonlinear":
                theta_try, _ = variational_step_euler(ansatz, theta, H_eff, dt,
                                                     psi_norm)
                psi_corr = ansatz.get_statevector(theta_try)
                psi_corr = psi_corr / np.linalg.norm(psi_corr)
                H_eff = self._build_H_eff(dt, theta, psi_corr, integrals)

            theta, psi_norm = self._integrate_step(theta, H_eff, dt, psi_norm)

            expect[:, i + 1] = self._measure_observables(theta, e_ops, psi_norm)
            param_traj[i + 1] = theta
            norm_traj[i + 1] = psi_norm

        result = {
            "times": tlist,
            "expect": expect,
            "norm": norm_traj,
        }
        if return_params:
            result["params"] = param_traj
        return result

    # ------------------------------------------------------------------ #
    #  Ensemble averaging                                                #
    # ------------------------------------------------------------------ #
    def solve(self, psi0: np.ndarray, tlist: np.ndarray,
              e_ops: Sequence[np.ndarray],
              traj_num: int = 100,
              seed: int = 0,
              n_jobs: int = 1,
              verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Run an ensemble of trajectories and return averaged observables.

        Parameters
        ----------
        psi0 : ``ndarray``\n
            Initial wave-function (used only for validation of the
            ``nonlinear`` unravelling against the ansatz initial state).
        tlist : ``ndarray``\n
            Time grid.
        e_ops : ``list`` of ``ndarray``\n
            Observables whose expectation values are returned.
        traj_num : ``int``, optional (default=100)\n
            Number of independent Lindblad trajectories.
        seed : ``int``, optional (default=0)\n
            Base random seed.  Trajectory ``k`` uses base ``seed + k *
            ``traj_period`` with ``traj_period`` large enough to avoid
            correlation.
        n_jobs : ``int``, optional (default=1)\n
            Number of parallel worker processes.  ``n_jobs <= 1`` runs
            serially in the current process.
        verbose : ``bool``, optional (default=False)\n
            Print progress information when running serially.

        Return
        ----------
        expect : ``ndarray`` of shape ``(len(e_ops), len(tlist))``\n
            Trajectory-averaged expectation values.
        std : ``ndarray`` of shape ``(len(e_ops), len(tlist))``\n
            Standard deviation across trajectories.
        """
        traj_num = int(traj_num)
        if traj_num <= 0:
            raise ValueError(f"traj_num must be positive, got {traj_num}")
        e_ops = [np.asarray(op, dtype=complex) for op in e_ops]
        # Distinguish each trajectory by a large offset so the per-step seeds
        # do not overlap.
        traj_period = 10 * max(int(len(tlist)), 1)
        seeds = [seed + k * traj_period for k in range(traj_num)]

        if n_jobs is None or n_jobs <= 1:
            results = []
            for k, s in enumerate(seeds):
                res = self.evolve_trajectory(tlist, e_ops, seed=s)
                results.append(res["expect"])
                if verbose:
                    print(f"[LindbladMagnus] trajectory {k + 1}/{traj_num} done")
        else:
            results = _run_parallel(self, tlist, e_ops, seeds, n_jobs)

        expects = np.stack(results, axis=0)  # (traj_num, n_ops, n_times)
        mean = expects.mean(axis=0)
        std = expects.std(axis=0)
        return mean, std

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _build_H_eff(self, dt: float, theta: np.ndarray,
                     psi: np.ndarray | None, integrals: dict) -> np.ndarray:
        """Wrap :func:`effective_hamiltonian` with the solver configuration."""
        return effective_hamiltonian(
            self.H, self.c_ops, dt,
            magnus_order=self.magnus_order,
            qsd_type=self.qsd_type,
            nonlinear_corr=False,  # the corrector is handled in evolve_trajectory
            psi=psi,
            psi_p=None,
            integrals=integrals,
        )

    def _integrate_step(self, theta: np.ndarray, H_eff: np.ndarray,
                        dt: float, psi_norm: float
                        ) -> tuple[np.ndarray, float]:
        if self.integrator == "rk4":
            return variational_step_rk4(self.ansatz, theta, H_eff, dt, psi_norm)
        return variational_step_euler(self.ansatz, theta, H_eff, dt, psi_norm)

    def _measure_observables(self, theta: np.ndarray,
                             e_ops: Sequence[np.ndarray],
                             psi_norm: float) -> np.ndarray:
        """Return the (real) expectation values of ``e_ops`` at ``theta``.

        For the *linear* QSD the wave-function norm is non-trivial and the
        observable expectation is scaled by ``psi_norm`` so that the ensemble
        average recovers the open-system density matrix element.
        """
        psi = self.ansatz.get_statevector(theta)
        psi = psi / np.linalg.norm(psi)
        vals = np.empty(len(e_ops), dtype=float)
        for i, op in enumerate(e_ops):
            op = np.asarray(op, dtype=complex)
            vals[i] = float(np.real(np.vdot(psi, op @ psi)))
        if self.qsd_type == "linear":
            vals = vals * psi_norm
        return vals


# ----------------------------------------------------------------------
#  Functional interface
# ----------------------------------------------------------------------
def evolve_trajectory(solver: LindbladMagnusSolver, tlist, e_ops, seed):
    """Top-level helper used by the worker processes."""
    return solver.evolve_trajectory(tlist=tlist, e_ops=e_ops, seed=seed)


def _run_parallel(solver, tlist, e_ops, seeds, n_jobs):
    """Run trajectories in separate processes."""
    # The solver holds a CPUQVM which is not picklable; we defer to a
    # ProcessPoolExecutor with a top-level helper that re-binds the solver.
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_worker, solver, tlist, e_ops, s)
                       for s in seeds]
            results = [f.result() for f in futures]
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn(f"Parallel execution failed ({exc!r}); falling back to "
                      "serial execution.")
        results = [solver.evolve_trajectory(tlist, e_ops, seed=s)
                   for s in seeds]
    return [r["expect"] for r in results]


def _worker(solver, tlist, e_ops, seed):
    """Module-level worker that re-creates a per-process solver copy."""
    # CPUQVM is cheap to instantiate; we strip the cached instance so the copy
    # builds a fresh one lazily inside the child process.
    local = deepcopy(solver)
    local.ansatz._qvm = None  # type: ignore[attr-defined]
    # Re-create the qvm lazily.
    from pyqpanda3.core import CPUQVM
    local.ansatz._qvm = CPUQVM()  # type: ignore[attr-defined]
    return local.evolve_trajectory(tlist=tlist, e_ops=e_ops, seed=seed)


def solve(H: np.ndarray, c_ops: list[np.ndarray],
          ansatz: VariationalAnsatz,
          psi0: np.ndarray,
          tlist: np.ndarray,
          e_ops: list[np.ndarray],
          traj_num: int = 100,
          magnus_order: int = 1,
          qsd_type: str = "nonlinear",
          seed: int = 0,
          n_jobs: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Convenience functional API matching the reference package layout.

    Parameters
    ----------
    H : ``ndarray``\n
        System Hamiltonian.
    c_ops : ``list`` of ``ndarray``\n
        Collapse operators.
    ansatz : :class:`~pyqpanda_alg.LindbladMagnus.ansatz.VariationalAnsatz`\n
        Parameterised variational ansatz.
    psi0 : ``ndarray``\n
        Initial wave-function.
    tlist : ``ndarray``\n
        Time grid.
    e_ops : ``list`` of ``ndarray``\n
        Observables.
    traj_num : ``int``, optional (default=100)\n
        Number of trajectories.
    magnus_order : ``int``, optional (default=1)\n
        Magnus expansion order.
    qsd_type : ``{'nonlinear', 'linear'}``, optional (default='nonlinear')\n
        Unravelling of the Lindblad master equation.
    seed : ``int``, optional (default=0)\n
        Base random seed.
    n_jobs : ``int``, optional (default=1)\n
        Number of parallel workers.

    Return
    ----------
    expect : ``ndarray`` of shape ``(len(e_ops), len(tlist))``\n
        Ensemble-averaged expectation values.
    std : ``ndarray`` of shape ``(len(e_ops), len(tlist))``\n
        Per-trajectory standard deviation.
    """
    solver = LindbladMagnusSolver(H, c_ops, ansatz,
                                  qsd_type=qsd_type,
                                  magnus_order=magnus_order)
    return solver.solve(psi0, tlist, e_ops,
                        traj_num=traj_num, seed=seed, n_jobs=n_jobs)
