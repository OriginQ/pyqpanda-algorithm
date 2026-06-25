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

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .ansatz import VariationalAnsatz
from .magnus import effective_hamiltonian, sample_wiener_integrals
from .variational import (mclachlan_system, variational_step_euler,
                          variational_step_rk4)

__all__ = ["LindbladMagnusSolver", "LindbladResult", "solve"]

_LOGGER = logging.getLogger("pyqpanda_alg.LindbladMagnus")


def _have_tqdm() -> bool:
    """Return ``True`` if the optional :mod:`tqdm` dependency is available."""
    try:
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------
#  Result container
# ----------------------------------------------------------------------
@dataclass
class LindbladResult:
    """Container for the output of :meth:`LindbladMagnusSolver.solve`.

    Attributes
    ----------
    times : ``ndarray`` of shape ``(n_times,)``\n
        Time grid used by the simulation.
    expect : ``ndarray`` of shape ``(n_ops, n_times,)``\n
        Trajectory-averaged expectation values of the requested observables.
    std : ``ndarray`` of shape ``(n_ops, n_times,)``\n
        Per-trajectory standard deviation of every observable (a measure of
        the Monte-Carlo noise).
    norms : ``ndarray`` of shape ``(n_times,)``\n
        Mean of the auxiliary norm :math:`N` tracked by the variational
        step.  For the *linear* QSD this is the physical wave-function
        squared norm and decays from ``1.0``.  For the *nonlinear* QSD the
        ansatz state is always renormalised, so the reported value is a
        diagnostic of the non-Hermiticity of the effective Hamiltonian
        rather than the physical norm.
    traj_num : ``int``\n
        Number of trajectories that were averaged.
    seeds : ``list`` of ``int``\n
        Random seeds used for each trajectory (for reproducibility).
    solver_info : ``dict``\n
        Read-only copy of the solver configuration (Hamiltonian shape,
        Magnus order, QSD type, integrator, ...).
    """

    times: np.ndarray
    expect: np.ndarray
    std: np.ndarray
    norms: np.ndarray = field(default_factory=lambda: np.empty(0))
    traj_num: int = 0
    seeds: list = field(default_factory=list)
    solver_info: dict = field(default_factory=dict)

    @property
    def n_ops(self) -> int:
        """Number of observables."""
        return self.expect.shape[0]

    @property
    def n_times(self) -> int:
        """Number of time-grid points."""
        return self.expect.shape[1]

    def __repr__(self) -> str:
        return (f"LindbladResult(n_ops={self.n_ops}, "
                f"n_times={self.n_times}, traj_num={self.traj_num})")


class LindbladMagnusSolver:
    """Variational Lindblad solver based on the stochastic Magnus expansion.

    The solver describes the unravelling (``qsd_type``), the integration scheme
    (``magnus_order``) and the variational manifold (``ansatz``) once and for
    all.  Each call to :meth:`solve` then runs an independent ensemble of
    trajectories and returns the averaged expectation values wrapped in a
    :class:`LindbladResult`.

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
    eps : ``float``, optional (default=1e-12)\n
        Tikhonov regularisation added to the McLachlan metric when solving the
        linear system for ``dtheta/dt``.  Increase this value if the variational
        dynamics becomes unstable.

    Examples
    --------
    >>> import numpy as np
    >>> from pyqpanda_alg.LindbladMagnus import (LindbladMagnusSolver,
    ...                                          tfim_model, HardwareEfficientAnsatz)
    >>> H, c_ops, e_ops, psi0, _ = tfim_model()
    >>> ans = HardwareEfficientAnsatz(n_qubits=2, layers=2, init_state=psi0)
    >>> solver = LindbladMagnusSolver(H, c_ops, ans, magnus_order=1)
    >>> times = np.linspace(0, 1, 6)
    >>> result = solver.solve(psi0, times, e_ops, traj_num=4, seed=42)
    >>> result.expect.shape
    (3, 6)
    """

    def __init__(self, H: np.ndarray, c_ops: Sequence[np.ndarray],
                 ansatz: VariationalAnsatz,
                 qsd_type: str = "nonlinear",
                 magnus_order: int = 1,
                 nonlinear_corr: bool = False,
                 integrator: str = "rk4",
                 init_params: np.ndarray | None = None,
                 eps: float = 1e-12):
        self.H = np.asarray(H, dtype=complex)
        if self.H.ndim != 2 or self.H.shape[0] != self.H.shape[1]:
            raise ValueError(
                f"H must be a square matrix, got shape {self.H.shape}")
        expected_dim = 1 << ansatz.n_qubits
        if self.H.shape[0] != expected_dim:
            raise ValueError(
                f"H has dimension {self.H.shape[0]} but the ansatz uses "
                f"{ansatz.n_qubits} qubits (dimension {expected_dim})")
        self.c_ops = [np.asarray(op, dtype=complex) for op in c_ops]
        for i, op in enumerate(self.c_ops):
            if op.shape != self.H.shape:
                raise ValueError(
                    f"c_ops[{i}] has shape {op.shape} but must match H "
                    f"shape {self.H.shape}")
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
        if self.nonlinear_corr and qsd_type != "nonlinear":
            _LOGGER.warning(
                "nonlinear_corr=True has no effect with qsd_type=%r; the "
                "predictor-corrector only applies to the nonlinear QSD.",
                qsd_type)
        if integrator not in ("euler", "rk4"):
            raise ValueError(
                f"integrator must be 'euler' or 'rk4', got {integrator!r}")
        self.integrator = integrator
        self.eps = float(eps)
        self.init_params = (np.zeros(ansatz.n_parameters, dtype=float)
                            if init_params is None
                            else np.asarray(init_params, dtype=float).reshape(-1))
        if self.init_params.size != ansatz.n_parameters:
            raise ValueError(
                f"init_params has length {self.init_params.size} but the "
                f"ansatz expects {ansatz.n_parameters} parameters")

    def __repr__(self) -> str:
        return (f"LindbladMagnusSolver(H_dim={self.H.shape[0]}, "
                f"n_c_ops={len(self.c_ops)}, "
                f"qsd_type={self.qsd_type!r}, "
                f"magnus_order={self.magnus_order}, "
                f"integrator={self.integrator!r}, "
                f"n_params={self.ansatz.n_parameters})")

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
        if len(tlist) < 2:
            raise ValueError(f"tlist must have >= 2 points, got {len(tlist)}")
        n_steps = len(tlist) - 1
        ansatz = self.ansatz
        # Accept either raw arrays or pre-converted ndarrays; the conversion
        # is idempotent so it is safe when called from ``solve()`` which has
        # already validated the shapes.
        e_ops_arr = [np.asarray(op, dtype=complex) for op in e_ops]

        theta = self.init_params.copy()
        psi_norm = 1.0
        expect = np.zeros((len(e_ops_arr), len(tlist)), dtype=float)
        param_traj = np.zeros((len(tlist), ansatz.n_parameters), dtype=float)
        norm_traj = np.zeros(len(tlist), dtype=float)

        # Initial measurement.
        expect[:, 0] = self._measure_observables(theta, e_ops_arr, psi_norm)
        param_traj[0] = theta
        norm_traj[0] = psi_norm

        for i in range(n_steps):
            dt = float(tlist[i + 1] - tlist[i])
            if dt <= 0:
                raise ValueError(
                    f"tlist must be strictly increasing; got "
                    f"tlist[{i+1}]={tlist[i+1]} <= tlist[{i}]={tlist[i]}")
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
            # predictor H_eff to obtain a predicted state, then *average* the
            # drift expectations of the predictor and the predicted state and
            # rebuild H_eff.  This matches the reference implementation of
            # the paper, where the corrected drift uses
            # ``Re(<L>_psi + <L>_psi_p)`` (equivalent to averaging through
            # the ``2 Re(<L>)`` prefactor of the nonlinear QSD drift).
            if self.nonlinear_corr and self.qsd_type == "nonlinear":
                theta_try, _ = variational_step_euler(
                    ansatz, theta, H_eff, dt, psi_norm, eps=self.eps)
                psi_corr = ansatz.get_statevector(theta_try)
                psi_corr = psi_corr / np.linalg.norm(psi_corr)
                H_eff = effective_hamiltonian(
                    self.H, self.c_ops, dt,
                    magnus_order=self.magnus_order,
                    qsd_type=self.qsd_type,
                    nonlinear_corr=True,
                    psi=psi_pred,
                    psi_p=psi_corr,
                    integrals=integrals,
                )

            theta, psi_norm = self._integrate_step(theta, H_eff, dt, psi_norm)

            expect[:, i + 1] = self._measure_observables(theta, e_ops_arr, psi_norm)
            param_traj[i + 1] = theta
            norm_traj[i + 1] = psi_norm

        result: dict = {
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
              verbose: bool | str = False) -> LindbladResult:
        """Run an ensemble of trajectories and return averaged observables.

        Parameters
        ----------
        psi0 : ``ndarray``\n
            Initial wave-function.  When possible the solver checks that the
            ansatz reproduces ``psi0`` at ``init_params`` and warns otherwise.
        tlist : ``ndarray``\n
            Strictly increasing time grid.
        e_ops : ``list`` of ``ndarray``\n
            Observables whose expectation values are returned.
        traj_num : ``int``, optional (default=100)\n
            Number of independent Lindblad trajectories.
        seed : ``int``, optional (default=0)\n
            Base random seed.  Trajectory ``k`` uses base
            ``seed + k * traj_period`` with ``traj_period`` large enough to
            avoid correlation.
        n_jobs : ``int``, optional (default=1)\n
            Number of parallel worker processes.  ``n_jobs <= 1`` runs
            serially in the current process.
        verbose : ``bool`` or ``str``, optional (default=False)\n
            If ``True``, log a message after each trajectory.  If ``"tqdm"``,
            display a :mod:`tqdm` progress bar (requires ``tqdm`` installed).
            Use ``False`` for silent runs.

        Return
        ----------
        result : :class:`LindbladResult`\n
            Container with the ensemble-averaged expectation values,
            per-trajectory standard deviations, mean norms and metadata.
        """
        traj_num = int(traj_num)
        if traj_num <= 0:
            raise ValueError(f"traj_num must be positive, got {traj_num}")
        tlist = np.asarray(tlist, dtype=float).reshape(-1)
        if len(tlist) < 2:
            raise ValueError(
                f"tlist must have at least 2 points, got {len(tlist)}")
        if np.any(np.diff(tlist) <= 0):
            raise ValueError("tlist must be strictly increasing")
        e_ops_arr = [np.asarray(op, dtype=complex) for op in e_ops]
        for i, op in enumerate(e_ops_arr):
            if op.shape != self.H.shape:
                raise ValueError(
                    f"e_ops[{i}] has shape {op.shape} but must match H "
                    f"shape {self.H.shape}")

        # Sanity check: ansatz(init_params) should reproduce psi0.
        self._validate_initial_state(psi0)

        # Distinguish each trajectory by a large offset so the per-step seeds
        # do not overlap.
        traj_period = 10 * max(int(len(tlist)), 1)
        seeds = [seed + k * traj_period for k in range(traj_num)]

        if verbose == "tqdm" and not _have_tqdm():
            _LOGGER.warning("verbose='tqdm' requested but tqdm is not "
                            "installed; falling back to verbose=False.")
            verbose = False

        if n_jobs is None or n_jobs <= 1:
            results = list(self._iter_trajectories_serial(
                tlist, e_ops_arr, seeds, verbose))
        else:
            results = _run_parallel(self, tlist, e_ops_arr, seeds,
                                    n_jobs, verbose)

        expects = np.stack([r["expect"] for r in results], axis=0)
        norms = np.stack([r["norm"] for r in results], axis=0)
        mean = expects.mean(axis=0)
        std = expects.std(axis=0)
        mean_norm = norms.mean(axis=0)

        return LindbladResult(
            times=tlist,
            expect=mean,
            std=std,
            norms=mean_norm,
            traj_num=traj_num,
            seeds=seeds,
            solver_info={
                "qsd_type": self.qsd_type,
                "magnus_order": self.magnus_order,
                "integrator": self.integrator,
                "nonlinear_corr": self.nonlinear_corr,
                "n_c_ops": len(self.c_ops),
                "n_params": self.ansatz.n_parameters,
                "eps": self.eps,
            },
        )

    def _iter_trajectories_serial(self, tlist, e_ops, seeds, verbose):
        """Yield single-trajectory results, optionally reporting progress."""
        total = len(seeds)
        iterator = range(total)
        if verbose == "tqdm":
            import tqdm
            iterator = tqdm.tqdm(range(total), desc="trajectories",
                                 unit="traj")
        elif verbose:
            iterator = _LoggingIterator(range(total), total, _LOGGER)
        for k in iterator:
            yield self.evolve_trajectory(tlist, e_ops, seed=seeds[k])

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _validate_initial_state(self, psi0: np.ndarray) -> None:
        """Warn if the ansatz at ``init_params`` does not reproduce ``psi0``.

        The check is best-effort: if anything goes wrong (e.g. the user
        supplied a non-basis ``init_state`` that the ansatz cannot prepare),
        the warning is skipped but a debug-level message is emitted so the
        issue can still be diagnosed through ``logging.DEBUG``.
        """
        try:
            psi_ansatz = self.ansatz.get_statevector(self.init_params)
            psi_ansatz = psi_ansatz / np.linalg.norm(psi_ansatz)
            psi0_arr = np.asarray(psi0, dtype=complex).reshape(-1)
            psi0_norm = psi0_arr / np.linalg.norm(psi0_arr)
            if psi0_norm.size != psi_ansatz.size:
                _LOGGER.debug("Skipping initial-state check: dimension mismatch"
                              " (%d vs %d).", psi0_norm.size, psi_ansatz.size)
                return
            overlap = abs(np.vdot(psi0_norm, psi_ansatz))
            if overlap < 0.95:
                _LOGGER.warning(
                    "ansatz(init_params) overlaps psi0 by only %.3f; the "
                    "simulation may not start from the requested state.",
                    overlap)
        except Exception as exc:  # pragma: no cover - defensive, just logging
            _LOGGER.debug("Initial-state check failed: %r", exc)

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
            return variational_step_rk4(self.ansatz, theta, H_eff, dt,
                                        psi_norm, eps=self.eps)
        return variational_step_euler(self.ansatz, theta, H_eff, dt,
                                      psi_norm, eps=self.eps)

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
            vals[i] = float(np.real(np.vdot(psi, op @ psi)))
        if self.qsd_type == "linear":
            vals = vals * psi_norm
        return vals


class _LoggingIterator:
    """Wrap an iterable and log progress every ``log_every`` items."""

    def __init__(self, iterable, total, logger, log_every: int = 1):
        self._iter = iter(iterable)
        self._total = total
        self._logger = logger
        self._log_every = log_every
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iter)
        self._i += 1
        if self._i % self._log_every == 0:
            self._logger.info("trajectory %d/%d", self._i, self._total)
        return item


# ----------------------------------------------------------------------
#  Parallel execution
# ----------------------------------------------------------------------
def _run_parallel(solver: LindbladMagnusSolver, tlist, e_ops, seeds,
                  n_jobs: int, verbose):
    """Run trajectories in separate processes.

    The solver and its ansatz are picklable thanks to
    :meth:`VariationalAnsatz.__getstate__` (which drops the non-picklable
    :class:`CPUQVM` and lets the child process re-create one).  Results are
    collected with :func:`concurrent.futures.as_completed` so that the
    progress bar advances smoothly regardless of per-trajectory wall time.
    """
    from concurrent.futures import as_completed
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_for_seed = {ex.submit(_worker, solver, tlist, e_ops, s): s
                               for s in seeds}
            # Preserve the submission order so that the returned list is
            # indexed by trajectory index (matches the serial path).
            result_by_seed: dict = {}
            pending = as_completed(future_for_seed)
            if verbose == "tqdm" and _have_tqdm():
                import tqdm
                pending = tqdm.tqdm(pending, total=len(future_for_seed),
                                     desc="trajectories", unit="traj")
            elif verbose:
                _LOGGER.info("Running %d trajectories across %d workers",
                             len(seeds), n_jobs)
            for fut in pending:
                result_by_seed[future_for_seed[fut]] = fut.result()
            return [result_by_seed[s] for s in seeds]
    except Exception as exc:
        _LOGGER.warning("Parallel execution failed (%r); falling back to "
                        "serial execution.", exc)
        return [solver.evolve_trajectory(tlist, e_ops, seed=s)
                for s in seeds]


def _worker(solver: LindbladMagnusSolver, tlist, e_ops, seed):
    """Module-level worker used by :func:`_run_parallel`."""
    return solver.evolve_trajectory(tlist=tlist, e_ops=e_ops, seed=seed)


# ----------------------------------------------------------------------
#  Functional API
# ----------------------------------------------------------------------
def solve(H: np.ndarray, c_ops: list[np.ndarray],
          ansatz: VariationalAnsatz,
          psi0: np.ndarray,
          tlist: np.ndarray,
          e_ops: list[np.ndarray],
          traj_num: int = 100,
          magnus_order: int = 1,
          qsd_type: str = "nonlinear",
          seed: int = 0,
          n_jobs: int = 1,
          **kwargs) -> LindbladResult:
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
    **kwargs\n
        Forwarded to :class:`LindbladMagnusSolver` (e.g. ``integrator``,
        ``eps``, ``nonlinear_corr``).

    Return
    ----------
    result : :class:`LindbladResult`\n
        Ensemble-averaged expectation values and per-trajectory statistics.
    """
    solver = LindbladMagnusSolver(H, c_ops, ansatz,
                                  qsd_type=qsd_type,
                                  magnus_order=magnus_order,
                                  **kwargs)
    return solver.solve(psi0, tlist, e_ops,
                        traj_num=traj_num, seed=seed, n_jobs=n_jobs)
