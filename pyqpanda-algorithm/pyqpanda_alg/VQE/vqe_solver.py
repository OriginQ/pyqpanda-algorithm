"""
VQE Solver — the main variational quantum eigensolver class.

Implements the hybrid quantum-classical optimization loop:
    1. Prepare trial state |ψ(θ)⟩ via parameterized ansatz
    2. Measure energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
    3. Update θ with classical optimizer to minimize E(θ)
    4. Repeat until convergence

Author: Bai
"""

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .hamiltonian import PauliHamiltonian
from .ansatz import hardware_efficient_ansatz, ry_linear_ansatz, param_count
from .measurement import compute_energy


class VQESolver:
    """
    Variational Quantum Eigensolver.

    Finds the minimum eigenvalue (ground state energy) of a Hamiltonian
    using parameterized quantum circuits and classical optimization.

    Parameters
        hamiltonian : PauliHamiltonian
            Target Hamiltonian expressed as sum of Pauli terms.
        ansatz : str or callable, default='hea'
            Ansatz type. Built-in options: 'hea' (Hardware Efficient),
            'ry_linear' (RY-only). Or pass a custom callable:
            func(qubits, params) -> QCircuit.
        num_layers : int, default=2
            Number of ansatz repetition layers.
        optimizer : str, default='COBYLA'
            Classical optimizer. Supports any scipy.optimize.minimize method:
            'COBYLA', 'Nelder-Mead', 'Powell', 'SLSQP', 'L-BFGS-B'.
        shots : int, default=1024
            Measurement shots per energy evaluation.
        tol : float, default=1e-6
            Convergence tolerance for the optimizer.
        maxiter : int, default=200
            Maximum number of optimizer iterations.

    Attributes
        energy_history : list[float]
            Energy value recorded at each optimizer iteration.
        optimal_params : ndarray or None
            Optimized parameter vector after calling run().
        ground_energy : float or None
            Final optimized energy value after calling run().

    Examples:
        .. code-block:: python

            from pyqpanda_alg.VQE import VQESolver, PauliHamiltonian

            # Build H2 molecule Hamiltonian
            H = PauliHamiltonian()
            H.add_term(-1.0523, "II")
            H.add_term( 0.3979, "IZ")
            H.add_term(-0.3979, "ZI")
            H.add_term(-0.0112, "ZZ")
            H.add_term( 0.1809, "XX")

            # Run VQE
            solver = VQESolver(H, ansatz='hea', num_layers=2)
            result = solver.run()

            print(f"VQE ground energy: {result['energy']:.6f}")
            print(f"Exact energy:      {H.exact_ground_energy():.6f}")
            print(f"Iterations:        {result['num_iterations']}")
    """

    def __init__(self, hamiltonian, ansatz='hea', num_layers=2,
                 optimizer='COBYLA', shots=1024, tol=1e-6, maxiter=200):

        if not isinstance(hamiltonian, PauliHamiltonian):
            raise TypeError("hamiltonian must be a PauliHamiltonian instance.")
        if hamiltonian.num_qubits == 0:
            raise ValueError("Hamiltonian has no terms. Add terms first.")

        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.num_qubits
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.shots = shots
        self.tol = tol
        self.maxiter = maxiter

        # resolve ansatz
        self._ansatz_name = ansatz if isinstance(ansatz, str) else 'custom'
        self._ansatz_fn = self._resolve_ansatz(ansatz)
        self._num_params = self._compute_num_params()

        # results (populated after run())
        self.energy_history = []
        self.optimal_params = None
        self.ground_energy = None

    def _resolve_ansatz(self, ansatz):
        """Map ansatz specification to a callable."""
        if callable(ansatz):
            return ansatz
        elif ansatz == 'hea':
            return lambda qubits, params: hardware_efficient_ansatz(
                qubits, params, self.num_layers
            )
        elif ansatz == 'ry_linear':
            return lambda qubits, params: ry_linear_ansatz(
                qubits, params, self.num_layers
            )
        else:
            raise ValueError(
                f"Unknown ansatz '{ansatz}'. Use 'hea', 'ry_linear', "
                f"or a custom callable."
            )

    def _compute_num_params(self):
        """Determine total parameter count based on ansatz type."""
        if self._ansatz_name == 'hea':
            return param_count(self.num_qubits, self.num_layers, 'hea')
        elif self._ansatz_name == 'ry_linear':
            return param_count(self.num_qubits, self.num_layers, 'ry_linear')
        else:
            # for custom ansatz, user must provide init_params in run()
            return None

    def _cost_function(self, params):
        """
        Objective function for the optimizer.
        Builds ansatz circuit with given params, then measures energy.
        """
        qubits = list(range(self.num_qubits))
        circuit = self._ansatz_fn(qubits, params)
        energy = compute_energy(circuit, qubits, self.hamiltonian, self.shots)
        self.energy_history.append(energy)
        return energy

    def run(self, init_params=None):
        """
        Execute the VQE optimization loop.

        Parameters
            init_params : array_like or None
                Initial parameter values. If None, uses random initialization
                in range [0, π]. Must be provided if using a custom ansatz
                with unknown parameter count.

        Returns
            dict with keys:
                'energy' : float — optimized ground state energy
                'optimal_params' : ndarray — best parameters found
                'num_iterations' : int — number of cost function evaluations
                'history' : list[float] — energy at each iteration
                'success' : bool — whether optimizer converged
        """
        # initialize parameters
        if init_params is not None:
            x0 = np.asarray(init_params, dtype=float)
        elif self._num_params is not None:
            x0 = np.random.uniform(0, np.pi, size=self._num_params)
        else:
            raise ValueError(
                "Custom ansatz requires init_params to be provided."
            )

        # reset history
        self.energy_history = []

        # run scipy optimizer
        result = scipy_minimize(
            self._cost_function,
            x0,
            method=self.optimizer,
            tol=self.tol,
            options={'maxiter': self.maxiter}
        )

        self.optimal_params = result.x
        self.ground_energy = result.fun

        return {
            'energy': result.fun,
            'optimal_params': result.x,
            'num_iterations': len(self.energy_history),
            'history': self.energy_history,
            'success': result.success,
        }

    def run_with_callback(self, init_params=None, callback=None):
        """
        Run VQE with a user-defined callback after each iteration.

        Parameters
            init_params : array_like or None
                Initial parameters (same as run()).
            callback : callable or None
                Called as callback(params) after each optimizer step.
                Useful for logging or visualization.

        Returns
            dict (same structure as run())
        """
        if init_params is not None:
            x0 = np.asarray(init_params, dtype=float)
        elif self._num_params is not None:
            x0 = np.random.uniform(0, np.pi, size=self._num_params)
        else:
            raise ValueError(
                "Custom ansatz requires init_params to be provided."
            )

        self.energy_history = []

        result = scipy_minimize(
            self._cost_function,
            x0,
            method=self.optimizer,
            tol=self.tol,
            options={'maxiter': self.maxiter},
            callback=callback,
        )

        self.optimal_params = result.x
        self.ground_energy = result.fun

        return {
            'energy': result.fun,
            'optimal_params': result.x,
            'num_iterations': len(self.energy_history),
            'history': self.energy_history,
            'success': result.success,
        }

    def __repr__(self):
        return (
            f"VQESolver(qubits={self.num_qubits}, ansatz='{self._ansatz_name}', "
            f"layers={self.num_layers}, optimizer='{self.optimizer}', "
            f"shots={self.shots})"
        )
