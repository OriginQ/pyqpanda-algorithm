'''
Lindblad dynamics via the stochastic Magnus expansion, based on pyqpanda3.

The **LindbladMagnus** module implements the variational quantum simulation of
open quantum systems described in

    J.-C. Huang, H.-E. Li, Y.-C. Wang, G.-Z. Zhang, J. Li, H.-S. Hu,
    "Towards Robust Variational Quantum Simulation of Lindblad Dynamics via
    Stochastic Magnus Expansion", PRX Quantum 6, 040312 (2025).

It provides:

* high-order stochastic Magnus integrators (Scheme I-IV) and an Euler-Maruyama
  scheme for the unravelling quantum state diffusion equation;
* a flexible parameterised variational ansatz built on top of :mod:`pyqpanda3`;
* the McLachlan variational principle with Euler and RK4 integrators;
* a high-level :class:`~pyqpanda_alg.LindbladMagnus.lindblad.LindbladMagnusSolver`
  with trajectory ensemble averaging;
* ready-to-use open quantum system models (FMO, TFIM with damping, radical pair
  model) and a Liouvillian-based exact Lindblad solver.

All quantum circuits are constructed and simulated through :mod:`pyqpanda3`; no
other quantum computing framework is required.

Examples
--------
>>> import numpy as np
>>> from pyqpanda_alg.LindbladMagnus import (LindbladMagnusSolver,
...                                          fmo_model, HardwareEfficientAnsatz)
>>> H, c_ops, e_ops, psi0, _ = fmo_model()
>>> ansatz = HardwareEfficientAnsatz(n_qubits=3, layers=2, init_state=psi0)
>>> solver = LindbladMagnusSolver(H, c_ops, ansatz, magnus_order=1)
>>> times = np.linspace(0, 50, 11)
>>> expect, std = solver.solve(psi0, times, e_ops, traj_num=4, seed=0)
'''

from .ansatz import HardwareEfficientAnsatz, VariationalAnsatz
from .lindblad import LindbladMagnusSolver, LindbladResult, solve
from .magnus import effective_hamiltonian, sample_wiener_integrals
from .models import fmo_model, liouvillian, mesolve, rpm_model, tfim_model
from .variational import (mclachlan_system, variational_step_euler,
                          variational_step_rk4)

__all__ = [
    "LindbladMagnusSolver",
    "LindbladResult",
    "solve",
    "effective_hamiltonian",
    "sample_wiener_integrals",
    "VariationalAnsatz",
    "HardwareEfficientAnsatz",
    "mclachlan_system",
    "variational_step_euler",
    "variational_step_rk4",
    "fmo_model",
    "tfim_model",
    "rpm_model",
    "mesolve",
    "liouvillian",
]
