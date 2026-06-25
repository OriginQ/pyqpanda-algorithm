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

"""Demo: Lindblad dynamics of the transverse-field Ising model with amplitude
damping, simulated through the stochastic Magnus expansion coupled to the
McLachlan variational principle.

The example reproduces the qualitatively correct dynamics of the open TFIM
system studied in
    Huang et al., PRX Quantum 6, 040312 (2025),
and compares the variational Lindblad-Magnus trajectories with the exact
Liouvillian solution provided by :func:`pyqpanda_alg.LindbladMagnus.mesolve`.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pyqpanda_alg.LindbladMagnus import (HardwareEfficientAnsatz,
                                         LindbladMagnusSolver, mesolve,
                                         tfim_model)


def main(out_dir: str = "test_outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Model: TFIM with two amplitude-damping channels                   #
    # ------------------------------------------------------------------ #
    H, c_ops, e_ops, psi0, labels = tfim_model()
    print("=== Transverse-field Ising model with amplitude damping ===")
    print(f"Hamiltonian dimension: {H.shape[0]} (2 qubits)")
    print(f"Collapse operators:    {len(c_ops)}")
    print(f"Observables:           {len(e_ops)} ({', '.join(labels)})")
    print(f"Initial state:         |11>")

    # ------------------------------------------------------------------ #
    #  Variational configuration                                         #
    # ------------------------------------------------------------------ #
    # The hardware-efficient ansatz uses parameterised RX/RZ rotations on
    # every qubit and parameterised RZZ entanglers in a ring.  All
    # parameterised gates reduce to the identity at theta=0, so the ansatz
    # reproduces the initial state |11> when theta = 0.
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=2, init_state=psi0)
    print(f"Ansatz:                HardwareEfficientAnsatz, "
          f"{ansatz.n_parameters} parameters")

    # Time grid.  dt=0.05 is small enough to resolve the Ising dynamics
    # (period ~ 2*pi/|g| ~ 6) and the moderate damping rate g=0.1.
    times = np.linspace(0.0, 2.5, 51)
    traj_num = 30

    # ------------------------------------------------------------------ #
    #  Exact reference (Liouvillian)                                     #
    # ------------------------------------------------------------------ #
    print("\nComputing exact Lindblad solution (Liouvillian) ...")
    exact = mesolve(H, psi0, times, c_ops, e_ops)

    # ------------------------------------------------------------------ #
    #  Variational Lindblad-Magnus simulation                            #
    # ------------------------------------------------------------------ #
    fig_evo, ax_evo = plt.subplots(figsize=(8, 5))
    ax_evo.set_xlabel("Time")
    ax_evo.set_ylabel("Population")
    ax_evo.set_xlim(times[0], times[-1])

    palette = plt.cm.tab10.colors
    for i, lab in enumerate(labels):
        ax_evo.plot(times, exact[i], "-", color=palette[i],
                    label=f"Exact {lab}")

    for order in (1, 2):
        print(f"\nRunning variational Lindblad-Magnus solver, "
              f"order={order}, traj={traj_num} ...")
        solver = LindbladMagnusSolver(H, c_ops, ansatz,
                                      qsd_type="nonlinear",
                                      magnus_order=order,
                                      integrator="rk4")
        mean, std = solver.solve(psi0, times, e_ops,
                                 traj_num=traj_num, seed=42)
        max_err = float(np.abs(mean - exact).max())
        print(f"  -> maximum error vs exact: {max_err:.4f}")
        for i, lab in enumerate(labels):
            ax_evo.plot(times, mean[i], "--", color=palette[i],
                        label=f"M{order} {lab}")

    ax_evo.legend(loc="best", fontsize=8)
    ax_evo.set_title("TFIM with amplitude damping: Lindblad-Magnus "
                     "variational simulation")
    out_path = os.path.join(out_dir, "lindblad_magnus_tfim.png")
    fig_evo.tight_layout()
    fig_evo.savefig(out_path, dpi=200)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
