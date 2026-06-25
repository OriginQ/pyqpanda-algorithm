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

"""Demo: Lindblad dynamics of the Fenna-Matthews-Olson (FMO) pigment-protein
complex, simulated through the stochastic Magnus expansion coupled to the
McLachlan variational principle.

The FMO complex is the canonical example of environment-assisted quantum
transport: an electronic excitation is initialised on bacteriochlorophyll site
1 and its population is transferred through the network of coupled sites
towards a reaction centre (the sink) while being lost to the electronic ground
state through radiative decay and dephased by the protein environment.

The example follows the parameter set of
    Huang et al., PRX Quantum 6, 040312 (2025),
and contrasts the variational Lindblad-Magnus trajectory ensemble against the
exact Liouvillian solution provided by :func:`pyqpanda_alg.LindbladMagnus.mesolve`.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pyqpanda_alg.LindbladMagnus import (HardwareEfficientAnsatz,
                                         LindbladMagnusSolver, fmo_model,
                                         mesolve)


def main(out_dir: str = "test_outputs", traj_num: int = 30,
         t_final: float = 200.0, n_steps: int = 41,
         magnus_order: int = 1, layers: int = 2,
         qsd_type: str = "nonlinear") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Model: 5-site FMO sub-network (padded to 3 qubits)                #
    # ------------------------------------------------------------------ #
    H, c_ops, e_ops, psi0, labels = fmo_model()
    print("=== Fenna-Matthews-Olson complex (5-site sub-network) ===")
    print(f"Hamiltonian dimension: {H.shape[0]} (3 qubits, padded)")
    print(f"Collapse operators:    {len(c_ops)} (3 dephasing + 3 decay + 1 sink)")
    print(f"Observables:           {len(e_ops)} ({', '.join(labels)})")
    print(f"Initial state:         |Site 1>")

    # ------------------------------------------------------------------ #
    #  Variational configuration                                         #
    # ------------------------------------------------------------------ #
    ansatz = HardwareEfficientAnsatz(n_qubits=3, layers=layers, init_state=psi0)
    print(f"Ansatz:                HardwareEfficientAnsatz, "
          f"{ansatz.n_parameters} parameters")

    times = np.linspace(0.0, t_final, n_steps)
    dt = times[1] - times[0]
    print(f"Time grid:             {n_steps - 1} steps, "
          f"dt = {dt:.3f}, T = {t_final:.1f}")

    # ------------------------------------------------------------------ #
    #  Exact reference (Liouvillian)                                     #
    # ------------------------------------------------------------------ #
    print("\nComputing exact Lindblad solution (Liouvillian) ...")
    exact = mesolve(H, psi0, times, c_ops, e_ops)

    # ------------------------------------------------------------------ #
    #  Variational Lindblad-Magnus simulation                            #
    # ------------------------------------------------------------------ #
    print(f"\nRunning variational Lindblad-Magnus solver, "
          f"order={magnus_order}, traj={traj_num}, {qsd_type} QSD ...")
    solver = LindbladMagnusSolver(H, c_ops, ansatz,
                                  qsd_type=qsd_type,
                                  magnus_order=magnus_order,
                                  integrator="rk4")
    result = solver.solve(psi0, times, e_ops,
                          traj_num=traj_num, seed=42, verbose=True)
    mean = result.expect
    std = result.std
    max_err = float(np.abs(mean - exact).max())
    print(f"  -> maximum error vs exact: {max_err:.4f}")
    print(f"  -> final populations:")
    for i, lab in enumerate(labels):
        print(f"     {lab:>8}: exact = {exact[i, -1]:.4f}, "
              f"sim = {mean[i, -1]:.4f} +- {std[i, -1]:.4f}")

    # ------------------------------------------------------------------ #
    #  Plotting                                                          #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = plt.cm.tab10.colors

    ax = axes[0]
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_xlim(times[0], times[-1])
    for i, lab in enumerate(labels):
        ax.plot(times, exact[i], "-", color=palette[i], label=f"Exact {lab}")
        ax.plot(times, mean[i], "--", color=palette[i], label=f"Sim {lab}")
    ax.set_title("FMO excitation transfer (Lindblad-Magnus variational)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute error")
    ax.set_xlim(times[0], times[-1])
    ax.set_yscale("log")
    for i, lab in enumerate(labels):
        err = np.abs(mean[i] - exact[i])
        # Guard against log(0)
        err_plot = np.where(err > 1e-12, err, 1e-12)
        ax.plot(times, err_plot, color=palette[i], label=lab)
    ax.set_title("Trajectory error vs exact Liouvillian")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "lindblad_magnus_fmo.png")
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FMO Lindblad-Magnus variational simulation demo")
    parser.add_argument("--traj", type=int, default=30,
                        help="Number of Lindblad trajectories (default 30).")
    parser.add_argument("--t-final", type=float, default=200.0,
                        help="Final simulation time (default 200).")
    parser.add_argument("--n-steps", type=int, default=41,
                        help="Number of time-grid points (default 41).")
    parser.add_argument("--magnus-order", type=int, default=1,
                        help="Order of the stochastic Magnus expansion "
                             "(0-4, default 1).")
    parser.add_argument("--layers", type=int, default=2,
                        help="Number of hardware-efficient ansatz layers "
                             "(default 2).")
    parser.add_argument("--qsd-type", choices=["nonlinear", "linear"],
                        default="nonlinear",
                        help="QSD unravelling (default nonlinear).")
    parser.add_argument("--out-dir", default="test_outputs",
                        help="Output directory for figures.")
    args = parser.parse_args()

    main(out_dir=args.out_dir, traj_num=args.traj, t_final=args.t_final,
         n_steps=args.n_steps, magnus_order=args.magnus_order,
         layers=args.layers, qsd_type=args.qsd_type)
