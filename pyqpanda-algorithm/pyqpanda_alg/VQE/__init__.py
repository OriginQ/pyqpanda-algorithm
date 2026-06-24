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

'''
A Variational Quantum Eigensolver (VQE) algorithm set, based on pyqpanda3.

The **VQE** module implements the hybrid quantum-classical variational
eigensolver of Peruzzo et al. (2014), the canonical NISQ-era algorithm for
finding ground-state (and low-lying excited) energies of Hamiltonians

.. math::
    H\ket{\psi_0} = E_0 \ket{\psi_0}.

VQE parameterizes a trial state :math:`\ket{\psi(\vec\theta)} = U(\vec\theta)\ket{0}`
with a quantum circuit (the *ansatz*), measures the energy expectation value

.. math::
    E(\vec\theta) = \bra{\psi(\vec\theta)} H \ket{\psi(\vec\theta)},

and minimizes it with a classical optimizer. At the optimum, :math:`E(\vec\theta^*)`
approximates the ground-state energy :math:`E_0` and :math:`\ket{\psi(\vec\theta^*)}`
approximates the ground state.

Some key features included in the VQE module are:

- **Hamiltonian construction** (:mod:`hamiltonian`): Jordan-Wigner mapping of
  molecular electronic Hamiltonians, pre-built benchmarks (H2, LiH) and
  condensed-matter models (transverse-field Ising, Heisenberg).
- **Ansatz library** (:mod:`ansatz`): hardware-efficient, UCC-style and
  symmetry-preserving variational forms.
- **Core solver** (:mod:`vqe`): the :class:`~pyqpanda_alg.VQE.vqe.VQE` class with
  multiple optimizers, parameter-shift gradients, shot-based measurement and a
  built-in excited-state routine via Quantum Subspace Expansion (QSE).

Overall, the VQE module provides a standardized, end-to-end toolkit for
variational eigensolver applications in quantum chemistry and condensed-matter
physics on the QPanda3 framework.

References
    [1] PERUZZO A, MCCLEAN J, SHADBOLT P, et al. A variational eigenvalue solver
    on a photonic quantum processor[J]. Nature Communications, 2014, 5: 4213.
    DOI: 10.1038/ncomms5213.\n
    [2] MCCLEAN J R, ROMERO J, BABBUSH R, et al. The theory of variational
    hybrid quantum-classical algorithms[J]. New Journal of Physics, 2016,
    18(2): 023023. DOI: 10.1088/1367-2630/18/2/023023.
'''
from . import hamiltonian
from . import ansatz
from . import vqe


__all__ = ['hamiltonian', 'ansatz', 'vqe']
