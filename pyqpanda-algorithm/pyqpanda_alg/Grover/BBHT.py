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

"""Boyer-Brassard-Hoyer-Tapp search for an unknown number of marked states."""

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
from pyqpanda3.core import CPUQVM, QProg

from ..plugin import measure_all
from .Grover_core import Grover, mark_data_reflection


class BBHTResult(NamedTuple):
    """Result returned by BBHTSearch.run."""

    found: bool
    bitstring: Optional[str]
    attempts: int
    grover_iterations: int


class BBHTSearch:
    """Randomized Grover search when the number of marked states is unknown.

    BBHT removes the need to know the number of solutions in advance. Each
    attempt samples a Grover iteration count uniformly from an expanding
    interval and verifies the measured candidate classically. The expansion
    is capped at sqrt(N) as in the Boyer-Brassard-Hoyer-Tapp algorithm.

    Parameters
    ----------
    n_index:
        Number of qubits in the search register. The search-space size is
        N = 2 ** n_index.
    oracle_circuit:
        Callable f(qubits) returning a phase-marking circuit. qubits contains
        the search register followed by any optional workspace qubits
        requested in run().
    predicate:
        Callable f(bitstring) -> bool used to verify a measured candidate.
        BBHT needs this classical verifier because the number of marked states
        is deliberately unknown.
    mark_data:
        Convenience alternative to oracle_circuit + predicate for explicit
        marked bitstrings. The algorithm still does not use the number of
        marked states when choosing Grover iteration counts.
    in_operator:
        Optional initial-state circuit callable accepted by Grover.
        Defaults to Hadamards.
    zero_flip:
        Optional zero-reflection callable accepted by Grover.
    growth_rate:
        BBHT interval growth factor. It must satisfy 1 < growth_rate <= 4/3.
        The default 6/5 is the standard conservative choice.
    seed:
        Optional random seed for reproducible randomized schedules.

    References
    ----------
    M. Boyer, G. Brassard, P. Hoyer, A. Tapp,
    "Tight bounds on quantum searching", Fortschritte der Physik 46,
    493-505 (1998), https://arxiv.org/abs/quant-ph/9605034
    """

    def __init__(
        self,
        n_index: int,
        oracle_circuit: Optional[Callable] = None,
        predicate: Optional[Callable[[str], bool]] = None,
        mark_data: Optional[Sequence[str]] = None,
        in_operator: Optional[Callable] = None,
        zero_flip: Optional[Callable] = None,
        growth_rate: float = 6.0 / 5.0,
        seed: Optional[int] = None,
    ):
        if not isinstance(n_index, int) or n_index <= 0:
            raise ValueError("n_index must be a positive integer")
        if not (1.0 < growth_rate <= 4.0 / 3.0):
            raise ValueError("growth_rate must satisfy 1 < growth_rate <= 4/3")
        if mark_data is None and (oracle_circuit is None or predicate is None):
            raise ValueError(
                "provide mark_data, or provide both oracle_circuit and predicate"
            )
        if mark_data is not None and oracle_circuit is not None:
            raise ValueError("mark_data and oracle_circuit are alternative inputs")

        self.n_index = n_index
        self.in_operator = in_operator
        self.zero_flip = zero_flip
        self.growth_rate = float(growth_rate)
        self._rng = np.random.default_rng(seed)
        self._explicit_mark_data = None

        if mark_data is not None:
            if isinstance(mark_data, str):
                mark_data = [mark_data]
            normalized = tuple(mark_data)
            if not normalized:
                raise ValueError("mark_data must contain at least one bitstring")
            for value in normalized:
                if not isinstance(value, str):
                    raise TypeError("mark_data entries must be strings")
                if len(value) != n_index or any(bit not in "01" for bit in value):
                    raise ValueError(
                        "each mark_data entry must be a binary string of length n_index"
                    )

            marked = frozenset(normalized)
            self._explicit_mark_data = normalized

            def explicit_oracle(qubits):
                return mark_data_reflection(
                    qubits=qubits[: self.n_index],
                    mark_data=self._explicit_mark_data,
                )

            self.oracle_circuit = explicit_oracle
            self.predicate = (
                predicate if predicate is not None else lambda value: value in marked
            )
        else:
            self.oracle_circuit = oracle_circuit
            self.predicate = predicate

    def run(
        self,
        workspace_qubits: int = 0,
        max_attempts: Optional[int] = None,
        process_show: bool = False,
    ) -> BBHTResult:
        """Run randomized BBHT search.

        workspace_qubits is the number of auxiliary qubits made available to
        oracle_circuit. Search qubits are always the first n_index qubits.

        max_attempts bounds the randomized attempts before returning
        found=False. If omitted, a finite 9 * sqrt(N) budget is used so an
        empty marked set cannot loop forever.
        """
        if not isinstance(workspace_qubits, int) or workspace_qubits < 0:
            raise ValueError("workspace_qubits must be a non-negative integer")

        search_size = 2 ** self.n_index
        root_n = float(np.sqrt(search_size))
        if max_attempts is None:
            max_attempts = max(1, int(np.ceil(9.0 * root_n)))
        elif not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        machine = CPUQVM()
        m = 1.0
        total_iterations = 0

        for attempt in range(1, max_attempts + 1):
            upper = max(1, int(np.ceil(m)))
            iteration_count = int(self._rng.integers(low=0, high=upper))
            total_iterations += iteration_count

            all_qubits = QProg(self.n_index + workspace_qubits).qubits()
            search_qubits = all_qubits[: self.n_index]
            oracle_qubits = all_qubits if workspace_qubits else search_qubits

            grover = Grover(
                in_operator=self.in_operator,
                flip_operator=self.oracle_circuit,
                zero_flip=self.zero_flip,
            )
            circuit = grover.cir(
                q_input=search_qubits,
                q_flip=oracle_qubits,
                q_zero=search_qubits,
                iternum=iteration_count,
            )

            program = QProg()
            program << circuit
            program << measure_all(search_qubits, search_qubits)
            machine.run(program, shots=1)

            probabilities = machine.result().get_prob_dict()
            if not probabilities:
                raise RuntimeError("quantum backend returned no measurement result")
            candidate = max(probabilities, key=probabilities.get)

            if process_show:
                print(
                    "BBHT attempt",
                    attempt,
                    "m=",
                    round(m, 6),
                    "iterations=",
                    iteration_count,
                    "candidate=",
                    candidate,
                )

            if self.predicate(candidate):
                return BBHTResult(
                    found=True,
                    bitstring=candidate,
                    attempts=attempt,
                    grover_iterations=total_iterations,
                )

            m = min(self.growth_rate * m, root_n)

        return BBHTResult(
            found=False,
            bitstring=None,
            attempts=max_attempts,
            grover_iterations=total_iterations,
        )
