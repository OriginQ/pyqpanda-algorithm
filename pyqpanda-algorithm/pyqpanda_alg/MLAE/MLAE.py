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

"""Maximum-likelihood quantum amplitude estimation.

This module implements the maximum-likelihood amplitude-estimation (MLAE)
family described by Suzuki et al. It avoids phase-estimation ancillas: the
state-preparation circuit is amplified with a schedule of Grover powers, the
target qubit is sampled, and a global likelihood fit recovers the amplitude.

The classical likelihood decoder is public so callers can fit counts produced
by hardware, cloud backends, or recorded experiments without using CPUQVM.
"""

from typing import Iterable, Sequence, Union

import numpy as np
from pyqpanda3.core import CPUQVM, QProg

from pyqpanda_alg.Grover import amp_operator
from .. plugin import *


class MLAE:
    """Maximum-Likelihood Amplitude Estimation.

    operator_in is the state-preparation circuit A(qubits). The probability of
    measuring 1 on res_index after A is the amplitude to estimate.

    Experiment power k has success probability sin^2((2*k + 1) * theta).
    The default geometric schedule is (0, 1, 2, 4, 8, 16).

    Reference:
    Y. Suzuki, S. Uno, R. Raymond, T. Tanaka, T. Onodera and N. Yamamoto,
    "Amplitude Estimation without Phase Estimation", Quantum Information
    Processing 19, 75 (2020).
    """

    def __init__(
        self,
        operator_in=None,
        qnumber: int = 0,
        res_index: int = -1,
        powers: Sequence[int] = None,
        shots: Union[int, Sequence[int]] = 256,
        grid_size: int = 4097,
        refine_steps: int = 3,
    ):
        if operator_in is None or not callable(operator_in):
            raise ValueError("operator_in must be a callable state-preparation circuit")
        if not isinstance(qnumber, int) or qnumber <= 0:
            raise ValueError("qnumber must be a positive integer")

        if res_index < 0:
            res_index += qnumber
        if res_index < 0 or res_index >= qnumber:
            raise ValueError("res_index is outside the qubit register")

        if powers is None:
            powers = (0, 1, 2, 4, 8, 16)
        powers = tuple(int(k) for k in powers)
        if not powers or any(k < 0 for k in powers):
            raise ValueError("powers must contain non-negative integers")
        if len(set(powers)) != len(powers):
            raise ValueError("powers must not contain duplicates")

        if isinstance(shots, (int, np.integer)):
            if int(shots) <= 0:
                raise ValueError("shots must be positive")
            shot_schedule = (int(shots),) * len(powers)
        else:
            shot_schedule = tuple(int(n) for n in shots)
            if len(shot_schedule) != len(powers) or any(n <= 0 for n in shot_schedule):
                raise ValueError("shots must provide one positive count per power")

        if not isinstance(grid_size, int) or grid_size < 257:
            raise ValueError("grid_size must be an integer >= 257")
        if not isinstance(refine_steps, int) or refine_steps < 0:
            raise ValueError("refine_steps must be a non-negative integer")

        self.operator = operator_in
        self.qnumber = qnumber
        self.res_index = res_index
        self.powers = powers
        self.shots = shot_schedule
        self.grid_size = grid_size
        self.refine_steps = refine_steps
        self.machine = CPUQVM()
        self.last_result = None

    @staticmethod
    def _validate_observations(successes: Iterable[int], shots: Iterable[int]):
        success_array = np.asarray(tuple(successes), dtype=np.int64)
        shot_array = np.asarray(tuple(shots), dtype=np.int64)

        if success_array.ndim != 1 or shot_array.ndim != 1:
            raise ValueError("successes and shots must be one-dimensional")
        if len(success_array) == 0 or len(success_array) != len(shot_array):
            raise ValueError("successes and shots must have the same non-zero length")
        if np.any(shot_array <= 0):
            raise ValueError("all shot counts must be positive")
        if np.any(success_array < 0) or np.any(success_array > shot_array):
            raise ValueError("every success count must lie between 0 and its shot count")
        return success_array, shot_array

    @staticmethod
    def _likelihood_grid(theta, powers, successes, shots):
        theta = np.asarray(theta, dtype=float)
        multipliers = (2 * np.asarray(powers, dtype=np.int64) + 1)[:, None]
        probabilities = np.sin(multipliers * theta[None, :]) ** 2
        tiny = np.finfo(float).eps
        probabilities = np.clip(probabilities, tiny, 1.0 - tiny)

        return np.sum(
            successes[:, None] * np.log(probabilities)
            + (shots - successes)[:, None] * np.log1p(-probabilities),
            axis=0,
        )

    @classmethod
    def estimate_from_counts(
        cls,
        successes: Sequence[int],
        shots: Union[int, Sequence[int]],
        powers: Sequence[int] = None,
        grid_size: int = 4097,
        refine_steps: int = 3,
        return_details: bool = False,
    ):
        """Fit an amplitude from externally produced success counts."""
        if powers is None:
            powers = (0, 1, 2, 4, 8, 16)
        powers = tuple(int(k) for k in powers)
        if not powers or any(k < 0 for k in powers):
            raise ValueError("powers must contain non-negative integers")

        if isinstance(shots, (int, np.integer)):
            shot_schedule = (int(shots),) * len(powers)
        else:
            shot_schedule = tuple(int(n) for n in shots)

        success_array, shot_array = cls._validate_observations(
            successes, shot_schedule
        )
        if len(success_array) != len(powers):
            raise ValueError("powers must align with successes and shots")
        if not isinstance(grid_size, int) or grid_size < 257:
            raise ValueError("grid_size must be an integer >= 257")
        if not isinstance(refine_steps, int) or refine_steps < 0:
            raise ValueError("refine_steps must be a non-negative integer")

        lower = 0.0
        upper = np.pi / 2.0
        best_theta = 0.0
        best_log_likelihood = -np.inf

        for _ in range(refine_steps + 1):
            theta_grid = np.linspace(lower, upper, grid_size)
            log_likelihood = cls._likelihood_grid(
                theta_grid, powers, success_array, shot_array
            )
            best_index = int(np.argmax(log_likelihood))
            best_theta = float(theta_grid[best_index])
            best_log_likelihood = float(log_likelihood[best_index])

            if best_index == 0:
                left_index = 0
                right_index = min(2, grid_size - 1)
            elif best_index == grid_size - 1:
                left_index = max(0, grid_size - 3)
                right_index = grid_size - 1
            else:
                left_index = best_index - 1
                right_index = best_index + 1
            lower = float(theta_grid[left_index])
            upper = float(theta_grid[right_index])

        amplitude = float(np.sin(best_theta) ** 2)
        details = {
            "amplitude": amplitude,
            "theta": best_theta,
            "log_likelihood": best_log_likelihood,
            "powers": powers,
            "shots": tuple(int(n) for n in shot_array),
            "successes": tuple(int(n) for n in success_array),
            "oracle_queries": int(
                np.sum((2 * np.asarray(powers) + 1) * shot_array)
            ),
        }
        return details if return_details else amplitude

    def _sample_power(self, power: int, shots: int) -> int:
        qlist = QProg(self.qnumber).qubits()
        clist = QProg(self.qnumber).cbits()

        target = qlist[self.res_index]
        amplification = amp_operator(
            q_input=qlist,
            q_flip=[target],
            q_zero=qlist,
            in_operator=self.operator,
        )

        prog = QProg()
        prog << self.operator(qlist)
        for _ in range(power):
            prog << amplification
        prog << measure_all([target], [clist[self.res_index]])

        self.machine.run(prog, shots)
        counts = self.machine.result().get_counts()
        return int(counts.get("1", 0))

    def run(self, return_details: bool = False):
        """Execute the MLAE schedule on CPUQVM and return the fitted amplitude."""
        successes = [
            self._sample_power(power, shots)
            for power, shots in zip(self.powers, self.shots)
        ]
        result = self.estimate_from_counts(
            successes=successes,
            shots=self.shots,
            powers=self.powers,
            grid_size=self.grid_size,
            refine_steps=self.refine_steps,
            return_details=True,
        )
        self.last_result = result
        return result if return_details else result["amplitude"]
