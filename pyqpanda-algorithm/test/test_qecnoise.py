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
"""Tests for pyqpanda_alg.QECNoise."""

from pyqpanda_alg.QECNoise import run_theta4_scan, run_pauli_control


def test_theta4_slope_five_qubit():
    _, slope = run_theta4_scan("[[5,1,3]]", trials=10, seed=42)
    assert 3.5 < slope < 4.5


def test_theta4_slope_steane():
    _, slope = run_theta4_scan("[[7,1,3]]", trials=10, seed=42)
    assert 3.5 < slope < 4.5


def test_pauli_control_slope():
    _, slope = run_pauli_control("[[7,1,3]]")
    assert 1.0 < slope < 3.0
