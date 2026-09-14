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
"""Demo: coherent-rotation theta^4 loss scaling for small QEC codes.

Run:
    python3 example/QECNoise/demo_theta4_scaling.py
"""

from pyqpanda_alg.QECNoise import run_theta4_scan, run_pauli_control


def main():
    print("10.29 theta^4 loss scaling law - pyQPanda-algorithm demo")
    print("=" * 72)

    print("Coherent single-qubit rotation noise (expected slope ~4):")
    for code in ("[[5,1,3]]", "[[7,1,3]]"):
        losses, slope = run_theta4_scan(code, trials=10, seed=42)
        print(f"  {code}: loss={['%.2e' % x for x in losses]}, log-log slope={slope:.2f}")

    print("-" * 72)
    print("Random-Pauli incoherent control (expected slope ~2 at low p):")
    losses, slope = run_pauli_control("[[7,1,3]]")
    print(f"  [[7,1,3]]: loss={['%.2e' % x for x in losses]}, log-log slope={slope:.2f}")


if __name__ == "__main__":
    main()
