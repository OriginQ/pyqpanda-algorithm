from __future__ import annotations


def singlet_pair_state_prep_qasm(num_qubits: int, measure: bool = True) -> str:
    if num_qubits % 2 != 0:
        raise ValueError("SU(2) singlet-pair ansatz 需要偶数 qubit")

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{num_qubits}];",
        f"creg c[{num_qubits}];",
    ]
    for left in range(0, num_qubits, 2):
        right = left + 1
        lines.append(f"x q[{right}];")
        lines.append(f"h q[{left}];")
        lines.append(f"cx q[{left}],q[{right}];")
        lines.append(f"z q[{left}];")
    if measure:
        for q in range(num_qubits):
            lines.append(f"measure q[{q}] -> c[{q}];")
    return "\n".join(lines) + "\n"


def su2_color_quench_probe_qasm(num_qubits: int) -> str:
    if num_qubits % 2 != 0:
        raise ValueError("SU(2) quench probe ansatz 需要偶数 qubit")

    lines = singlet_pair_state_prep_qasm(num_qubits, measure=False).splitlines()
    center_left = num_qubits // 2 - 1
    center_right = num_qubits // 2
    lines.append(f"cx q[{center_left}],q[{center_right}];")
    lines.append(f"rz(0.35) q[{center_right}];")
    lines.append(f"cx q[{center_left}],q[{center_right}];")
    for q in range(num_qubits):
        lines.append(f"measure q[{q}] -> c[{q}];")
    return "\n".join(lines) + "\n"
