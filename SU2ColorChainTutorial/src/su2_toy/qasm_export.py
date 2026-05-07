from __future__ import annotations

from math import pi


def _angle(value: float) -> str:
    return f"{value:.12g}"


def _append_rzz(lines: list[str], left: int, right: int, angle: float) -> None:
    lines.append(f"cx q[{left}],q[{right}];")
    lines.append(f"rz({_angle(angle)}) q[{right}];")
    lines.append(f"cx q[{left}],q[{right}];")


def _append_xx(lines: list[str], left: int, right: int, angle: float) -> None:
    lines.append(f"h q[{left}];")
    lines.append(f"h q[{right}];")
    _append_rzz(lines, left, right, angle)
    lines.append(f"h q[{left}];")
    lines.append(f"h q[{right}];")


def _append_yy(lines: list[str], left: int, right: int, angle: float) -> None:
    lines.append(f"rx({_angle(pi / 2.0)}) q[{left}];")
    lines.append(f"rx({_angle(pi / 2.0)}) q[{right}];")
    _append_rzz(lines, left, right, angle)
    lines.append(f"rx({_angle(-pi / 2.0)}) q[{left}];")
    lines.append(f"rx({_angle(-pi / 2.0)}) q[{right}];")


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


def su2_color_quench_probe_qasm(num_qubits: int, theta: float = 0.35) -> str:
    if num_qubits % 2 != 0:
        raise ValueError("SU(2) quench probe ansatz 需要偶数 qubit")

    lines = singlet_pair_state_prep_qasm(num_qubits, measure=False).splitlines()
    center_left = num_qubits // 2 - 1
    center_right = num_qubits // 2
    pair_angle = theta / 2.0
    _append_xx(lines, center_left, center_right, pair_angle)
    _append_yy(lines, center_left, center_right, pair_angle)
    _append_rzz(lines, center_left, center_right, pair_angle)
    for q in range(num_qubits):
        lines.append(f"measure q[{q}] -> c[{q}];")
    return "\n".join(lines) + "\n"
