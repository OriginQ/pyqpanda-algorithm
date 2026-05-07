from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path
from typing import Any


def load_pyqpanda() -> Any:
    try:
        import pyqpanda as pq
    except Exception as exc:
        raise SystemExit("未安装 pyqpanda。请先运行：python -m pip install pyqpanda") from exc
    return pq


def insert_gate(prog: Any, gate: Any) -> Any:
    method = getattr(prog, "insert", None)
    if callable(method):
        return method(gate)
    return prog << gate


def append_rzz(pq: Any, prog: Any, qubits: list[Any], left: int, right: int, angle: float) -> None:
    insert_gate(prog, pq.CNOT(qubits[left], qubits[right]))
    insert_gate(prog, pq.RZ(qubits[right], angle))
    insert_gate(prog, pq.CNOT(qubits[left], qubits[right]))


def append_su2_color_probe(pq: Any, prog: Any, qubits: list[Any], left: int, right: int, theta: float) -> None:
    rx = getattr(pq, "RX", None)
    if not callable(rx):
        raise RuntimeError("pyqpanda 未找到 RX gate")

    pair_angle = theta / 2.0
    insert_gate(prog, pq.H(qubits[left]))
    insert_gate(prog, pq.H(qubits[right]))
    append_rzz(pq, prog, qubits, left, right, pair_angle)
    insert_gate(prog, pq.H(qubits[left]))
    insert_gate(prog, pq.H(qubits[right]))

    insert_gate(prog, rx(qubits[left], pi / 2.0))
    insert_gate(prog, rx(qubits[right], pi / 2.0))
    append_rzz(pq, prog, qubits, left, right, pair_angle)
    insert_gate(prog, rx(qubits[left], -pi / 2.0))
    insert_gate(prog, rx(qubits[right], -pi / 2.0))

    append_rzz(pq, prog, qubits, left, right, pair_angle)


def build_singlet_pair_program(pq: Any, qubits: list[Any], cbits: list[Any], theta: float, with_quench: bool) -> Any:
    prog = pq.QProg()
    for left in range(0, len(qubits), 2):
        right = left + 1
        insert_gate(prog, pq.X(qubits[right]))
        insert_gate(prog, pq.H(qubits[left]))
        insert_gate(prog, pq.CNOT(qubits[left], qubits[right]))
        insert_gate(prog, pq.Z(qubits[left]))

    if with_quench:
        center_left = len(qubits) // 2 - 1
        center_right = len(qubits) // 2
        append_su2_color_probe(pq, prog, qubits, center_left, center_right, theta)

    measure = getattr(pq, "Measure", None)
    if callable(measure):
        for qubit, cbit in zip(qubits, cbits):
            insert_gate(prog, measure(qubit, cbit))
        return prog

    measure_all = getattr(pq, "measure_all", None)
    if callable(measure_all):
        insert_gate(prog, measure_all(qubits, cbits))
        return prog

    raise RuntimeError("pyqpanda 未找到 Measure 或 measure_all")


def run_cpuqvm(num_qubits: int, shots: int, theta: float, with_quench: bool) -> dict[str, Any]:
    pq = load_pyqpanda()
    qvm = pq.CPUQVM()
    qvm.init_qvm()
    try:
        qubits = qvm.qAlloc_many(num_qubits)
        cbits = qvm.cAlloc_many(num_qubits)
        prog = build_singlet_pair_program(pq, qubits, cbits, theta, with_quench)
        result = qvm.run_with_configuration(prog, cbits, shots)
        return {
            "num_qubits": num_qubits,
            "shots": shots,
            "theta": theta,
            "with_quench": with_quench,
            "counts": dict(result),
        }
    finally:
        finalize = getattr(qvm, "finalize", None)
        if callable(finalize):
            finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--theta", type=float, default=0.35)
    parser.add_argument("--no-quench", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs_pyqpanda" / "counts.json"))
    args = parser.parse_args()

    if args.num_qubits < 2 or args.num_qubits % 2 != 0:
        raise SystemExit("num-qubits 必须是大于等于 2 的偶数。")
    if args.shots <= 0:
        raise SystemExit("shots 必须为正整数。")

    payload = run_cpuqvm(args.num_qubits, args.shots, args.theta, with_quench=not args.no_quench)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("pyQPanda CPUQVM SU(2) singlet-pair demo 完成")
    print(f"输出文件：{path}")
    print(f"非零 bitstring 数：{len(payload['counts'])}")


if __name__ == "__main__":
    main()
