# NISQ Hardware Benchmarking Suite

Submitted for the **2026 CCF Quantum Computing Programming Challenge "OriginQ Cup" — Open-Source Innovation Track** by team **Quantum Gap**.

## Why this

`pyqpanda-algorithm` ships strong application algorithms (Grover, QAOA, QSVM, QAE, ...) but **no hardware-characterisation tooling**. A user picking between Origin Wukong `WK_C180` and `WK_C180_2` for a real experiment currently has no in-library way to compare them. This package closes that gap with the standard primitives used across IBM, Google and Sandia benchmarking work — implemented natively in `pyqpanda3`, runnable on both local `CPUQVM` and the Origin QCloud (`full_amplitude` / `WK_C180`).

Every benchmark reports a single, comparable number so users can characterise a backend in one command:

```bash
python -m contest2.OriginQCup_QuantumGap.runner --backend cpu --qubits 3
ORIGINQC_API_KEY=... python -m contest2.OriginQCup_QuantumGap.runner --backend WK_C180 --qubits 3
```

## What's inside

| Benchmark              | What it measures                                                                 | Noiseless target |
|------------------------|----------------------------------------------------------------------------------|------------------|
| `bell_benchmark`       | 2-qubit entanglement fidelity proxy: P(\|00⟩) + P(\|11⟩)                         | 1.0              |
| `ghz_benchmark`        | n-qubit GHZ population fidelity proxy: P(\|0…0⟩) + P(\|1…1⟩)                     | 1.0              |
| `quantum_volume_probe` | Single-trial heavy-output frequency (Cross et al. 2019)                          | (1+ln 2)/2 ≈ 0.85; pass ≥ 2/3 |
| `mirror_circuit_probe` | Forward-then-inverse Clifford layers, P(\|0…0⟩) survival (Proctor et al. 2022)   | 1.0              |

A small `runner.py` ships:

- `cpuqvm_runner(prog, shots)` — local CPUQVM
- `qcloud_runner(backend_name, api_key=...)` — Origin QCloud (`WK_C180`, `WK_C180_2`, `full_amplitude`, `partial_amplitude`, `single_amplitude`)
- `run_suite(runner, n, shots, seed)` — runs all four benchmarks, returns a JSON-safe report

Both adapters expose the same minimal signature `runner(prog, shots) -> dict[str, int]`, so every benchmark is backend-agnostic — write once, run anywhere.

## Validation

### Local CPUQVM (noiseless reference)

```
=== Bell ===
  bell_fidelity_proxy = 1.0000   counts: {'00': 497, '11': 527}
=== GHZ (n=3) ===
  ghz_fidelity_proxy = 1.0000    counts: {'000': 507, '111': 517}
=== Mirror (n=3, d=3) ===
  survival_probability = 1.0000  counts: {'000': 1024}
=== QV probe (n=3) ===
  heavy_output_frequency  = 0.9561
  ideal_heavy_output_freq = 0.9618
  pass_threshold          = 0.6667
```

Every benchmark hits its noiseless target on the local CPUQVM. The QV probe's heavy-output frequency tracks the noiseless ideal to within shot noise.

### Origin QCloud end-to-end

The Bell state has been exercised against the Origin `full_amplitude` cloud backend during development — real Origin task ID `2316D46C0EBBFD5C384896D0028A44C3` returned the expected `[0.5, 0, 0, 0.5]` amplitude vector for `|00⟩, |01⟩, |10⟩, |11⟩`.

## Quick usage

```python
from contest2.OriginQCup_QuantumGap import bell_benchmark, ghz_benchmark
from contest2.OriginQCup_QuantumGap.runner import cpuqvm_runner, qcloud_runner

# Local simulator
bell = bell_benchmark(cpuqvm_runner, shots=1024)
print(f"bell fidelity proxy: {bell['bell_fidelity_proxy']:.4f}")

# Origin Wukong real hardware
runner = qcloud_runner("WK_C180")          # uses ORIGINQC_API_KEY
ghz = ghz_benchmark(runner, n=3, shots=1024)
print(f"GHZ fidelity proxy on WK_C180: {ghz['ghz_fidelity_proxy']:.4f}")
```

## Design notes

- **Backend-agnostic by construction.** Benchmarks never touch `CPUQVM` or `QCloudBackend` directly; they receive a `runner(prog, shots)` callable. New backends (e.g. future Wukong successors) plug in by writing a 10-line adapter.
- **One-circuit-per-trial QV probe**, not a full QV campaign. Computing a full QV score wants ≥ 100 trials and statistical significance testing; the goal here is a fast pre-flight check before you commit QPU time to that campaign.
- **Mirror inverse via gate-level reversal.** {H, X} are self-inverse; S⁻¹ = S³. We expand the inverse layer explicitly rather than relying on a `dagger()` helper that does not exist in this version of `pyqpanda3.core`.
- **No mock data.** Every reported number comes from a real `prog` executed by the supplied runner. If a benchmark cannot run on a backend (e.g. a missing gate), it raises — never silently returns a synthetic number.

## File layout

```
contest2/OriginQCup_QuantumGap/
├── __init__.py            # public exports
├── README.md              # this file
├── bell.py                # Bell-state benchmark
├── ghz.py                 # n-qubit GHZ benchmark
├── quantum_volume.py      # Cross 2019 QV probe
├── mirror_circuit.py      # Proctor 2022 mirror benchmark
└── runner.py              # CPUQVM + QCloud adapters + CLI
```

## License

Apache-2.0, matching the upstream `pyqpanda-algorithm` project.

## References

- A. W. Cross, L. S. Bishop, S. Sheldon, P. D. Nation, J. M. Gambetta. *Validating quantum computers using randomized model circuits.* Phys. Rev. A 100, 032328 (2019). arXiv:1811.12926
- T. Proctor, K. Rudinger, K. Young, E. Nielsen, R. Blume-Kohout. *Measuring the capabilities of quantum computers.* Nat. Phys. 18, 75–79 (2022).
- D. M. Greenberger, M. A. Horne, A. Zeilinger. *Going beyond Bell's theorem.* In: *Bell's Theorem, Quantum Theory and Conceptions of the Universe*, 1989.
