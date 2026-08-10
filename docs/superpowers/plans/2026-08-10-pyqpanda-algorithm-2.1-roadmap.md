# PyQPanda Algorithm 2.1 Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver pyqpanda-algorithm 2.1.0 with a reliable release baseline, a capability-based qpanda3-runtime execution layer, runtime paths for all public algorithms, and stable VQE, HHL, and Shor APIs.

**Architecture:** Work is split into seven independently reviewable plans. The execution layer is completed before algorithm migration; migrated algorithms then provide the reusable task and measurement patterns needed by VQE, HHL, and Shor. Release qualification runs only after every preceding plan is complete.

**Tech Stack:** Python 3, pyqpanda3, qpanda3-runtime, NumPy, SciPy, pytest, setuptools/PEP 621, Sphinx, GitHub Actions.

## Global Constraints

- Target version is exactly `2.1.0`.
- Existing valid CPU calls remain compatible and default to `LocalBackend`.
- `qpanda3-runtime` is an optional dependency exposed through `pyqpanda_alg[runtime]`.
- Runtime failures never silently fall back to CPU or classical substitute results.
- Algorithms never authenticate, read API keys, or select a device automatically.
- New `backend` and `execution_options` parameters are keyword-only.
- Standard CI is credential-free; FakeBackend and real QPU execution belong to RC qualification.
- Each implementation task uses test-first development and ends in an independently reviewable commit.

---

## Plan Order

1. `2026-08-10-release-baseline.md`
   - Produces trustworthy packaging, testing, CI, versioning, and repaired current APIs.
2. `2026-08-10-execution-layer.md`
   - Produces backend protocols, local/runtime adapters, task state machines, and checkpoint support.
3. `2026-08-10-existing-algorithms-runtime-migration.md`
   - Migrates all 13 current modules away from internal `CPUQVM` construction.
4. `2026-08-10-vqe.md`
   - Adds stable VQE using the estimator and variational-session capabilities.
5. `2026-08-10-hhl.md`
   - Adds stable HHL with explicit resource limits and optional reconstruction.
6. `2026-08-10-shor.md`
   - Adds stable small-scale Shor with generic order finding and bounded retries.
7. `2026-08-10-release-qualification.md`
   - Produces FakeBackend/QPU evidence and enforces the final release gate.

## Cross-Plan Interfaces

- Plan 1 owns package metadata, dependency groups, versioning, root pytest configuration, and CI.
- Plan 2 owns every symbol under `pyqpanda_alg.execution`; later plans must not create competing backend abstractions.
- Plan 3 establishes the migration pattern for current algorithms and must finish before the three new algorithms are declared stable.
- Plans 4-6 consume `ExecutionBackend`, `ExecutionOptions`, `BackendTask`, and `AlgorithmTask` exactly as defined by Plan 2.
- Plan 7 consumes the public APIs produced by Plans 3-6 and must not patch algorithm behavior merely to make qualification pass.

## Program Completion Check

- [ ] Run every plan's full verification command from a clean checkout.
- [ ] Confirm the 13 existing modules and VQE/HHL/Shor import from the built wheel.
- [ ] Confirm every executable algorithm has a LocalBackend test, runtime contract test, and real-QPU RC record.
- [ ] Confirm every circuit-only component transpiles against the selected RC device.
- [ ] Confirm version, tag, Changelog, documentation, and wheel metadata all say `2.1.0`.
- [ ] Archive the final sanitized qualification manifest as an immutable workflow/Release artifact for the exact tagged commit and wheel.
