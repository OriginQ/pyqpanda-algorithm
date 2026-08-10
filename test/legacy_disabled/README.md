# legacy_disabled

Fully commented-out test files moved out of active pytest discovery.

These files contain no executable lines (every line is a comment), so they
were silently passing as no-op tests while still being collected by pytest.
They are moved here under filenames that do not match the root `pytest.ini`
patterns (`python_files = test_*.py Test_*.py`) and are excluded from the
test inventory check (`test/meta/test_test_inventory.py` skips anything under
`legacy_disabled`).

Each file must be restored (un-commented and re-migrated to the runtime
execution layer) or deleted by the task in
[`docs/superpowers/plans/2026-08-10-existing-algorithms-runtime-migration.md`](../../docs/superpowers/plans/2026-08-10-existing-algorithms-runtime-migration.md)
(Plan 3 of the 2.1.0 release roadmap) listed below.

| File | Migration plan task |
|------|---------------------|
| `QAOA/complete_xy_mixer.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/default_circuits_linear_w_state.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/default_circuits_prepare_dicke_state.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/dstate_linear_w_state.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/dstate_prepare_dicke_state.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/init_d_state.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/parity_partition_xy_mixer.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/qaoa_QAOA_calculate_energy.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/qaoa_p_0.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/qaoa_p_1.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/qaoa_pauli_z_operator_to_circuit.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/qaoa_problem_to_z_operator.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAOA/xy_mixer.py` | Plan 3 Task 2 (QAOA estimator execution) |
| `QAlgBase/grover_amp_operator.py` | Plan 3 Task 3 (Grover sampling) |
| `QAlgBase/grover_cir.py` | Plan 3 Task 3 (Grover sampling) |
| `QAlgBase/grover_mark_data_reflection.py` | Plan 3 Task 3 (Grover sampling) |
| `QAlgBase/grover_run.py` | Plan 3 Task 3 (Grover sampling) |
| `QAlgBase/QAE_IQAE.py` | Plan 3 Task 3 (QAE sampling) |
| `QAlgBase/QAE_QAE.py` | Plan 3 Task 3 (QAE sampling) |
| `QAlgBase/QUBO_cir.py` | Plan 3 Task 3 (QUBO sampling); uncertain — tests `QUBO.QuadraticBinary.cir()` circuit construction shared by QUBO_QAOA (Task 2) and QUBO_GAS (Task 3); mapped to Task 3 per the QUBO sampling classification |
| `QAlgBase/comparator_interpolation_comparator.py` | Plan 3 Task 4 (QKmeans/QPCA/QSVM/QSVR/QSEncode) |
| `QAlgBase/comparator_int_comparator.py` | Plan 3 Task 4 (QKmeans/QPCA/QSVM/QSVR/QSEncode) |
| `QAlgBase/comparator_qft_comparator.py` | Plan 3 Task 4 (QKmeans/QPCA/QSVM/QSVR/QSEncode) |
| `QAlgBase/comparator_qft_qubit_comparator.py` | Plan 3 Task 4 (QKmeans/QPCA/QSVM/QSVR/QSEncode) |
| `QAlgBase/comparator_qubit_comparator.py` | Plan 3 Task 4 (QKmeans/QPCA/QSVM/QSVR/QSEncode) |
