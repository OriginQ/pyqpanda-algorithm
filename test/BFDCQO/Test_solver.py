import importlib.util
import numpy as np
import pytest

from pyqpanda_alg.BFDCQO import BFDCQO, IsingProblem


def make_instance() -> IsingProblem:
    """Return a fixed frustrated six-spin regression instance."""
    h = np.array([0.42, -0.31, 0.18, -0.57, 0.23, -0.14])
    J = np.array([
        [0.00,  0.71, -0.64,  0.00,  0.32, -0.28],
        [0.71,  0.00,  0.51, -0.77,  0.00,  0.24],
        [-0.64, 0.51,  0.00,  0.46, -0.39,  0.00],
        [0.00, -0.77, 0.46,  0.00,  0.62, -0.55],
        [0.32,  0.00, -0.39,  0.62,  0.00,  0.67],
        [-0.28, 0.24,  0.00, -0.55,  0.67,  0.00],
    ])
    return IsingProblem(h=h, J=J)


def test_bfdcqo_reaches_exact_ground_state_with_seeded_sampling():
    p = make_instance()
    result = BFDCQO(p, n_trotter=3, dt=0.1, iterations=10, shots=4000, seed=11).solve()
    assert result.success
    assert result.best_solution in result.exact_ground_states
    assert result.history[-1].ground_state_probability > result.history[0].ground_state_probability


def test_exact_feedback_is_deterministic():
    p = make_instance()
    a = BFDCQO(p, n_trotter=3, dt=0.1, iterations=5, shots=None, seed=1).solve()
    b = BFDCQO(p, n_trotter=3, dt=0.1, iterations=5, shots=None, seed=999).solve()
    assert np.allclose(
        [r.ground_state_probability for r in a.history],
        [r.ground_state_probability for r in b.history],
    )


@pytest.mark.skipif(importlib.util.find_spec("pyqpanda3") is None, reason="pyqpanda3 not installed")
def test_pyqpanda_backend_matches_numpy_backend():
    p = make_instance()
    # Exact-feedback one-iteration comparison isolates circuit semantics.
    a = BFDCQO(p, n_trotter=2, dt=0.1, iterations=1, shots=None, backend="numpy").solve()
    b = BFDCQO(p, n_trotter=2, dt=0.1, iterations=1, shots=None, backend="pyqpanda").solve()
    assert np.isclose(a.history[0].expected_energy, b.history[0].expected_energy, atol=1e-8)
    assert np.isclose(a.history[0].ground_state_probability, b.history[0].ground_state_probability, atol=1e-8)


def test_unified_api_basic_and_adaptive_construct_and_run():
    from pyqpanda_alg.BFDCQO import BFDCQOConfig, BFDCQOSolver
    p = make_instance()
    basic = BFDCQOSolver(
        p,
        BFDCQOConfig(variant="basic", iterations=2, shots=64),
        backend="numpy",
        seed=3,
    ).solve()
    assert basic.total_shots == 128

    adaptive = BFDCQOSolver(
        p,
        BFDCQOConfig(
            variant="adaptive_cvar",
            iterations=3,
            shots=64,
            total_shot_budget=192,
            min_shots=32,
            max_shots=96,
            initial_shots=64,
        ),
        backend="numpy",
        seed=3,
    ).solve()
    assert adaptive.total_shots == 192


def test_unified_api_hybrid_constructs_bounded_local_refinement():
    from pyqpanda_alg.BFDCQO import BFDCQOConfig, BFDCQOSolver
    p = make_instance()
    result = BFDCQOSolver(
        p,
        BFDCQOConfig(
            variant="hybrid",
            base_variant="basic",
            iterations=2,
            shots=64,
            neighborhood_size=3,
            local_rounds=1,
        ),
        backend="numpy",
        seed=5,
    ).solve()
    assert result.total_shots == 128
    assert result.local_evaluations == 8
    assert result.best_energy <= result.quantum_best_energy + 1e-12
