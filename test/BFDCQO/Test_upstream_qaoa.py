import itertools

import numpy as np
import pytest
import sympy as sp

from pyqpanda_alg.BFDCQO.problems import IsingProblem
from pyqpanda_alg.BFDCQO.benchmark.upstream_qaoa import (
    UpstreamQAOASolver,
    ising_to_sympy,
    upstream_qaoa,
)
from pyqpanda_alg.QAOA import qaoa


def test_ising_to_sympy_preserves_every_basis_state_energy():
    model = IsingProblem(
        h=np.array([0.75, -1.25, 0.5]),
        J=np.array(
            [
                [0.0, 1.5, -0.25],
                [1.5, 0.0, 0.8],
                [-0.25, 0.8, 0.0],
            ]
        ),
        offset=-0.375,
    )

    expression = ising_to_sympy(model)
    symbols = sp.symbols("x0:3")

    for bits in itertools.product((0, 1), repeat=model.n):
        actual = float(expression.subs(dict(zip(symbols, bits))))
        assert actual == pytest.approx(model.energy_bits(bits), abs=1e-12)


def test_ising_to_sympy_rejects_an_unconstrained_variable():
    model = IsingProblem(
        h=np.array([1.0, 0.0]),
        J=np.zeros((2, 2)),
        offset=0.25,
    )

    with pytest.raises(ValueError, match="unconstrained"):
        ising_to_sympy(model)


def test_symbol_order_matches_qubit_order_above_ten_variables():
    fields = np.arange(1.0, 12.0)
    model = IsingProblem(h=fields, J=np.zeros((11, 11)))
    expression = ising_to_sympy(model)
    engine = upstream_qaoa.QAOA(expression)

    bit_two = [0] * model.n
    bit_two[2] = 1
    bit_ten = [0] * model.n
    bit_ten[10] = 1

    assert engine.calculate_energy(bit_two) == pytest.approx(
        model.energy_bits(bit_two), abs=1e-12
    )
    assert engine.calculate_energy(bit_ten) == pytest.approx(
        model.energy_bits(bit_ten), abs=1e-12
    )


def test_adapter_uses_repository_upstream_qaoa_class():
    assert upstream_qaoa is qaoa
    assert upstream_qaoa.QAOA is qaoa.QAOA
    assert upstream_qaoa.QAOA.__module__ == "pyqpanda_alg.QAOA.qaoa"


def test_real_cpuqvm_run_reports_observed_budget_and_little_endian_keys():
    # The unique optimum flips only q0.  In the public CPUQVM result format q0
    # is the right-most bit, so the non-palindromic optimum must be ``001``.
    model = IsingProblem(
        h=np.array([1.0, -1.0, -1.0]),
        J=np.zeros((3, 3)),
    )
    adapter = UpstreamQAOASolver(
        model,
        p=1,
        shots=128,
        max_evals=8,
        initial_parameter_seed=0,
    )

    result = adapter.run()
    engine = adapter.qaoa_instance

    assert engine.__class__ is upstream_qaoa.QAOA
    assert result.distribution
    assert sum(result.distribution.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(result.distribution) <= {format(i, "03b") for i in range(8)}
    assert max(result.distribution, key=result.distribution.get) == "001"
    assert model.energy_bitstring("001") < model.energy_bitstring("100")

    assert result.parameters.shape == (2,)
    assert np.all(np.isfinite(result.parameters))
    assert np.isfinite(result.loss)
    assert result.initial_parameter_seed == 0
    assert np.array_equal(
        result.initial_parameters,
        np.random.default_rng(0).random(2) * np.pi,
    )
    assert result.objective_evaluations == max(engine.iter - 1, 0)
    assert result.circuit_evaluations == engine.circuit_iter
    assert result.total_shots == result.circuit_evaluations * adapter.shots
    assert result.total_shots > adapter.shots


def test_same_initial_parameter_seed_generates_same_explicit_initial_parameters(monkeypatch):
    captured = []

    def fake_run(engine, **kwargs):
        captured.append(np.asarray(kwargs["initial_para"]).copy())
        return {"0": 1.0}, np.asarray(kwargs["initial_para"]), 0.0

    monkeypatch.setattr(upstream_qaoa.QAOA, "run", fake_run)
    model = IsingProblem(h=np.array([1.0]), J=np.zeros((1, 1)))

    first = UpstreamQAOASolver(
        model, p=2, shots=4, max_evals=4, initial_parameter_seed=17
    ).run()
    second = UpstreamQAOASolver(
        model, p=2, shots=4, max_evals=4, initial_parameter_seed=17
    ).run()

    assert np.array_equal(captured[0], captured[1])
    assert np.array_equal(first.initial_parameters, second.initial_parameters)
    assert first.initial_parameter_seed == second.initial_parameter_seed == 17


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        ({}, "non-empty"),
        ({"0": float("nan")}, "finite"),
        ({"0": -0.1, "1": 1.1}, "non-negative"),
        ({"0": 0.0, "1": 0.0}, "positive total"),
    ],
)
def test_invalid_upstream_distributions_are_rejected(monkeypatch, distribution, message):
    def fake_run(engine, **kwargs):
        return distribution, np.asarray(kwargs["initial_para"]), 0.0

    monkeypatch.setattr(upstream_qaoa.QAOA, "run", fake_run)
    model = IsingProblem(h=np.array([1.0]), J=np.zeros((1, 1)))

    with pytest.raises(RuntimeError, match=message):
        UpstreamQAOASolver(model, shots=4, max_evals=4).run()


def test_upstream_distribution_is_normalized(monkeypatch):
    def fake_run(engine, **kwargs):
        return {"0": 2.0, "1": 1.0}, np.asarray(kwargs["initial_para"]), 0.0

    monkeypatch.setattr(upstream_qaoa.QAOA, "run", fake_run)
    model = IsingProblem(h=np.array([1.0]), J=np.zeros((1, 1)))

    result = UpstreamQAOASolver(model, shots=4, max_evals=4).run()

    assert result.distribution == pytest.approx({"0": 2 / 3, "1": 1 / 3})


@pytest.mark.parametrize("name", ["p", "shots", "max_evals"])
def test_bool_is_not_accepted_as_an_integer_parameter(name):
    model = IsingProblem(h=np.array([1.0]), J=np.zeros((1, 1)))
    arguments = {name: True}

    with pytest.raises(ValueError, match=name):
        UpstreamQAOASolver(model, **arguments)


@pytest.mark.parametrize("seed", [True, -1, 1.5, "7", 2**32])
def test_invalid_initial_parameter_seed_is_rejected(seed):
    model = IsingProblem(h=np.array([1.0]), J=np.zeros((1, 1)))

    with pytest.raises(ValueError, match="initial_parameter_seed"):
        UpstreamQAOASolver(model, initial_parameter_seed=seed)


def test_seed_alias_has_only_initial_parameter_semantics():
    model = IsingProblem(h=np.array([1.0]), J=np.zeros((1, 1)))

    with pytest.warns(DeprecationWarning, match="initial parameters"):
        adapter = UpstreamQAOASolver(model, seed=7)

    assert adapter.initial_parameter_seed == 7


def test_seed_alias_cannot_be_combined_with_explicit_name():
    model = IsingProblem(h=np.array([1.0]), J=np.zeros((1, 1)))

    with pytest.raises(ValueError, match="not both"):
        UpstreamQAOASolver(model, initial_parameter_seed=7, seed=7)


def test_zero_variable_model_is_rejected():
    model = IsingProblem(h=np.array([]), J=np.zeros((0, 0)))

    with pytest.raises(ValueError, match="at least one variable"):
        UpstreamQAOASolver(model)
