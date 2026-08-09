import numpy as np
import pytest
import sympy as sp

from pyqpanda_alg.QAOA.qaoa import QAOA


@pytest.fixture
def risk_qaoa():
    qaoa = QAOA.__new__(QAOA)
    qaoa.energy_dict = {
        '00': 3.0,
        '01': -1.0,
        '10': -4.0,
    }
    return qaoa


def test_cvar_uses_lowest_energy_tail_and_normalizes(risk_qaoa):
    risk_qaoa.alpha = 0.2
    distribution = {'00': 0.7, '01': 0.2, '10': 0.1}

    result = risk_qaoa._loss_function_cvar(distribution)

    # The lowest 20% consists of all p=0.1 at E=-4 and p=0.1 at E=-1.
    assert result == pytest.approx((-4.0 * 0.1 - 1.0 * 0.1) / 0.2)


def test_cvar_alpha_one_matches_energy_expectation(risk_qaoa):
    risk_qaoa.alpha = 1.0
    distribution = {'00': 0.7, '01': 0.2, '10': 0.1}

    assert risk_qaoa._loss_function_cvar(distribution) == pytest.approx(
        risk_qaoa._loss_function_default(distribution)
    )


@pytest.mark.parametrize('alpha', [0, -0.1, 1.1, np.nan, np.inf, True, '0.5'])
def test_cvar_rejects_invalid_alpha(risk_qaoa, alpha):
    risk_qaoa.alpha = alpha

    with pytest.raises(ValueError, match=r'\(0, 1\]'):
        risk_qaoa._loss_function_cvar({'00': 1.0})


def test_gibbs_matches_direct_formula_for_regular_values(risk_qaoa):
    risk_qaoa.temperature = 0.5
    distribution = {'00': 0.7, '01': 0.2, '10': 0.1}
    expected = -np.log(sum(
        probability * np.exp(-risk_qaoa.energy_dict[solution] / risk_qaoa.temperature)
        for solution, probability in distribution.items()
    ))

    assert risk_qaoa._loss_function_Gibbs(distribution) == pytest.approx(expected)


def test_gibbs_remains_finite_for_large_energy_magnitudes(risk_qaoa):
    risk_qaoa.temperature = 0.1
    risk_qaoa.energy_dict = {'0': -1000.0, '1': 1000.0}

    result = risk_qaoa._loss_function_Gibbs({'0': 0.5, '1': 0.5})

    assert np.isfinite(result)
    assert result == pytest.approx(-10000.0 + np.log(2.0))


@pytest.mark.parametrize('temperature', [0, -0.1, 1.1, np.nan, np.inf, True, '0.5'])
def test_gibbs_rejects_invalid_temperature(risk_qaoa, temperature):
    risk_qaoa.temperature = temperature

    with pytest.raises(ValueError, match=r'\(0, 1\]'):
        risk_qaoa._loss_function_Gibbs({'00': 1.0})


def test_gibbs_rejects_distribution_without_positive_probability(risk_qaoa):
    risk_qaoa.temperature = 0.5

    with pytest.raises(ValueError, match='positive probability'):
        risk_qaoa._loss_function_Gibbs({'00': 0.0})


def test_symbolic_energy_function_is_compiled_once(monkeypatch):
    x0, x1 = sp.symbols('x0:2')
    qaoa = QAOA(2 * x0 * x1 - x0)

    def fail_if_recompiled(*args, **kwargs):
        raise AssertionError('calculate_energy should reuse the compiled function')

    monkeypatch.setattr(sp, 'lambdify', fail_if_recompiled)

    assert qaoa.calculate_energy([1, 0]) == -1
    assert qaoa.calculate_energy([1, 1]) == 1
