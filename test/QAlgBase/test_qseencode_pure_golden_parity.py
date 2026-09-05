"""Small Frozen scientific parity fixture.

Provenance (constants copied; no runtime import):
- source: qseencode-final-technical-hardening-2026-08-18/
  final_technical_hardening.py
- source SHA-256:
  B24336428BD62DFF3C479B5C1207329BC61AA68B92009E77A2EE6537981D11A8
- result source: kstar_independent_recomputation.csv
- result SHA-256:
  C71AEBAB8E5C8EB11D31727AA76DBCB42DA0D77533BFAEBCE394E5B35AC24F69
- case identity: gaussian, N=8, fidelity_target=0.99, walsh/fourier
"""

import numpy as np

from pyqpanda_alg.QSEncode._error_budget import (
    find_k_star,
    top_k_coefficients,
    top_k_indices,
)
from pyqpanda_alg.QSEncode._transforms import normalized_fourier, normalized_fwht


PROBABILITIES = np.array(
    [
        0.0006917643261373052,
        0.015724004731018214,
        0.1261730210273901,
        0.3574112099154543,
        0.3574112099154544,
        0.1261730210273902,
        0.01572400473101823,
        0.0006917643261373052,
    ]
)
FROZEN_WALSH = np.array(
    [
        0.7811719797235759,
        0.0,
        7.850462293418875e-17,
        0.10149554964818927,
        -1.1775693440128312e-16,
        -0.2416356009244349,
        -0.566640298480657,
        3.925231146709437e-17,
    ]
)
FROZEN_FOURIER_UNNORMALIZED = np.array(
    [
        2.209488016541843 + 0j,
        -1.1381776680196536 - 0.47144862648392233j,
        0.1435363828329812 + 0.14353638283298098j,
        -0.004897071058339253 - 0.011822575364947241j,
        0.0 + 0j,
        -0.004897071058339253 + 0.011822575364947241j,
        0.1435363828329812 - 0.14353638283298098j,
        -1.1381776680196536 + 0.47144862648392233j,
    ]
)


def test_frozen_gaussian_walsh_coefficients_ranking_energy_and_kstar():
    coefficients = normalized_fwht(np.sqrt(PROBABILITIES))
    np.testing.assert_allclose(coefficients, FROZEN_WALSH, atol=1e-12, rtol=1e-10)
    result = find_k_star(coefficients, 0.99)
    assert result.k_star == 4
    np.testing.assert_array_equal(top_k_indices(coefficients, 4), [3, 5, 6, 0])
    assert abs(result.previous_retained_energy - 0.989698653401612) <= 1e-12
    assert abs(result.retained_energy - 1.0) <= 1e-12


def test_frozen_gaussian_fourier_scale_scientific_invariants_and_kstar():
    coefficients = normalized_fourier(np.sqrt(PROBABILITIES))
    np.testing.assert_allclose(
        coefficients * np.sqrt(8),
        FROZEN_FOURIER_UNNORMALIZED,
        atol=1e-12,
        rtol=1e-10,
    )
    result = find_k_star(coefficients, 0.99)
    assert result.k_star == 4
    np.testing.assert_array_equal(top_k_indices(coefficients, 4), [6, 1, 7, 0])
    assert abs(result.previous_retained_energy - 0.9896577147533093) <= 1e-12
    assert abs(result.retained_energy - 0.9948083880525034) <= 1e-12
    np.testing.assert_allclose(
        top_k_coefficients(coefficients, 4, normalize=True),
        top_k_coefficients(FROZEN_FOURIER_UNNORMALIZED, 4, normalize=True),
        atol=1e-12,
        rtol=1e-10,
    )
