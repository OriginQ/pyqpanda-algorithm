from pyqpanda_alg.QARM import QuantumAssociationRulesMining


def _fixture_qarm(min_conf=0.0):
    qarm = QuantumAssociationRulesMining(
        [
            ['A', 'B', 'C'],
            ['A', 'B', 'C'],
            ['A', 'B'],
            ['A', 'C'],
        ],
        min_support=0.5,
        min_conf=min_conf,
    )
    frequent_levels = [
        [(1,), (2,), (3,)],
        [(1, 2), (1, 3), (2, 3)],
        [(1, 2, 3)],
    ]
    frequent_data = {
        (1,): [[0, 1, 2, 3], 1.0],
        (2,): [[0, 1, 2], 0.75],
        (3,): [[0, 1, 3], 0.75],
        (1, 2): [[0, 1, 2], 0.75],
        (1, 3): [[0, 1, 3], 0.75],
        (2, 3): [[0, 1], 0.5],
        (1, 2, 3): [[0, 1], 0.5],
    }
    qarm._fk_result = lambda *_args, **_kwargs: (frequent_levels, frequent_data)
    return qarm


def test_get_all_conf_enumerates_all_nonempty_proper_subsets():
    qarm = _fixture_qarm()
    result = qarm._get_all_conf(None, None, 0, None, '', 'CPU')

    assert result == {
        'A->B': 0.75,
        'B->A': 1.0,
        'A->C': 0.75,
        'C->A': 1.0,
        'B->C': 0.67,
        'C->B': 0.67,
        'A->B,C': 0.5,
        'B->A,C': 0.67,
        'C->A,B': 0.67,
        'A,B->C': 0.67,
        'A,C->B': 0.67,
        'B,C->A': 1.0,
    }


def test_get_all_conf_applies_confidence_threshold_to_higher_order_rules():
    qarm = _fixture_qarm(min_conf=0.8)
    result = qarm._get_all_conf(None, None, 0, None, '', 'CPU')

    assert result == {
        'B->A': 1.0,
        'C->A': 1.0,
        'B,C->A': 1.0,
    }


def test_get_conf_key_is_deterministic_for_unordered_sets():
    qarm = _fixture_qarm()
    assert qarm._get_conf_key(frozenset({3, 1}), frozenset({2})) == 'A,C->B'
