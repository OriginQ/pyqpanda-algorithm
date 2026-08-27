"""
Regression tests for QARM singleton row recovery, support calculation,
and probability-decoding correctness.

The secondary _get_all_conf() rule-generation defect is not tested here.
"""
import sys
import os
from pathlib import Path
import pytest
from pyqpanda_alg.QARM import QuantumAssociationRulesMining
from pyqpanda_alg import QARM

sys.path.append(Path.cwd().parent.parent.__str__())


# ============================================================
# Helpers
# ============================================================

def read_dataset(file_path):
    if os.path.exists(file_path):
        trans_data = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.strip():
                    data_list = line.strip().split(',')
                    trans_data.append([data.strip() for data in data_list])
        return trans_data
    else:
        raise FileNotFoundError(f'The file {file_path} does not exist!')


def classical_exact_rows(transactions, target_item):
    """Return sorted list of transaction indices containing target_item."""
    return sorted(i for i, t in enumerate(transactions) if target_item in t)


def classical_exact_support(transactions, target_item):
    """Return exact support = len(rows) / len(transactions)."""
    rows = classical_exact_rows(transactions, target_item)
    return len(rows) / len(transactions)


def qarm_f1_for_item(transactions, min_support, min_confidence, item_name):
    """
    Run the real _find_f1() and return (rows, support) for the named item.
    This exercises the F1 row-recovery contract through _find_f1(),
    including the two-item classical fallback.
    """
    from pyqpanda3.core import CPUQVM, QProg
    qarm = QuantumAssociationRulesMining(transactions, min_support, min_confidence)
    machine = CPUQVM()
    qarm.machine = machine
    prog = QProg(qarm.number_qubits)
    qlist = prog.qubits()
    clist = prog.cbits()
    position = 0

    c1 = qarm._create_c1(qarm.transaction_matrix)
    f1, f1_dict = qarm._find_f1(qlist, clist, position, c1, None, "", "CPU")

    # Find the item_id for the given item_name
    item_id = None
    for iid, iname in qarm.items_dict.items():
        if iname == item_name:
            item_id = iid
            break
    if item_id is None:
        raise ValueError(f"Item '{item_name}' not found in dataset")

    # Look up the item in f1_dict by its tuple key
    target_key = (item_id,)
    if target_key in f1_dict:
        rows, support = f1_dict[target_key]
        return sorted(rows), support
    return [], 0.0


def qarm_public_run(transactions, min_support, min_confidence):
    """Run QARM public API and return the rule dictionary."""
    qarm = QuantumAssociationRulesMining(transactions, min_support, min_confidence)
    return qarm.run()


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def data2():
    """Bundled data2.txt dataset."""
    data_path = QARM.__path__[0]
    return read_dataset(os.path.join(data_path, 'dataset/data2.txt'))


@pytest.fixture
def data2_ground_truth():
    """Classical ground truth for all data2.txt items."""
    data_path = QARM.__path__[0]
    transactions = read_dataset(os.path.join(data_path, 'dataset/data2.txt'))
    return {item: {
        'rows': classical_exact_rows(transactions, item),
        'support': classical_exact_support(transactions, item),
    } for item in ['面包', '牛奶', '奶酪', '黄油']}


@pytest.fixture
def dense_3item_8txn():
    """3 items, 8 txns. Target A in all 8. N=32, M=8, M/N=1/4."""
    return [
        ['A', 'B', 'C'], ['A', 'B'], ['A', 'C'], ['A', 'B', 'C'],
        ['A', 'B'], ['A', 'C'], ['A', 'B', 'C'], ['A', 'B'],
    ]


@pytest.fixture
def boundary_2item_4txn():
    """2 items, 4 txns. Target A in all 4. N=8, M=4, M/N=1/2."""
    return [
        ['A', 'B'], ['A'], ['A', 'B'], ['A'],
    ]


@pytest.fixture
def padded_3item_5txn():
    """3 items, 5 txns (non-power-of-2). N=32, 3 padded tx states."""
    return [
        ['A', 'B'], ['A', 'C'], ['A', 'B'], ['B', 'C'], ['A', 'B'],
    ]


# ============================================================
# Contract A+B: data2 F1 rows, support, and invariant
# ============================================================

class TestData2F1:
    """data2.txt: QARM F1 rows/support must match classical, 0 <= s <= 1."""

    @pytest.mark.parametrize("item_name", ['面包', '牛奶', '奶酪', '黄油'])
    def test_f1_rows_and_support(self, data2, data2_ground_truth, item_name):
        qarm_rows, qarm_support = qarm_f1_for_item(data2, 0.2, 0.5, item_name)
        expected = data2_ground_truth[item_name]
        assert qarm_rows == expected['rows'], \
            f"F1 rows for '{item_name}': QARM={qarm_rows}, expected={expected['rows']}"
        assert qarm_support == pytest.approx(expected['support'], abs=0.001), \
            f"Support for '{item_name}': QARM={qarm_support:.4f}, expected={expected['support']:.4f}"
        assert 0.0 <= qarm_support <= 1.0, \
            f"Support for '{item_name}' = {qarm_support} violates [0,1] invariant"


# ============================================================
# Contract C: user-visible end-to-end rule
# ============================================================

class TestEndToEndRule:
    """Public run() API: 牛奶->面包 confidence must be 0.80."""

    def test_milk_to_bread_confidence(self, data2):
        result = qarm_public_run(data2, 0.2, 0.5)
        rule_key = '牛奶->面包'
        assert rule_key in result, \
            f"Rule '{rule_key}' not found in QARM output: {list(result.keys())}"
        assert result[rule_key] == 0.8, \
            f"'{rule_key}' confidence: expected 0.8, got {result[rule_key]}"


# ============================================================
# Contract D: dense 3-item quantum-path case (M/N=1/4)
# ============================================================

class TestDense3ItemCase:
    """3 items, 8 txns: A in all 8. M/N=1/4, t=3 has maximal separation."""

    def test_f1_rows_and_support(self, dense_3item_8txn):
        qarm_rows, qarm_support = qarm_f1_for_item(dense_3item_8txn, 0.3, 0.5, 'A')
        expected_rows = classical_exact_rows(dense_3item_8txn, 'A')
        expected_support = classical_exact_support(dense_3item_8txn, 'A')
        assert qarm_rows == expected_rows, \
            f"Dense A rows: QARM={qarm_rows}, expected={expected_rows}"
        assert qarm_support == pytest.approx(expected_support), \
            f"Dense A support: QARM={qarm_support:.4f}, expected={expected_support:.4f}"
        assert 0.0 <= qarm_support <= 1.0


# ============================================================
# Contract E: two-item fallback boundary (M/N=1/2)
# ============================================================

class TestTwoItemBoundary:
    """2 items, 4 txns: A in all 4. M/N=1/2, Grover degeneracy."""

    def test_f1_rows_and_support(self, boundary_2item_4txn):
        qarm_rows, qarm_support = qarm_f1_for_item(boundary_2item_4txn, 0.3, 0.5, 'A')
        expected_rows = classical_exact_rows(boundary_2item_4txn, 'A')
        expected_support = classical_exact_support(boundary_2item_4txn, 'A')
        assert qarm_rows == expected_rows, \
            f"Boundary A rows: QARM={qarm_rows}, expected={expected_rows}"
        assert qarm_support == pytest.approx(expected_support), \
            f"Boundary A support: QARM={qarm_support:.4f}, expected={expected_support:.4f}"
        assert 0.0 <= qarm_support <= 1.0


# ============================================================
# Contract F: padded search-space case
# ============================================================

class TestPaddedSpace:
    """3 items, 5 txns (non-power-of-2): no padded indices, correct rows."""

    def test_f1_rows_and_support(self, padded_3item_5txn):
        qarm = QuantumAssociationRulesMining(padded_3item_5txn, 0.3, 0.5)
        qarm_rows, qarm_support = qarm_f1_for_item(padded_3item_5txn, 0.3, 0.5, 'A')
        expected_rows = classical_exact_rows(padded_3item_5txn, 'A')
        expected_support = classical_exact_support(padded_3item_5txn, 'A')
        assert all(r < qarm.transaction_number for r in qarm_rows), \
            f"Padded indices found: {[r for r in qarm_rows if r >= qarm.transaction_number]}"
        assert qarm_rows == expected_rows, \
            f"Padded A rows: QARM={qarm_rows}, expected={expected_rows}"
        assert qarm_support == pytest.approx(expected_support), \
            f"Padded A support: QARM={qarm_support:.4f}, expected={expected_support:.4f}"
        assert 0.0 <= qarm_support <= 1.0


# ============================================================
# Contract G: dictionary-order independence
# ============================================================

class TestDecodingOrderIndependence:
    """
    Prove that _get_result() correctness is invariant to
    dictionary insertion order of get_prob_dict().
    """

    def test_result_independent_of_dict_order(self, monkeypatch):
        """Monkeypatch _iter_cir to return a dict with non-ascending key order."""
        data = [['A', 'B'], ['A'], ['B'], ['A', 'B']]
        qarm = QuantumAssociationRulesMining(data, 0.3, 0.5)

        # Crafted dict with non-ascending insertion order:
        # key '010' (int 2) has the highest probability 0.98
        # key '111' (int 7) and '000' (int 0) are low
        # Inserted in order: 7, 2, 0
        crafted_dict = {
            '111': 0.01,   # key_int=7, dict position 0
            '010': 0.98,   # key_int=2, dict position 1 — actual maximum
            '000': 0.01,   # key_int=0, dict position 2
        }

        # Pre-fix failure chain:
        #   max probability 0.98 belongs to key '010'
        #   -> result.values() has maximum at position 1
        #   -> the old code passes integer 1 to _get_index
        #   -> integer 1 is decoded as basis state '001' (not '010')
        #   -> wrong transaction/item returned
        #
        # Correct behaviour: use actual key '010' (int 2) directly.

        def mock_iter_cir(*args, **kwargs):
            return crafted_dict

        monkeypatch.setattr(qarm, '_iter_cir', mock_iter_cir)

        from pyqpanda3.core import CPUQVM, QProg
        machine = CPUQVM()
        qarm.machine = machine
        prog = QProg(qarm.number_qubits)
        qlist = prog.qubits()
        clist = prog.cbits()
        position = 0

        result = qarm._get_result(qlist, clist, position, 1, 1, None, "", "CPU")

        # With idx_qn=3, trans_qn=2, items_qn=1:
        # key '010' (int 2) -> bin='010' -> tx=int('01',2)=1, item=int('0',2)=0
        # key '001' (int 1, the WRONG result from position-based decoding) -> tx=0, item=1
        assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
        tx, item = result[0]
        assert tx == 1, \
            f"tx=1 expected from key '010', got tx={tx} (position-based gives tx=0)"
        assert item == 0, \
            f"item=0 expected from key '010', got item={item}"


# ============================================================
# Contract H: probability-plateau decoding (corrected)
# ============================================================

class TestDecodingPlateau:
    """
    Prove that decoding recovers all true maxima sharing the same
    target item index, and excludes states that round to the same
    4-decimal value but are genuinely lower.
    """

    def test_plateau_recovers_all_maxima(self, monkeypatch):
        """Two true maxima at same item index, one rounding-impostor."""
        data = [['A', 'B'], ['A'], ['B'], ['A', 'B']]
        qarm = QuantumAssociationRulesMining(data, 0.3, 0.5)

        # QARM structure for this dataset: items_qn=1, trans_qn=2, idx_qn=3
        # Keys: tx bits (2) + item bits (1)
        # Target item index = 0 (item A)
        #
        # '010' -> tx=1, item=0  : true maximum  (~0.25004000000001)
        # '100' -> tx=2, item=0  : true maximum  (~0.25004000000000)
        # '000' -> tx=0, item=0  : rounding impostor (0.24996, rounds to 0.25)
        # '001' -> tx=0, item=1  : low filler
        # '011' -> tx=1, item=1  : low filler
        # '101' -> tx=2, item=1  : low filler
        # '110' -> tx=3, item=0  : low filler
        # '111' -> tx=3, item=1  : low filler

        p_max_a = 0.25004000000001
        p_max_b = 0.25004000000000
        p_impostor = 0.24996
        p_low = (1.0 - p_max_a - p_max_b - p_impostor) / 5.0

        crafted_dict = {
            '000': p_impostor,   # impostor: rounds to 0.25 but is lower
            '001': p_low,
            '010': p_max_a,      # true maximum
            '011': p_low,
            '100': p_max_b,      # true maximum
            '101': p_low,
            '110': p_low,
            '111': p_low,
        }

        assert sum(crafted_dict.values()) == pytest.approx(1.0), \
            "Fixture must be normalized"

        # Verify that all three (p_max_a, p_max_b, p_impostor) round to the same
        assert round(p_max_a, 4) == round(p_max_b, 4) == round(p_impostor, 4), \
            "Fixture must demonstrate rounding ambiguity"

        def mock_iter_cir(*args, **kwargs):
            return crafted_dict

        monkeypatch.setattr(qarm, '_iter_cir', mock_iter_cir)

        from pyqpanda3.core import CPUQVM, QProg
        machine = CPUQVM()
        qarm.machine = machine
        prog = QProg(qarm.number_qubits)
        qlist = prog.qubits()
        clist = prog.cbits()
        position = 0

        result = qarm._get_result(qlist, clist, position, 1, 1, None, "", "CPU")

        # Expected: both true maxima decoded
        # key '010' -> tx=1, item=0
        # key '100' -> tx=2, item=0
        expected = {(1, 0), (2, 0)}
        assert set(result) == expected, \
            f"Expected {expected}, got {set(result)}. " \
            f"Impostor key '000' (tx=0,item=0) must be excluded."


# ============================================================
# GREEN control: existing normal case still passes
# ============================================================

class TestExistingNormalCase:
    """Verify that the existing normal-case test contract still holds."""

    def test_qarm_returns_dict(self, data2):
        """QARM run() returns a non-None dict (existing contract)."""
        result = qarm_public_run(data2, 0.2, 0.5)
        assert result is not None
        assert isinstance(result, dict)