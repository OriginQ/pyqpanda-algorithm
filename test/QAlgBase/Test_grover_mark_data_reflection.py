import pytest
import numpy as np
from pyqpanda_alg.Grover import mark_data_reflection
from pyqpanda_alg.Grover import Grover
from pyqpanda3.core import CPUQVM, QProg


class Test_grover_mark_data_reflection:

    def test_mark_data_reflection_basic(self):
        m = CPUQVM()
        q_state = QProg(3).qubits()

        def mark(qubits):
            return mark_data_reflection(qubits=qubits, mark_data=['101', '001'])

        demo_search = Grover(flip_operator=mark)
        prog = QProg()
        prog << demo_search.cir(q_input=q_state)

        m.run(prog, shots=1000)
        res = m.result().get_prob_dict()
        assert '101' in res, "目标态 '101' 应该在结果中"
        assert '001' in res, "目标态 '001' 应该在结果中"

        target_prob = res.get('101', 0) + res.get('001', 0)
        assert target_prob > 0.1, f"两个目标态的总概率应该显著高于随机，当前为: {target_prob}"

        total_prob = sum(res.values())
        assert abs(total_prob - 1.0) < 0.01, f"概率总和应该为1，当前为: {total_prob}"

    def test_mark_data_reflection_flips_only_marked_states(self):
        """该算子应当是对角的，且只在被标记的态上取 -1。"""
        qubits = list(range(3))
        mark_data = ['101', '001']
        matrix = np.asarray(mark_data_reflection(qubits=qubits, mark_data=mark_data).matrix())

        off_diagonal = np.max(np.abs(matrix - np.diag(np.diag(matrix))))
        assert off_diagonal < 1e-9, f"相位翻转算子应为对角矩阵，非对角最大值为 {off_diagonal}"

        marked = {int(s, 2) for s in mark_data}
        for index in range(matrix.shape[0]):
            expected = -1.0 if index in marked else 1.0
            assert abs(matrix[index, index] - expected) < 1e-9, \
                f"基态 {index:03b} 的相位应为 {expected}，实际为 {matrix[index, index]}"

    def test_mark_data_reflection_accepts_single_string(self):
        """单个字符串与仅含该字符串的列表应当等价。"""
        qubits = list(range(3))
        from_string = np.asarray(mark_data_reflection(qubits=qubits, mark_data='011').matrix())
        from_list = np.asarray(mark_data_reflection(qubits=qubits, mark_data=['011']).matrix())
        assert np.max(np.abs(from_string - from_list)) < 1e-9, "字符串与单元素列表应生成相同电路"

    def test_mark_data_reflection_rejects_length_mismatch(self):
        """标记串长度与量子比特数不一致时应报错，而不是静默标记错误的态。"""
        qubits = list(range(3))
        with pytest.raises(ValueError):
            mark_data_reflection(qubits=qubits, mark_data=['1010'])
        with pytest.raises(ValueError):
            mark_data_reflection(qubits=qubits, mark_data=['10'])

    def test_mark_data_reflection_rejects_invalid_characters(self):
        qubits = list(range(3))
        with pytest.raises(ValueError):
            mark_data_reflection(qubits=qubits, mark_data=['1x1'])

    def test_mark_data_reflection_rejects_empty_mark_data(self):
        qubits = list(range(3))
        with pytest.raises(ValueError):
            mark_data_reflection(qubits=qubits, mark_data=[])

    def test_mark_data_reflection_contains_no_barrier(self):
        """算子中不应包含对态无影响、却会阻断编译优化的 BARRIER。"""
        qubits = list(range(4))
        circuit = mark_data_reflection(qubits=qubits, mark_data=['1011', '0110'])
        op_names = {str(name).upper() for name in circuit.count_ops()}
        assert 'BARRIER' not in op_names, f"电路中不应包含 BARRIER，实际算子为 {op_names}"


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v", "-s"])
