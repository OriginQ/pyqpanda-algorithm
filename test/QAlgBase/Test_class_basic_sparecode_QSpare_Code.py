import pytest
import numpy as np
from pyqpanda3.core import QProg, QCircuit
from pyqpanda_alg.QSEncode import QSpare_Code
import warnings
import os


def _wht_matrix(n):
    """独立的 Walsh-Hadamard 变换矩阵（自然序）：W[i, j] = (-1)^popcount(i & j) / sqrt(n)。

    与 sympy.fwht 的约定一致（未归一化部分），用于对拍 QSpare_Code.Transform 的 walsh 模式。
    """
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = (-1) ** (bin(i & j).count("1"))
    return mat / np.sqrt(n)


def _dft_matrix(n, sign=-1, normalized=False):
    """独立的离散傅里叶变换矩阵：F[k, j] = exp(sign * 2πi * k * j / n)（可选 1/sqrt(n) 归一化）。

    sign=-1 且不归一化时与 scipy.fft.fft 约定一致（对拍 Transform 的 fourier 模式）；
    sign=+1 且归一化时与量子线路 QFT 酉操作约定一致（对拍端到端结果）。
    """
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    mat = np.exp(sign * 2j * np.pi * k * j / n)
    if normalized:
        mat = mat / np.sqrt(n)
    return mat


def _sparse_encode_predict(prob_list, cut, mode):
    """独立手工复现 QSpare_Code 的编码流程，作为小例子端到端对拍基准。

    步骤：归一化 -> 补零到 2^n -> 开方得振幅 -> 基变换 -> 按模长保留 top-cut
    -> 再归一化 -> 振幅编码后施加 H^⊗n（walsh）或 QFT（fourier），输出概率。
    """
    prob = np.array(prob_list, dtype=float)
    prob = prob / np.sum(prob)
    n_qubits = int(np.ceil(np.log2(len(prob))))
    prob = np.append(prob, np.zeros(2 ** n_qubits - len(prob)))
    amp = np.sqrt(prob)
    n = len(amp)
    if mode == 'walsh':
        forward = _wht_matrix(n)
        backward = _wht_matrix(n)
    else:
        forward = _dft_matrix(n, sign=-1, normalized=False)
        backward = _dft_matrix(n, sign=+1, normalized=True)
    transformed = forward @ amp
    top_idx = np.argsort(np.abs(transformed))[-cut:]
    sparse = np.zeros_like(transformed)
    sparse[top_idx] = transformed[top_idx]
    sparse = sparse / np.linalg.norm(sparse)
    out = backward @ sparse
    return np.abs(out) ** 2


class Test_class_basic_sparecode_QSpare_Code:
    """QSpare_Code模块测试类"""

    def setup_method(self):
        """测试方法前置设置"""
        warnings.filterwarnings("ignore")
        np.random.seed(42)  # 设置随机种子保证结果可重现
        # 确保测试输出目录存在
        os.makedirs('test_outputs', exist_ok=True)


    def test_interface12_walsh_mode_basic_1(self):
        mu = 0
        sigma = 1
        x = np.linspace(-3, 3, 2 ** 10)
        pdf_normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        ini = pdf_normal / np.linalg.norm(pdf_normal)
        res = QSpare_Code(ini ** 2, mode='walsh', cut_length=20).Quantum_Res()
        res1 = QSpare_Code(ini ** 2, mode='fourier', cut_length=20).Quantum_Res()

        prob_list = ini ** 2
        original_peak = x[np.argmax(prob_list)]
        result_peak = x[np.argmax(res)]
        result_peak1 = x[np.argmax(res1)]
        assert abs(original_peak - result_peak) < 0.5
        assert abs(original_peak - result_peak1) < 0.5

    # ---------------- 构造参数校验（异常路径） ----------------

    def test_interface13_init_none_prob_list_raises(self):
        """prob_list 缺省（None）应抛出 ValueError"""
        with pytest.raises(ValueError, match="prob list should be supported"):
            QSpare_Code()

    def test_interface14_init_wrong_type_raises(self):
        """prob_list 非 list/np.ndarray（如 tuple、str）应抛出 ValueError"""
        with pytest.raises(ValueError, match="np.ndarray or list"):
            QSpare_Code((0.5, 0.5))
        with pytest.raises(ValueError, match="np.ndarray or list"):
            QSpare_Code("0.5,0.5")

    def test_interface15_init_empty_list_raises(self):
        """prob_list 为空列表应抛出 ValueError"""
        with pytest.raises(ValueError, match="at least one number"):
            QSpare_Code([])

    def test_interface16_init_int_elements_raises(self):
        """prob_list 元素为 int（非 float）应抛出 ValueError"""
        with pytest.raises(ValueError, match="float type"):
            QSpare_Code([1, 2, 3])

    def test_interface17_init_negative_raises(self):
        """prob_list 含负数应抛出 ValueError"""
        with pytest.raises(ValueError, match="prob must > 0"):
            QSpare_Code([0.5, -0.1, 0.6])

    def test_interface18_init_sum_not_one_raises_warning(self):
        """prob_list 概率和与 1 的偏差超过 0.001 应抛出 Warning 异常"""
        with pytest.raises(Warning, match="sum of prob list should be 1"):
            QSpare_Code([0.5, 0.4])

    # ---------------- 构造基本行为（归一化/补零/默认截断） ----------------

    def test_interface19_init_normalize_and_attributes(self):
        """概率和与 1 偏差在 0.001 内会被自动归一化，ndarray 与 list 均可接受"""
        qs = QSpare_Code([0.3333, 0.3333, 0.3333])
        np.testing.assert_allclose(qs.prob, [1 / 3, 1 / 3, 1 / 3, 0.0], atol=1e-12)
        np.testing.assert_allclose(qs.amp, np.sqrt([1 / 3, 1 / 3, 1 / 3, 0.0]), atol=1e-12)
        assert qs.qubits_num == 2
        qs_nd = QSpare_Code(np.array([0.25, 0.75]))
        np.testing.assert_allclose(qs_nd.prob, [0.25, 0.75], atol=1e-12)

    def test_interface20_init_padding_to_power_of_two(self):
        """长度非 2 的幂时补零到 2^n，qubits_num 向上取整，补零后概率和仍为 1"""
        qs = QSpare_Code([0.2, 0.3, 0.5])
        assert qs.qubits_num == 2
        assert len(qs.prob) == 4
        np.testing.assert_allclose(qs.prob, [0.2, 0.3, 0.5, 0.0], atol=1e-12)
        assert abs(np.sum(qs.prob) - 1.0) < 1e-12

    def test_interface21_init_default_cut_length(self):
        """cut_length 缺省时默认 2 * log2(len)，显式传入时保持原值"""
        assert QSpare_Code([0.5, 0.5]).cut == 2          # 1 qubit -> 2 * 1
        assert QSpare_Code([0.2, 0.3, 0.5]).cut == 4     # 2 qubit -> 2 * 2
        assert QSpare_Code([0.5, 0.5], cut_length=1).cut == 1

    # ---------------- select_top_n_complex_numbers 单元测试 ----------------

    def test_interface22_select_top_n_keeps_largest_magnitude(self):
        """按模长保留 top-n，其余置零（逐值对拍）"""
        qs = QSpare_Code([0.5, 0.5])
        arr = np.array([1 + 1j, 3.0, 2j, 0.5])
        res = qs.select_top_n_complex_numbers(arr, 2)
        expected = np.array([0 + 0j, 3 + 0j, 0 + 2j, 0 + 0j])
        np.testing.assert_array_equal(res, expected)

    def test_interface23_select_top_n_invalid_n_raises(self):
        """n 非正整数或非 int 类型应抛出 ValueError"""
        qs = QSpare_Code([0.5, 0.5])
        arr = np.array([1.0, 2.0])
        for bad_n in (0, -1, 2.5, "2"):
            with pytest.raises(ValueError, match="n must > 0"):
                qs.select_top_n_complex_numbers(arr, bad_n)

    def test_interface24_select_top_n_not_less_than_len_keeps_all(self):
        """n >= len(arr) 时所有元素保留"""
        qs = QSpare_Code([0.5, 0.5])
        arr = np.array([1 + 1j, 3.0, 2j, 0.5])
        res = qs.select_top_n_complex_numbers(arr, 10)
        np.testing.assert_array_equal(res, arr)

    # ---------------- Transform 单元测试（与独立矩阵对拍） ----------------

    def test_interface25_transform_walsh_matches_independent_wht(self):
        """walsh 模式 Transform 结果与独立构造的 WHT 矩阵逐值一致"""
        qs = QSpare_Code([0.5, 0.25, 0.25, 0.0])
        expected = _wht_matrix(4) @ qs.amp
        # sympy.fwht 返回 object dtype（sympy.Float），此处仅比较数值
        np.testing.assert_allclose(np.asarray(qs.Transform(qs.amp), dtype=float), expected, atol=1e-12)

    def test_interface26_transform_fourier_matches_independent_dft(self):
        """fourier 模式 Transform 结果与独立构造的 DFT 矩阵（负号、未归一化）逐值一致"""
        qs = QSpare_Code([0.4, 0.3, 0.2, 0.1], mode='fourier')
        expected = _dft_matrix(4, sign=-1, normalized=False) @ qs.amp
        np.testing.assert_allclose(qs.Transform(qs.amp), expected, atol=1e-12)

    def test_interface27_transform_invalid_mode_raises(self):
        """非法 mode 在 Transform 阶段抛出 ValueError（构造阶段不校验）"""
        qs = QSpare_Code([0.5, 0.5], mode='bad_mode')
        with pytest.raises(ValueError, match="mode only support walsh or fourier"):
            qs.Transform(qs.amp)

    # ---------------- quantum_cir 单元测试 ----------------

    def test_interface28_quantum_cir_returns_qcircuit(self):
        """双模式 quantum_cir 均返回 QCircuit；非法 mode 与非法 cut_length 在构建期抛错"""
        prog = QProg(2)
        qubits = prog.qubits()
        cir_w = QSpare_Code([0.5, 0.25, 0.25, 0.0], mode='walsh').quantum_cir(qubits)
        cir_f = QSpare_Code([0.5, 0.25, 0.25, 0.0], mode='fourier').quantum_cir(qubits)
        assert isinstance(cir_w, QCircuit)
        assert isinstance(cir_f, QCircuit)
        with pytest.raises(ValueError, match="mode only support walsh or fourier"):
            QSpare_Code([0.5, 0.25, 0.25, 0.0], mode='bad_mode').quantum_cir(qubits)
        with pytest.raises(ValueError, match="n must > 0"):
            QSpare_Code([0.5, 0.25, 0.25, 0.0], cut_length=0).quantum_cir(qubits)

    # ---------------- Quantum_Res 端到端小例子逐值对拍 ----------------

    def test_interface29_quantum_res_walsh_one_qubit_exact(self):
        """walsh 模式单比特：delta 分布原样恢复，均匀分布经 H 后仍为均匀"""
        res_delta = QSpare_Code([1.0, 0.0], mode='walsh', cut_length=2).Quantum_Res()
        np.testing.assert_allclose(res_delta, [1.0, 0.0], atol=1e-9)
        res_uniform = QSpare_Code([0.5, 0.5], mode='walsh', cut_length=1).Quantum_Res()
        np.testing.assert_allclose(res_uniform, [0.5, 0.5], atol=1e-9)

    def test_interface30_quantum_res_walsh_two_qubit_matches_oracle(self):
        """walsh 模式双比特小向量与独立手工推算结果逐值对拍（变换系数模长无并列）"""
        prob = [0.5, 0.3, 0.15, 0.05]
        res = QSpare_Code(prob, mode='walsh', cut_length=2).Quantum_Res()
        expected = _sparse_encode_predict(prob, cut=2, mode='walsh')
        assert len(res) == 4
        np.testing.assert_allclose(res, expected, atol=1e-9)
        # 手工基准值（防 oracle 与被测实现同错）：振幅开方 -> WHT -> 保留 w0/w2 -> 归一化 -> 再过 WHT 取平方
        np.testing.assert_allclose(
            res, [0.40419839165941823, 0.40419839165941823, 0.09580160834058182, 0.09580160834058182],
            atol=1e-9)

    def test_interface31_quantum_res_fourier_matches_oracle(self):
        """fourier 模式（含非对称分布）与独立手工推算结果逐值对拍"""
        prob = [0.4, 0.3, 0.2, 0.1]
        res = QSpare_Code(prob, mode='fourier', cut_length=3).Quantum_Res()
        expected = _sparse_encode_predict(prob, cut=3, mode='fourier')
        np.testing.assert_allclose(res, expected, atol=1e-9)
        res_uniform = QSpare_Code([0.5, 0.5], mode='fourier', cut_length=2).Quantum_Res()
        np.testing.assert_allclose(res_uniform, [0.5, 0.5], atol=1e-9)

    def test_interface32_quantum_res_padding_three_elements(self):
        """长度 3 的输入补零到 4，双模式输出长度 4、概率和为 1 且与 oracle 一致"""
        prob = [0.2, 0.3, 0.5]
        for mode in ('walsh', 'fourier'):
            res = QSpare_Code(prob, mode=mode, cut_length=2).Quantum_Res()
            assert len(res) == 4
            assert abs(np.sum(res) - 1.0) < 1e-9
            np.testing.assert_allclose(res, _sparse_encode_predict(prob, cut=2, mode=mode), atol=1e-9)

    def test_interface33_quantum_res_default_cut_matches_oracle(self):
        """缺省 cut_length（2 * log2 n）时输出与 oracle 一致"""
        prob = [0.5, 0.25, 0.125, 0.125]
        for mode in ('walsh', 'fourier'):
            res = QSpare_Code(prob, mode=mode).Quantum_Res()
            np.testing.assert_allclose(res, _sparse_encode_predict(prob, cut=4, mode=mode), atol=1e-9)

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
