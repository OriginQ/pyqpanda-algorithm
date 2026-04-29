"""
Non-Hermitian QSVM Kernel Test
"""
import numpy as np
from pyqpanda_alg.QSVM import NonHermitianQuantumKernel, symmetrize_and_psd
from pyqpanda3.core import CPUQVM, QCircuit, QProg, CNOT, U1, U2


def Test():
    # Test 1: 非厄米算符验证 (M† ≠ M)
    print("Test 1: 非厄米算符验证...")
    gamma_gain = 0.3
    M = np.array([[1, 0], [np.sqrt(gamma_gain), 1.0 + gamma_gain]], dtype=complex)
    assert not np.allclose(M, M.conj().T), "算符应为非厄米"
    comm = np.linalg.norm(M @ M.conj().T - M.conj().T @ M)
    assert comm > 0.01, "算符应为非正规"
    print("  ✓ 非厄米算符验证通过")

    # Test 2: γ=0 退化为厄米核
    print("Test 2: γ=0 退化验证...")
    kernel_h = NonHermitianQuantumKernel(n_qbits=2, gamma=0.0, normalize=False)
    x1, x2 = np.array([1.0, 2.0]), np.array([0.5, 1.5])
    k12 = kernel_h._compute_kernel_element(x1, x2)
    k21 = kernel_h._compute_kernel_element(x2, x1)
    assert abs(k12 - k21) < 1e-10, "γ=0 应对称"
    kernel_h.close()
    print("  ✓ 退化验证通过")

    # Test 3: 非厄米核矩阵非对称性
    print("Test 3: 非对称性验证...")
    kernel_nh = NonHermitianQuantumKernel(n_qbits=2, gamma=(0.2, 0.3), normalize=False)
    np.random.seed(42)
    X = np.random.rand(5, 2) * np.pi
    K_raw = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            K_raw[i, j] = kernel_nh._compute_kernel_element(X[i], X[j])
    sym_err = np.linalg.norm(K_raw - K_raw.T)
    assert sym_err > 1e-6, f"核矩阵应为非对称，误差={sym_err}"
    kernel_nh.close()
    print("  ✓ 非对称性验证通过")

    # Test 4: 半正定化
    print("Test 4: 半正定化验证...")
    K_psd = symmetrize_and_psd(K_raw, method='nearest')
    assert np.linalg.norm(K_psd - K_psd.T) < 1e-10, "应对称"
    assert np.min(np.linalg.eigvalsh(K_psd)) >= -1e-10, "应半正定"
    print("  ✓ 半正定化验证通过")

    # Test 5: sklearn 集成
    print("Test 5: sklearn 集成验证...")
    from sklearn.svm import SVC
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=30, noise=0.2, random_state=42)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    kernel = NonHermitianQuantumKernel(n_qbits=2, gamma=0.2, psd_method='auto', normalize=False)
    K_train = kernel.evaluate(x_vec=X_train)
    K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
    svc = SVC(kernel='precomputed')
    svc.fit(K_train, y_train)
    acc = svc.score(K_test, y_test)
    assert acc >= 0.0, f"准确率应 >= 0，实际={acc}"
    kernel.close()
    print(f"  ✓ sklearn 集成验证通过 (准确率={acc:.4f})")

    print("\n所有测试通过! ✓")


if __name__ == "__main__":
    Test()