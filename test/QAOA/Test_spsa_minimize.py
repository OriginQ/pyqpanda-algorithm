import pytest
import numpy as np
import sys
from pathlib import Path

# 添加模块路径
sys.path.append((Path.cwd().parent.parent).__str__())
from pyqpanda_alg.QAOA import spsa


class TestSPSAMinimize:
    """spsa_minimize接口测试类"""
    
    @pytest.fixture
    def noise_function(self):
        """创建带噪声的测试函数"""
        class NoiseF:
            def __init__(self):
                self.eval_count = 0
                self.history = []
            
            def eval_f(self, x):
                self.eval_count += 1
                # 添加高斯噪声的二次函数
                return np.linalg.norm(x**2 + np.random.normal(0, 0.1, size=len(x)))
            
            def record(self, x):
                self.history.append([np.linalg.norm(x) / len(x), self.eval_count])
        
        return NoiseF()
    
    @pytest.fixture
    def simple_function(self):
        """创建简单的测试函数（无噪声）"""
        def func(x):
            return np.sum(x**2)  # 简单的二次函数
        return func
    
    def test_spsa_basic_functionality(self, noise_function):
        """测试SPSA基本功能"""
        x0 = np.array([1.0, 2.0, 3.0, 4.0])

        result = spsa.spsa_minimize(
            noise_function.eval_f, 
            x0, 
            callback=noise_function.record,
            maxiter=50  
        )
        
        # 验证结果类型和形状
        assert isinstance(result, np.ndarray)
        assert result.shape == x0.shape
        
        # 验证回调函数被调用
        assert len(noise_function.history) > 0
        assert noise_function.eval_count > 0

    def test_spsa_bounds_respected(self, simple_function):
        """Result stays inside per-variable bounds."""
        np.random.seed(0)
        x0 = np.array([5.0, 5.0])
        bounds = [(1.0, 2.0), (1.0, 2.0)]

        result = spsa.spsa_minimize(simple_function, x0, bounds=bounds, maxiter=100)

        assert (result >= 1.0).all()
        assert (result <= 2.0).all()

    def test_spsa_single_pair_bounds_respected(self, simple_function):
        """A single (min, max) pair applies to every variable."""
        np.random.seed(0)
        x0 = np.array([5.0, 5.0, 5.0])

        result = spsa.spsa_minimize(simple_function, x0, bounds=[(1.0, 2.0)], maxiter=100)

        assert result.shape == x0.shape
        assert (result >= 1.0).all()
        assert (result <= 2.0).all()

    def test_check_bounds_per_variable(self):
        """One pair per variable is returned as given."""
        bounds = [(0.0, 1.0), (-2.0, 2.0), (3.0, 4.0)]

        checked = spsa._check_bounds(3, bounds)

        assert checked is not None
        assert np.array_equal(checked, np.array(bounds))

    def test_check_bounds_single_pair_tiled(self):
        """A single pair is tiled, not multiplied."""
        checked = spsa._check_bounds(3, [(0, 1)])

        assert np.array_equal(checked, np.array([[0, 1], [0, 1], [0, 1]]))

    def test_check_bounds_validation(self):
        """Invalid bounds still raise, empty bounds still mean no bounds."""
        with pytest.raises(ValueError):
            spsa._check_bounds(2, [(2, 1), (0, 1)])

        with pytest.raises(IndexError):
            spsa._check_bounds(3, [(0, 1), (0, 1)])

        assert spsa._check_bounds(2, []) is None


