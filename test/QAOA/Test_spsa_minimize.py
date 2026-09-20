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
    def test_spsa_maxiter_remains_hard_cap_with_tolerance(self, simple_function):
        """maxiter must still terminate SPSA when tol is supplied but never met."""
        history = []
        result = spsa.spsa_minimize(
            simple_function,
            np.array([1.0, -2.0]),
            tol=0.0,
            callback=lambda x: history.append(x.copy()),
            maxiter=3,
        )

        assert isinstance(result, np.ndarray)
        assert len(history) == 3

    def test_spsa_tolerance_can_still_stop_before_maxiter(self, simple_function):
        """A satisfied tolerance should still stop SPSA before the hard cap."""
        history = []
        spsa.spsa_minimize(
            simple_function,
            np.array([1.0, -2.0]),
            tol=np.inf,
            callback=lambda x: history.append(x.copy()),
            maxiter=10,
        )

        assert len(history) == 1

