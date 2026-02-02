import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append((Path.cwd().parent.parent).__str__())
from pyqpanda_alg.QAOA import spsa


class TestSPSAMinimize:
    
    @pytest.fixture
    def noise_function(self):
        class NoiseF:
            def __init__(self):
                self.eval_count = 0
                self.history = []
            
            def eval_f(self, x):
                self.eval_count += 1
                return np.linalg.norm(x**2 + np.random.normal(0, 0.1, size=len(x)))
            
            def record(self, x):
                self.history.append([np.linalg.norm(x) / len(x), self.eval_count])
        
        return NoiseF()
    
    @pytest.fixture
    def simple_function(self):
        def func(x):
            return np.sum(x**2)
        return func
    
    def test_spsa_basic_functionality(self, noise_function):
        x0 = np.array([1.0, 2.0, 3.0, 4.0])

        result = spsa.spsa_minimize(
            noise_function.eval_f, 
            x0, 
            callback=noise_function.record,
            maxiter=50  
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x0.shape
        
        assert len(noise_function.history) > 0
        assert noise_function.eval_count > 0
    

