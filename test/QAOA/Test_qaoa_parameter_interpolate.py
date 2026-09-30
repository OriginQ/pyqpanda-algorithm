import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.append((Path.cwd().parent.parent).__str__())

from pyqpanda_alg.QAOA import qaoa


class TestParameterInterpolate:
    
    def test_basic_functionality(self):
        initial_parameter = np.array([0.1, 0.2, 0.2, 0.1])
        new_parameter = qaoa.parameter_interpolate(initial_parameter)
        
        assert isinstance(new_parameter, np.ndarray)
        assert new_parameter.shape == (6,)  # 2*(2+1)=6
        
        print(f"{initial_parameter}")
        print(f"{new_parameter}")


if __name__ == "__main__":
    # run test
    pytest.main([__file__, "-v", "-s"])