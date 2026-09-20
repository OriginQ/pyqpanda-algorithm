'''
QKMeans (Quantum K-Means) is a quantum algorithm that is used for clustering data into k clusters,
where k is a user-specified parameter. It is an extension of the classical K-Means algorithm,
which is widely used in machine learning and data analysis.
QKMeans has the potential to provide speedup over classical K-Means when processing large datasets.
'''

from functools import wraps
from numbers import Integral

from .QuantumKmeans import QuantumKmeans


_original_fit = QuantumKmeans.fit


@wraps(_original_fit)
def _validated_fit(self, data):
    if isinstance(self.K, bool) or not isinstance(self.K, Integral) or self.K <= 0:
        raise ValueError("k must be a positive integer")

    sample_count = data.shape[0]
    if self.K > sample_count:
        raise ValueError(
            f"k ({self.K}) cannot exceed the number of samples ({sample_count})"
        )

    return _original_fit(self, data)


QuantumKmeans.fit = _validated_fit

__all__ = ['QuantumKmeans']