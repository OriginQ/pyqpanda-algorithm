# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
from pyqpanda3.core import CPUQVM, QCircuit, QProg, Encode, H
from scipy.fft import fft

from .. plugin import *


def _fast_walsh_hadamard_transform(values):
    """Return the unnormalized Walsh-Hadamard transform of a 1-D array.

    The input is zero-padded to the next power-of-two length, matching the
    behavior of :func:`sympy.fwht`.  The butterfly is evaluated with NumPy
    ufuncs, so numeric values stay in a native numeric dtype instead of being
    converted to symbolic Python objects.

    The input is never modified.  Runtime is ``O(N log N)`` and auxiliary
    memory is ``O(N)``.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError('values must be a 1D array')
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError('values must contain numeric data')

    size = array.size
    dtype = np.result_type(array.dtype, np.float64)
    if size == 0:
        return np.empty(0, dtype=dtype)

    padded_size = 1 << (size - 1).bit_length()
    output = np.zeros(padded_size, dtype=dtype)
    output[:size] = array
    if padded_size == 1:
        return output

    # One reusable half-size buffer avoids allocating a Python/SymPy object
    # for every coefficient and avoids a fresh temporary at every stage.
    scratch = np.empty(padded_size // 2, dtype=dtype)
    step = 1
    while step < padded_size:
        blocks = output.reshape(-1, 2 * step)
        left = blocks[:, :step]
        right = blocks[:, step:]
        saved_left = scratch.reshape(-1, step)
        np.copyto(saved_left, left)
        np.add(saved_left, right, out=left)
        np.subtract(saved_left, right, out=right)
        step *= 2

    return output


class QSpare_Code:
    
    """
    Quantum Sparse State Encoding Class.

    This class implements a quantum state preparation technique that encodes a sparse approximation
    of a probability distribution using either Walsh-Hadamard or Fourier basis. It allows for amplitude
    encoding and quantum circuit construction to simulate and analyze sparse distributions efficiently.

    Parameters
    ----------
    prob_list : list[float] or np.ndarray
        A list of probabilities representing the distribution to encode.
    cut_length : int, optional
        The number of dominant components (in the transformed basis) to retain for sparse encoding.
        If not specified, defaults to ``2 * log2(len(prob_list))``.
    mode : str, optional
        The basis transformation method, one of {'walsh', 'fourier'}. Default is 'walsh'.

    Attributes
    ----------
    prob : np.ndarray
        Normalized probability distribution (padded to power-of-2 length).
    qubits_num : int
        Number of qubits required for the state.
    amp : np.ndarray
        Square root of probabilities, used for amplitude encoding.
    cut : int
        Number of top components retained after transformation.
    mode : str
        Basis transform mode used for sparse encoding.

    Methods
    -------
    select_top_n_complex_numbers(arr, n)
        Selects top-n complex values by magnitude from the input array and zeroes out the rest.

    Transform(amp)
        Applies Walsh-Hadamard or Fourier transform to the amplitude vector based on the mode.

    quantum_cir(qubits)
        Constructs a quantum circuit that prepares the sparse encoded quantum state.

    Quantum_Res()
        Simulates the quantum circuit and returns the output probability distribution.

    Examples
    
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pyqpanda_alg.QSEncode import QSpare_Code
    >>> mu = 0
    >>> sigma = 1
    >>> x = np.linspace(-3, 3, 2 ** 10)
    >>> pdf_normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    >>> # pdf_normal = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))
    >>> ini = pdf_normal / np.linalg.norm(pdf_normal)
    >>> # res = QSEncode.QSpare_Code(ini ** 2, mode='walsh', cut_length=80).Quantum_Res()
    >>> res = QSpare_Code(ini ** 2, mode='fourier', cut_length=20).Quantum_Res()

    >>> plt.plot(x, res)
    >>> plt.plot(x, ini ** 2)
    >>> plt.show()

    """
    
    def __init__(self,
                 prob_list=None,
                 cut_length=None,
                 mode='walsh'
                 ):

        """
        Initialize the sparse quantum state encoding.

        Parameters
        ----------
        prob_list : list[float] or np.ndarray
            Probability list used for quantum state encoding.
        cut_length : int, optional
            Truncation length for sparse encoding.
        mode : str, optional
            Coefficient encoding method. Options are {'walsh', 'fourier'}.

        Returns
        -------
        QSpare_Code
            Instantiated object of the sparse quantum encoding class.
        """
        if prob_list is None:
            raise ValueError('prob list should be supported')
        if (not isinstance(prob_list, np.ndarray)) and (not isinstance(prob_list, list)):
            raise ValueError('prob_list should be np.ndarray or list')
        if len(prob_list) == 0:
            raise ValueError('at least one number in the prob_list ')

        all_floats = all(map(lambda x: isinstance(x, float), prob_list))
        if not all_floats:
            raise ValueError('elements of prob_list should be float type')
        if min(prob_list) < 0:
            raise ValueError('prob must > 0')
        if abs(np.sum(prob_list) - 1) > 0.001:
            raise Warning('sum of prob list should be 1')

        self.prob = np.array(prob_list) / np.sum(np.array(prob_list))
        self.qubits_num = int(np.ceil(np.log2(len(self.prob))))
        self.prob = np.append(self.prob, np.array([0] * (2 ** self.qubits_num - len(self.prob))))
        self.amp = self.prob ** 0.5
        self.cut = cut_length
        if self.cut is None:
            self.cut = 2 * self.qubits_num
        self.mode = mode

    def select_top_n_complex_numbers(self, arr, n):
        """
        Selects top-n elements by magnitude from the input complex array.

        Parameters
        ----------
        arr : np.ndarray
            The input array of complex or real values.
        n : int
            The number of largest-magnitude elements to keep.

        Returns
        -------
        np.ndarray
            A new array with only top-n magnitudes retained, rest set to zero.

        Notes
        -----
        Selection uses ``numpy.argpartition`` and therefore takes average
        ``O(N)`` time.  If equal magnitudes cross the selection boundary, the
        larger original indices are retained to make the result deterministic.
        """
        
        if type(n) != int or n <= 0:
            raise ValueError('n must > 0 and with class int')
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError('arr must be a 1D array')
        if arr.size == 0:
            return arr.copy()

        magnitudes = np.abs(arr)
        if not np.all(np.isfinite(magnitudes)):
            raise ValueError('arr must contain only finite values')
        if n >= arr.size:
            return arr.copy()

        # argpartition finds the top-n boundary in average O(N), whereas a
        # complete argsort performs O(N log N) work.  Values tied at the
        # boundary are resolved by index so results are deterministic.
        partition_at = arr.size - n
        partitioned = np.argpartition(magnitudes, partition_at)
        threshold = magnitudes[partitioned[partition_at]]
        top_n_indices = partitioned[partition_at:]

        # The partition result is already complete for the overwhelmingly
        # common unique-boundary case.  Only scan and rebuild the selection
        # when equal magnitudes straddle the top-n boundary.
        selected_ties = np.count_nonzero(magnitudes[top_n_indices] == threshold)
        all_ties = np.count_nonzero(magnitudes == threshold)
        if selected_ties != all_ties:
            top_n_indices = np.flatnonzero(magnitudes > threshold)
            slots_left = n - top_n_indices.size
            tied_indices = np.flatnonzero(magnitudes == threshold)
            top_n_indices = np.concatenate((top_n_indices, tied_indices[-slots_left:]))

        result = np.zeros_like(arr, dtype=arr.dtype)
        result[top_n_indices] = arr[top_n_indices]
        return result

    def Transform(self, amp):
        """
        Applies the selected basis transformation to the input amplitude vector.

        Parameters
        ----------
        amp : np.ndarray
            Input amplitude vector.

        Returns
        -------
        np.ndarray
            Transformed amplitude vector in the selected basis.

        Notes
        -----
        Walsh mode uses a native NumPy fast Walsh-Hadamard butterfly with
        ``O(N log N)`` time and ``O(N)`` memory.
        """
        if self.mode == 'walsh':
            transform = _fast_walsh_hadamard_transform(amp)
            transform /= np.sqrt(2 ** self.qubits_num)
        elif self.mode == 'fourier':
            transform = fft(amp)
        else:
            raise ValueError('mode only support walsh or fourier')
        return np.array(transform)

    def quantum_cir(self, qubits):
        """
        Constructs a quantum circuit to prepare the sparse-encoded quantum state.

        Parameters
        ----------
        qubits : list[Qubit]
            List of qubits to be used for the circuit.

        Returns
        -------
        QCircuit
            A quantum circuit for sparse amplitude encoding.
        """
        spare_sel = self.select_top_n_complex_numbers(self.Transform(self.amp), self.cut)
        if self.mode == 'walsh':
            spare_sel = spare_sel.astype(float)
        spare_sel = np.array(spare_sel) / np.linalg.norm(spare_sel)
        Qcir = QCircuit()
        encode = Encode()
        encode.amplitude_encode(qubits, spare_sel)
        Qcir << encode.get_circuit()
        if self.mode == 'walsh':
            for i in qubits:
                Qcir << H(i)
        elif self.mode == 'fourier':
            Qcir << QFT(qubits)
        else:
            raise ValueError('mode only support walsh or fourier')
        return Qcir

    def Quantum_Res(self):
        """
        Simulates the sparse quantum state circuit and returns the resulting measurement distribution.

        Returns
        -------
        list[float]
            The probability distribution from the simulated quantum circuit.
        """
        machine = CPUQVM()
        prog = QProg(self.qubits_num)
        qubit = prog.qubits()
        prog << self.quantum_cir(qubit)
        machine.run(prog, 1000)
        result = machine.result().get_prob_list(qubit)
        result = parse_quantum_result_list(result, qubit, select_max=-1)
        return result
