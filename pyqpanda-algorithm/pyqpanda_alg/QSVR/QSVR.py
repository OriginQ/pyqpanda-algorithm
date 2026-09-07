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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, RX, RY, CZ, H
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .. plugin import *

class Quantum_SVR:
    """
    This class implements a Quantum Support Vector Regression (QSVR) framework using a custom 
    quantum kernel based on variational quantum circuits.

    Parameters
        x : ``array_like``\n
            Input features for training. If the number of features is less than 2, it will be padded;
            if more than 2, dimensionality reduction via PCA will be applied to reduce to 2D.
        y : ``array_like``\n
            Target regression labels.

    Attributes
        x : ``ndarray``\n
            2-dimensional input features after scaling and dimensionality adjustment.
        y : ``ndarray``\n
            Corresponding regression target values.

    Methods
        cir_real(qb, cs)
            Builds a real-valued parameterized quantum circuit for encoding input vectors into quantum states.

        dist(x, y)
            Computes the fidelity between two quantum-encoded states derived from input vectors `x` and `y`,
            used as a quantum kernel value.

        k_kernel(X, Y)
            Constructs the custom quantum kernel matrix based on all pairwise fidelity distances
            between samples in `X` and `Y`.

        get_res()
            Trains the quantum SVR model using the quantum kernel and returns both predicted and
            actual values on the training set.

        show_res()
            Visualizes the regression surface predicted by the QSVR model along with original training data,
            assuming the input has been reduced to 2D.

    Note
    
        This QSVR model uses pyQPanda to simulate quantum circuits and SciKit-Learn's SVR as the classical
        regression backend with a quantum-defined kernel.

    References
        [1] Schuld M., Sinayskiy I., Petruccione F. "An introduction to quantum machine learning."
            Contemporary Physics 56.2 (2015): 172-185.
            
    Examples:
        
    >>> from pyqpanda_alg import QSVR
    >>> import numpy as np
    >>> if __name__ == '__main__':
        
    >>>     n_samples = 100
    >>>     n_features = 2
    >>>     X = np.random.rand(n_samples, n_features) * 10
    >>>     y = (2 * np.sin(X[:, 0]) +
    >>>         1.5 * np.cos(X[:, 1]))
    >>>     QSVR.Quantum_SVR(X, y).show_res()

    """

    def __init__(self, x, y):
        scaler_x = StandardScaler()
        x = scaler_x.fit_transform(x)
        if x.shape[1] <= 1:
            column = np.zeros([len(x), 2 - x.shape[1]])
            x = np.hstack((x, column))
        elif x.shape[1] > 1:
            pca = PCA(n_components=2)
            x = pca.fit_transform(x)
        self.x = x
        self.y = y

    @staticmethod
    def cir_real(qb, cs):
        Qcir = QCircuit()
        for q in qb:
            Qcir << H(q)
        n = len(qb)
        for i in range(n):
            Qcir << RY(qb[i], cs[i])
        Qcir << CZ(qb[0], qb[1])
        for i in range(n):
            Qcir << RX(qb[i], cs[n - 1 - i])
        return Qcir

    def dist(self, x, y):
        machine = CPUQVM()
        prog = QProg(2)
        qv = prog.qubits()
        prog << self.cir_real(qv, x) << self.cir_real(qv, y).dagger()
        machine.run(prog, 1000)
        re = machine.result().get_prob_dict(qv)
        re = parse_quantum_result_dict(re, qv, select_max=-1)['0' * 2]
        return re

    def k_kernel(self, X, Y):
        """Build a fidelity kernel, reusing symmetry for equal sample arrays.

        For X == Y, K[i, j] = K[j, i] because fidelity is symmetric. Only
        n * (n + 1) / 2 simulator calls are needed instead of n ** 2.
        Distinct sample arrays retain the full pairwise evaluation, even
        when their shapes match. Diagonal values are still simulated.
        """
        matrix = np.zeros((len(X), len(Y)))
        symmetric = np.array_equal(X, Y)
        for i in range(len(X)):
            for j in range(i if symmetric else 0, len(Y)):
                matrix[i][j] = self.dist(X[i], Y[j])
                if symmetric:
                    matrix[j][i] = matrix[i][j]
        return matrix

    def get_res(self):
        svr = SVR(kernel=self.k_kernel, gamma=0.1)
        svr.fit(self.x, self.y)
        y_ppp = svr.predict(self.x)
        return y_ppp, self.y

    def show_res(self):
        svr = SVR(kernel=self.k_kernel, gamma=0.1)
        svr.fit(self.x, self.y)
        x0_test = np.linspace(min(self.x[:, 0]), max(self.x[:, 1]), 30)
        x1_test = np.linspace(min(self.x[:, 0]), max(self.x[:, 1]), 30)
        X0_test, X1_test = np.meshgrid(x0_test, x1_test)
        X_test = np.c_[X0_test.ravel(), X1_test.ravel()]
        y_pred = svr.predict(X_test).reshape(X0_test.shape)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x[:, 0], self.x[:, 1], self.y, color='darkorange', label='data')
        ax.plot_surface(X0_test, X1_test, y_pred, color='navy', alpha=0.5, label='SVR prediction')
        ax.set_xlabel('X[:, 0]')
        ax.set_ylabel('X[:, 1]')
        ax.set_zlabel('y')
        ax.set_title('SVR with 2-Dimensional Input')
        plt.show()


if __name__ == '__main__':
    n_samples = 100
    n_features = 2
    X = np.random.rand(n_samples, n_features) * 10
    y = (2 * np.sin(X[:, 0]) +
         1.5 * np.cos(X[:, 1]))
    Quantum_SVR(X, y).show_res()

