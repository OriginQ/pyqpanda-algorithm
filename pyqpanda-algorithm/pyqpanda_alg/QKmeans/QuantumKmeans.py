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

import numpy as np
from copy import deepcopy
from numpy import pi
from pyqpanda3.core import QCircuit, QProg, SWAP, U3, H, measure, draw_qprog

from ..execution import ExecutionOptions, resolve_backend




def _QuantumKmeansProg(theta0, phi0, theta, phi):
    """Build the swap-test distance circuit measuring the ancilla qubit."""
    prog = QProg(3)
    qlist = prog.qubits()
    cir = QCircuit()
    cir << H(qlist[2])
    cir << U3(qlist[0], theta0, phi0, 0)
    cir << U3(qlist[1], theta, phi, 0)
    cir << SWAP(qlist[0], qlist[1]).control(qlist[2])
    cir << H(qlist[2])
    prog << cir
    prog << measure(qlist[2], qlist[2])

    draw_qprog(prog)
    return prog


def _point_centroid_distances(point, centroids, k, backend, options):
    xval = [point[0]]
    for i in range(k):
      xval.append(centroids[i][0])
    yval = [point[1]]
    for i in range(k):
      yval.append(centroids[i][1])

    theta_t = [((x + 1) * pi / 2) for x in xval]
    theta_c = [((x + 1) * pi / 2) for x in yval]

    results_list = []

    for i in range(1, k + 1):
        prog = _QuantumKmeansProg(theta_c[0], theta_t[0], theta_c[i], theta_t[i])
        sample_task = backend.submit_sample(prog, options=options)
        counts = sample_task.result().single_counts()
        shots = sum(counts.values())
        results_list.append(counts.get('1', 0) / shots if shots else 0)
    return results_list


class QuantumKmeans:
    """
    QKMeans (Quantum K-Means) is a quantum algorithm that is used for clustering data into k clusters. 
    
    Parameters:
            data: ``ndarray``
                Input array, can be complex.

    Returns:
        centers_new: ``int``
                  Cluster center of input data.
        clusters: ``int``
               Category number of the entered data.

    Examples
        .. code-block:: python

            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import MinMaxScaler
            from pyqpanda_alg.QKmeans import QuantumKmeans
            from pyqpanda_alg import QKmeans


            def ReadCSV(path):
                import csv
                csvFile = open(path, "r")
                reader = csv.reader(csvFile)
                data = []
                label = []
                for item in reader:
                    data0 = []
                    if reader.line_num == 1:
                        continue
                    for i in range(1, 5):
                        data0.append(np.float64(item[i]))
                    data.append(data0)
                    label.append(item[5])
                csvFile.close()
                return np.array(data), label


            if __name__ == '__main__':
                data_path = QKmeans.__path__[0]
                data_file = os.path.join(data_path, 'dataset/Iris.csv')
                data, label = ReadCSV(data_file)
                data = data[:, 2:]
                label = [i for i in label]
                for _ in range(len(label)):
                    if label[_] == 'Iris-setosa':
                        label[_] = 'r'
                    elif label[_] == 'Iris-versicolor':
                        label[_] = 'b'
                    else:
                        label[_] = 'g'
                rows, col = data.shape[0], data.shape[1]

                scaler = MinMaxScaler(feature_range=(-1, 1))
                data = scaler.fit_transform(data)

                n_clusters = 3
                Colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown', 'pink']

                print('Classical K_means start...')
                Ctest = KMeans(n_clusters)
                Ctest.fit(data)
                Ccenter = Ctest.cluster_centers_
                print('Final Classical Result:', Ccenter)

                print('Quantum K_means start...')
                Qtest = QuantumKmeans(n_clusters)
                Qcenters, clusters = Qtest.fit(data)
                print('Final Quantum Result:', Qcenters)

                x0 = data[clusters == 0]
                x1 = data[clusters == 1]
                x2 = data[clusters == 2]
                plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
                plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
                plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')

                plot_title = 'Iris Quantum K_means Result'
                plt.title(plot_title)
                plt.show()


    """
    def __init__(self, k=2, tol=0.01):
        self.K = k
        self.tol = tol

    def fit(self, data, *, backend=None, execution_options=None):
        """
        Classify the input data.

        Parameters:
            data: ``ndarray``
                Input array, can be complex.
            backend: execution backend, keyword-only
                The sampling backend driving the distance estimation.
                Defaults to the local simulator.
            execution_options: ``ExecutionOptions``, keyword-only
                Sampling options (shots, timeout, ...) applied to every
                distance circuit.


        """
        exec_backend = resolve_backend(backend)
        options = execution_options if execution_options is not None else ExecutionOptions()

        n = data.shape[0]
        c = data.shape[1]

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        centers = np.random.randn(self.K, c) * std + mean

        centers_old = np.zeros(centers.shape)
        centers_new = deepcopy(centers)

        error = np.linalg.norm(centers_new - centers_old)
        upper_error = error + 1

        iter_counter = 1
        clusters = None
        while abs(error - upper_error) > self.tol:
            centers = centers_new

            distances = np.array(list(map(lambda x: _point_centroid_distances(x, centers, self.K, exec_backend, options), data)))

            clusters = np.argmin(distances, axis=1)

            if len(set(clusters)) != self.K:
                print('CALCULATE ERROR!RECALCULATE!')
                centers_new = np.random.randn(self.K, c) * std + mean
                continue

            centers_old = deepcopy(centers_new)

            for i in range(self.K):
                centers_new[i] = np.mean(data[clusters == i], axis=0)

            upper_error = deepcopy(error)
            error = np.linalg.norm(centers_new - centers_old)

            iter_counter += 1
            if error < self.tol:
                break
        return centers_new, clusters
