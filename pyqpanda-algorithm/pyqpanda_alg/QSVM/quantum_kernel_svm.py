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
from pyqpanda3.core import CPUQVM, QCircuit, QProg, CNOT, U1, U2, measure


from ..plugin import *





def _build_circuit(qlist, n_qbits, weights_x, weights_y):
    circuit = QCircuit()
    for i in range(n_qbits):
        circuit << U2(qlist[i], 0, np.pi)
        circuit << U1(qlist[i], 2.0 * weights_x[i])
    circuit << CNOT(qlist[0], qlist[1])
    circuit << U1(qlist[1], 2.0 * (np.pi - weights_x[0]) * (np.pi - weights_x[1]))
    circuit << CNOT(qlist[0], qlist[1])

    for i in range(n_qbits):
        circuit << U2(qlist[i], 0, np.pi)
        circuit << U1(qlist[i], 2.0 * weights_x[i])
    circuit << CNOT(qlist[0], qlist[1])
    circuit << U1(qlist[1], 2.0 * (np.pi - weights_x[0]) * (np.pi - weights_x[1]))
    circuit << U1(qlist[1], -2.0 * (np.pi - weights_y[0]) * (np.pi - weights_y[1]))
    circuit << CNOT(qlist[0], qlist[1])

    for i in range(n_qbits):
        circuit << U1(qlist[i], -2.0 * weights_y[i])
        circuit << U2(qlist[i], 0, np.pi)
    circuit << CNOT(qlist[0], qlist[1])
    circuit << U1(qlist[1], -2.0 * (np.pi - weights_y[0]) * (np.pi - weights_y[1]))
    circuit << CNOT(qlist[0], qlist[1])
    for i in range(n_qbits):
        circuit << U1(qlist[i], -2.0 * weights_y[i])
        circuit << U2(qlist[i], 0, np.pi)

    return circuit


def _run_circuit(n_qbits, weights_x, weights_y):
    machine = CPUQVM()
    prog = QProg(n_qbits)
    qubits = prog.qubits()
    circuit = QCircuit()
    circuit << _build_circuit(qubits, n_qbits, weights_x, weights_y)
    prog << circuit
    prog << measure(qubits, qubits)
    machine.run(prog, 1024)
    result = machine.result().get_counts()
    return result


class QuantumKernel_vqnet:
    """
    A help class to create Quantum Kernal Matrix.The evaluate function can be used in ``sklearn.svm`` .

    Parameters
        x_vec: ``ndarray``\n
            Train or test dataset features.
        y_vec: ``ndarray``\n
            Train or test dataset labels.

    Returns
        out: ``ndarray``\n
            Kernal matrix.

    Examples
        .. code-block:: python

            import sys
            from pathlib import Path
            sys.path.append((Path.cwd().parent.parent).__str__())
            import os
            import numpy as np
            from sklearn.svm import SVC
            import matplotlib
            try:
                matplotlib.use('TkAgg')
            except:
                pass
            import matplotlib.pyplot as plt

            from pyqpanda_alg.QSVM import QuantumKernel_vqnet

            from pyqpanda_alg import QSVM
            data_path = QSVM.__path__[0]


            def _read_vqc_qsvm_data(path):
                train_features = np.loadtxt(os.path.join(path, "dataset/qsvm_train_features.txt"))
                test_features = np.loadtxt(os.path.join(path, "dataset/qsvm_test_features.txt"))
                train_labels = np.loadtxt(os.path.join(path, "dataset/qsvm_train_labels.txt"))
                test_labels = np.loadtxt(os.path.join(path, "dataset/qsvm_test_labels.txt"))
                samples = np.loadtxt(os.path.join(path, "dataset/qsvm_samples.txt"))
                return train_features, test_features, train_labels, test_labels, samples
            def qsvm_classification():
                train_features, test_features, train_labels, test_labels, samples = _read_vqc_qsvm_data(data_path)
                plt.figure(figsize=(5, 5))
                plt.ylim(0, 2 * np.pi)
                plt.xlim(0, 2 * np.pi)
                plt.imshow(
                    np.asmatrix(samples).T,
                    interpolation="nearest",
                    origin="lower",
                    cmap="RdBu",
                    extent=[0, 2 * np.pi, 0, 2 * np.pi],
                )

                plt.scatter(
                    train_features[np.where(train_labels[:] == 0), 0],
                    train_features[np.where(train_labels[:] == 0), 1],
                    marker="s",
                    facecolors="w",
                    edgecolors="b",
                    label="A train",
                )
                plt.scatter(
                    train_features[np.where(train_labels[:] == 1), 0],
                    train_features[np.where(train_labels[:] == 1), 1],
                    marker="o",
                    facecolors="w",
                    edgecolors="r",
                    label="B train",
                )
                plt.scatter(
                    test_features[np.where(test_labels[:] == 0), 0],
                    test_features[np.where(test_labels[:] == 0), 1],
                    marker="s",
                    facecolors="b",
                    edgecolors="w",
                    label="A test",
                )
                plt.scatter(
                    test_features[np.where(test_labels[:] == 1), 0],
                    test_features[np.where(test_labels[:] == 1), 1],
                    marker="o",
                    facecolors="r",
                    edgecolors="w",
                    label="B test",
                )

                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
                plt.title("samples dataset for classification")

                plt.show()
                samples_datasets_kernel = QuantumKernel_vqnet(n_qbits=2)
                samples_datasets_svc = SVC(kernel=samples_datasets_kernel.evaluate)
                samples_datasets_svc.fit(train_features, train_labels)
                samples_datasets_score = samples_datasets_svc.score(test_features, test_labels)

                print(f"Callable kernel classification test score: {samples_datasets_score}")

                samples_datasets_matrix_train = samples_datasets_kernel.evaluate(x_vec=train_features)
                samples_datasets_matrix_test = samples_datasets_kernel.evaluate(x_vec=test_features, y_vec=train_features)

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(
                    np.asmatrix(samples_datasets_matrix_train), interpolation="nearest", origin="upper", cmap="Blues"
                )
                axs[0].set_title("samples training kernel matrix")
                axs[1].imshow(np.asmatrix(samples_datasets_matrix_test), interpolation="nearest", origin="upper", cmap="Reds")
                axs[1].set_title("samples testing kernel matrix")
                plt.show()


            if __name__ == "__main__":
                qsvm_classification()
    """

    def __init__(
            self,
            batch_size: int = 100,
            n_qbits=None,
    ) -> None:

        self._batch_size = batch_size

        self._n_qbits = n_qbits

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        """
        Evaluation function to build quantum kernel.

        Parameters
            x_vec: ``ndarray``\n
                Train or test dataset features.
            y_vec: ``ndarray``\n
                Train or test dataset labels.

        Returns
            out: ``ndarray``\n
                Kernal matrix.

        Examples
            .. code-block:: python

                import os
                import numpy as np
                from sklearn.svm import SVC
                import matplotlib
                try:
                    matplotlib.use('TkAgg')
                except:
                    pass
                import matplotlib.pyplot as plt


                from pyqpanda_alg.QSVM.quantum_kernel_svm import QuantumKernel_vqnet

                from pyqpanda_alg import QSVM
                data_path = QSVM.__path__[0]


                def _read_vqc_qsvm_data(path):
                    train_features = np.loadtxt(os.path.join(path, "dataset/qsvm_train_features.txt"))
                    test_features = np.loadtxt(os.path.join(path, "dataset/qsvm_test_features.txt"))
                    train_labels = np.loadtxt(os.path.join(path, "dataset/qsvm_train_labels.txt"))
                    test_labels = np.loadtxt(os.path.join(path, "dataset/qsvm_test_labels.txt"))
                    samples = np.loadtxt(os.path.join(path, "dataset/qsvm_samples.txt"))
                    return train_features, test_features, train_labels, test_labels, samples
                def qsvm_classification():
                    train_features, test_features, train_labels, test_labels, samples = _read_vqc_qsvm_data(data_path)
                    plt.figure(figsize=(5, 5))
                    plt.ylim(0, 2 * np.pi)
                    plt.xlim(0, 2 * np.pi)
                    plt.imshow(
                        np.asmatrix(samples).T,
                        interpolation="nearest",
                        origin="lower",
                        cmap="RdBu",
                        extent=[0, 2 * np.pi, 0, 2 * np.pi],
                    )

                    plt.scatter(
                        train_features[np.where(train_labels[:] == 0), 0],
                        train_features[np.where(train_labels[:] == 0), 1],
                        marker="s",
                        facecolors="w",
                        edgecolors="b",
                        label="A train",
                    )
                    plt.scatter(
                        train_features[np.where(train_labels[:] == 1), 0],
                        train_features[np.where(train_labels[:] == 1), 1],
                        marker="o",
                        facecolors="w",
                        edgecolors="r",
                        label="B train",
                    )
                    plt.scatter(
                        test_features[np.where(test_labels[:] == 0), 0],
                        test_features[np.where(test_labels[:] == 0), 1],
                        marker="s",
                        facecolors="b",
                        edgecolors="w",
                        label="A test",
                    )
                    plt.scatter(
                        test_features[np.where(test_labels[:] == 1), 0],
                        test_features[np.where(test_labels[:] == 1), 1],
                        marker="o",
                        facecolors="r",
                        edgecolors="w",
                        label="B test",
                    )

                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
                    plt.title("samples dataset for classification")

                    plt.show()
                    samples_datasets_kernel = QuantumKernel_vqnet(n_qbits=2)
                    samples_datasets_svc = SVC(kernel=samples_datasets_kernel.evaluate)
                    samples_datasets_svc.fit(train_features, train_labels)
                    samples_datasets_score = samples_datasets_svc.score(test_features, test_labels)

                    print(f"Callable kernel classification test score: {samples_datasets_score}")

                    samples_datasets_matrix_train = samples_datasets_kernel.evaluate(x_vec=train_features)
                    samples_datasets_matrix_test = samples_datasets_kernel.evaluate(x_vec=test_features, y_vec=train_features)

                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].imshow(
                        np.asmatrix(samples_datasets_matrix_train), interpolation="nearest", origin="upper", cmap="Blues"
                    )
                    axs[0].set_title("samples training kernel matrix")
                    axs[1].imshow(np.asmatrix(samples_datasets_matrix_test), interpolation="nearest", origin="upper", cmap="Reds")
                    axs[1].set_title("samples testing kernel matrix")
                    plt.show()


                if __name__ == "__main__":
                    qsvm_classification()

        """
        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)
        if y_vec is not None and not isinstance(y_vec, np.ndarray):
            y_vec = np.asarray(y_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = np.reshape(x_vec, (-1, len(x_vec)))

        if y_vec is not None and y_vec.ndim > 2:
            raise ValueError("y_vec must be a 1D or 2D array")

        if y_vec is not None and y_vec.ndim == 1:
            y_vec = np.reshape(y_vec, (-1, len(y_vec)))

        if y_vec is not None and y_vec.shape[1] != x_vec.shape[1]:
            raise ValueError(
                "x_vec and y_vec have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
            )

        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        # initialize kernel matrix
        kernel = np.zeros((x_vec.shape[0], y_vec.shape[0]))

        # set diagonal to 1 if symmetric
        if is_symmetric:
            np.fill_diagonal(kernel, 1)

        # get indices to calculate
        if is_symmetric:
            mus, nus = np.triu_indices(x_vec.shape[0], k=1)  # remove diagonal
        else:
            mus, nus = np.indices((x_vec.shape[0], y_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        is_statevector_sim = False
        measurement = not is_statevector_sim
        measurement_basis = "0" * self._n_qbits

        for idx in range(0, len(mus), self._batch_size):
            to_be_computed_data_pair = []
            to_be_computed_index = []
            for sub_idx in range(idx, min(idx + self._batch_size, len(mus))):
                i = mus[sub_idx]
                j = nus[sub_idx]
                x_i = x_vec[i]
                y_j = y_vec[j]
                if not np.all(x_i == y_j):
                    to_be_computed_data_pair.append((x_i, y_j))
                    to_be_computed_index.append((i, j))

            matrix_elements = []
            for x, y in to_be_computed_data_pair:
                result = _run_circuit(self._n_qbits, x, y)
                try:
                    counts = result[measurement_basis]
                    states = np.sum(list(result.values()))
                    probability = counts / states
                except:
                    probability = 0.0001
                matrix_elements.append(probability)

            for (i, j), value in zip(to_be_computed_index, matrix_elements):
                kernel[i, j] = value
                if is_symmetric:
                    kernel[j, i] = kernel[i, j]

        if is_symmetric:
            D, U = np.linalg.eig(kernel)
            kernel = U @ np.diag(np.maximum(0, D)) @ U.transpose()

        return kernel
