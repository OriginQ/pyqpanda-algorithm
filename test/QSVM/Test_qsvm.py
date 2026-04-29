# -*-coding:utf-8-*-
import sys
from pathlib import Path
import pytest

sys.path.append((Path.cwd().parent.parent).__str__())

import os
import numpy as np
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


def test_data_loading():
    train_features, test_features, train_labels, test_labels, samples = _read_vqc_qsvm_data(data_path)

    assert train_features.shape[1] == 2
    assert test_features.shape[1] == 2
    assert len(train_labels.shape) == 1
    assert len(test_labels.shape) == 1

    assert np.all(
        train_labels == [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.all(test_labels == [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])

