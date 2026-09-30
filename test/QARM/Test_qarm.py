import sys
from pathlib import Path
from pyqpanda_alg.QARM import QuantumAssociationRulesMining
from pyqpanda_alg import QARM
import os
import pytest

sys.path.append(Path.cwd().parent.parent.__str__())


def read(file_path):
    if os.path.exists(file_path):
        trans_data = []
        with open(file_path, 'r', encoding='utf8') as f:
            data_line = f.readlines()
            if data_line:
                for line in data_line:
                    if line:
                        data_list = line.strip().split(',')
                        trans_data.append([data.strip() for data in data_list])
            else:
                raise ValueError("The file {} has no any data!".format(file_path))
    else:
        raise FileNotFoundError('The file {} does not exists!'.format(file_path))
    return trans_data


def test_qarm_normal_case():
    data_path = QARM.__path__[0]
    data_file = os.path.join(data_path, 'dataset/data2.txt')
    trans_data = read(data_file)
    support = 0.2
    conf = 0.5
    qarm = QuantumAssociationRulesMining(trans_data, support, conf)
    result = qarm.run()

    assert result is not None
    assert isinstance(result, dict)

