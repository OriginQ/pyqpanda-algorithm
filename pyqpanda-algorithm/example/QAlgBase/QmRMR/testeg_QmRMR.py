import numpy as np
import warnings

from pyqpanda_alg.QmRMR.QmRMR_core import plot_bar, plot_loss, Feature_Selection
import os
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

def send_data(training_data, select_num):
    cor = np.corrcoef(training_data.T)
    cor = np.abs(cor)
    m = cor.shape[0]
    m -= 1
    a = cor[:m, :m]
    b = cor[:m, m:]
    linear = -b.ravel()
    quadratic = a / (select_num - 1)
    for i in range(m):
        quadratic[i][i] = 0
    if not isinstance(select_num, int) or select_num <= 1 or select_num >= len(linear):
        raise ValueError('The number of selected indicators should be a positive integer greater than 1 and smaller than the original number of features')
    tmp = Feature_Selection(quadratic, linear, select_num)
    qubits_num = len(linear)
    ini = np.random.random(
        int(int(qubits_num / 2) * (np.floor(qubits_num / 2) + np.floor((qubits_num - 1) / 2)))) * 4 * np.pi
    his, choice, dic = tmp.get_his_res(ini)
    l1 = tmp.l1
    l2 = tmp.l2
    plot_bar(dic)
    print([his, choice, dic])

    ll1 = []
    ll2 = []
    for i in range(len(l1)):
        if i % 3 == 0:
            ll1.append(l1[i])
            ll2.append(l2[i])

    plt.plot(his)
    plt.show()
    plt.clf()
    plt.plot(ll1, label='Correlation')
    plt.plot(ll2, label='Redundancy')
    plt.legend()
    plt.show()
    return his


if __name__ == '__main__':
    def load_csv_check_numeric_and_size(file_path):
        try:
            training_data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            return training_data
        except ValueError as e:
            print(f"Error loading file: {e}")
            return None


    file_path = os.path.join(os.path.dirname(__file__), 'traing.csv')
    training_data = load_csv_check_numeric_and_size(file_path)
    select_num = 3
    send_data(training_data, select_num)



