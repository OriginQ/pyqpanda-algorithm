import numpy as np
import matplotlib.pyplot as plt
from pyqpanda_alg.QSEncode import QSpare_Code
def t01():
    mu = 0
    sigma = 1
    x = np.linspace(-3, 3, 2 ** 10)
    pdf_normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    ini = pdf_normal / np.linalg.norm(pdf_normal)
    res = QSpare_Code(ini ** 2, mode='fourier', cut_length=20).Quantum_Res()
    # res = QSpare_Code(ini ** 2, mode='fourier', cut_length=20).Quantum_Res()

    plt.plot(x, res)
    plt.plot(x, ini ** 2)
    plt.show()

if __name__ == '__main__':
    t01()
