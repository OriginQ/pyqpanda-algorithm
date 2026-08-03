import sympy as sp
from pyqpanda3.core import QProg

from pyqpanda_alg import QUBO


if __name__ == '__main__':
    x0, x1, x2 = sp.symbols('x0 x1 x2')
    function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
    test0 = QUBO.QuadraticBinary(function)
    n_key, n_res = test0.query_qnumber()
    print(n_key, n_res)

    q_all = QProg(n_key + n_res).qubits()
    q_key = q_all[:n_key]
    q_res = q_all[n_key:]

    print(test0.cir(q_key, q_res))

    # calculate the quadratic function value above with x0, x1, x2= 0, 1, 0
    print(test0.function_value([0, 1, 0]))

    # find the minimum function value by traversing
    res0 = test0.qubobytraversal()
    print('result of traversal: ', res0)

    # find the minimum function value using GAS
    test1 = QUBO.QUBO_GAS_origin(function)
    res1 = test1.run(init_value=0, continue_times=10, process_show=False)
    print('result of Grover adaptive search: ', res1)

    # find the minimum function value using QAOA
    test2 = QUBO.QUBO_QAOA(function)
    res2 = test2.run(layer=5, optimizer='SLSQP',
                     optimizer_option={'options': {'eps': 1e-3}})
    print('result of QAOA: ', res2)
