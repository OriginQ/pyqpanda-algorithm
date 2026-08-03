from pyqpanda3.core import CPUQVM, QProg

from pyqpanda_alg.Grover import Grover, mark_data_reflection, iter_num, iter_analysis


if __name__ == '__main__':
    m = CPUQVM()
    q_state = QProg(3).qubits()

    def mark(qubits):
        return mark_data_reflection(qubits=qubits, mark_data=['101', '001'])

    demo_search = Grover(flip_operator=mark)
    best_iter = iter_num(q_num=len(q_state), sol_num=2)
    print('best iter num: ', best_iter)
    prob, angle = iter_analysis(q_num=len(q_state), sol_num=2, iternum=best_iter)
    print('prob for getting one of the solution with given iter num:', prob)

    prog = QProg()
    prog << demo_search.cir(q_input=q_state)

    m.run(prog, 1000)
    res = m.result().get_prob_dict(q_state)
    print(res)
