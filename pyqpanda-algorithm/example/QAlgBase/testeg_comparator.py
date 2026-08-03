from pyqpanda3.core import CPUQVM, QCircuit, QProg, H, X

from pyqpanda_alg import QCmp


if __name__ == '__main__':
    # 整数比较
    value = 3
    m = CPUQVM()
    q_state = [0, 1]
    q_anc_cmp = [2, 3]
    prog = QProg()
    cir = QCircuit()
    for q in q_state:
        cir << H(q)
    cir << QCmp.int_comparator(value, q_state, q_anc_cmp, function='g', reuse=True)
    prog << cir
    m.run(prog, 1000)
    print(m.result().get_prob_dict([q_anc_cmp[-1]]))

    # 插值方法
    value = 3.3
    m = CPUQVM()
    q_state = [0, 1, 2]
    q_anc_cmp = [3, 4, 5]
    prog = QProg()
    cir = QCircuit()
    for q in q_state[:2]:
        cir << X(q)
    cir << QCmp.interpolation_comparator(value, q_state, q_anc_cmp, function='g', reuse=True)
    prog << cir
    m.run(prog, 1000)
    print(m.result().get_prob_dict([q_anc_cmp[-1]]))

    # 两个态比较，示例中叠加态的0，1，2，3有0.5的概率大于态1
    m = CPUQVM()
    q_state_1 = [0, 1]
    q_state_2 = [2, 3]
    q_anc_cmp = [4, 5]
    prog = QProg()
    cir = QCircuit()
    for q in q_state_1:
        cir << H(q)
    cir << X(q_state_2[0])
    cir << QCmp.qubit_comparator(q_state_1, q_state_2, q_anc_cmp, function='g')
    prog << cir
    m.run(prog, 1000)
    print(m.result().get_prob_dict([q_anc_cmp[-1]]))
