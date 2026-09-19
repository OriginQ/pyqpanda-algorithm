"""功能范围有限的 PyQPanda 弱 Z 测量与逆转程序块，输出带有成功标志。

可追加到现有程序中，其数据态可以未知或处于纠缠状态。两个辅助量子比特
必须均为新制备的 |0> 态。两个结果比特都显式保留。本程序块从不重新初始化
数据，也不选择模拟器或读取概率。
"""
from math import acos, isfinite, sqrt
from pyqpanda3.core import CNOT, QProg, RY, measure


def _diagonal(program,data,auxiliary,d0,d1):
    t0,t1=2*acos(d0),2*acos(d1)
    program << RY(auxiliary,(t0+t1)/2)
    program << CNOT(data,auxiliary)
    program << RY(auxiliary,(t0-t1)/2)
    program << CNOT(data,auxiliary)


def weak_z_recovery_block(data,first_auxiliary,recovery_auxiliary,*,
                          first_cbit,recovery_cbit,strength=.8):
    """返回含测量的 QProg 程序块；recovery_cbit==0 是恢复成功的标志。

    输出数据仍可使用。在硬件噪声下，任何标志都不构成保证。
    首个测量后的辅助量子比特会被保留，并用作计算基下的控制比特；在理想
    模型中，这等价于根据其记录结果进行经典反馈。在两次用作受控非门的
    控制比特之前，绝不会将它重置。返回的程序块不含态制备指令、后端执行
    或隐式后选择。两个辅助量子比特都必须从 |0> 态开始；经典比特必须彼此
    不同，且归调用方所有。仅支持 0.5 < strength < 1。完整的概率和失败行为
    见 docs/weak_z_recovery.md；失败后，不可将数据当作新制备的数据重复执行。
    """
    qubits=(data,first_auxiliary,recovery_auxiliary)
    if any(type(q) is not int or q<0 for q in qubits) or len(set(qubits))!=3:
        raise ValueError('three distinct nonnegative qubit indices required')
    if any(type(c) is not int or c<0 for c in (first_cbit,recovery_cbit)) or first_cbit==recovery_cbit:
        raise ValueError('two distinct nonnegative classical bits required')
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not isfinite(strength) or not .5<strength<1:
        raise ValueError('strength must be finite and strictly between 0.5 and 1')
    program=QProg(max(qubits)+1)
    _diagonal(program,data,first_auxiliary,sqrt(strength),sqrt(1-strength))
    program << measure(first_auxiliary,first_cbit)
    # 测量所得标志决定衰减实际测量后状态中的哪个分量；
    # 两个 CX 门共同确保返回前恢复原始数据基。
    program << CNOT(first_auxiliary,data)
    _diagonal(program,data,recovery_auxiliary,sqrt((1-strength)/strength),1.)
    program << CNOT(first_auxiliary,data)
    program << measure(recovery_auxiliary,recovery_cbit)
    return program
