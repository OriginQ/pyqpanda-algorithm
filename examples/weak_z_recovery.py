"""显式使用SDK模拟器的示例；同时保留成功和失败的测量结果。"""
import json
from pyqpanda3.core import CPUQVM,QProg,H,S,CNOT,measure
from pyqpanda_alg.QRecovery import weak_z_recovery_block


def example(shots=256):
    p=QProg(4)
    p<<H(0)<<CNOT(0,1)
    p<<weak_z_recovery_block(0,2,3,first_cbit=0,recovery_cbit=1,strength=.8)
    p<<H(0)<<S(0)
    # 操作者侧的测试读出，不属于恢复操作或其输入。
    p<<S(0).dagger()<<H(0)<<CNOT(0,1)<<H(0)
    p<<measure(0,2)<<measure(1,3)
    simulator=CPUQVM();simulator.run(p,shots)
    counts=simulator.result().get_counts()
    success=sum(n for bits,n in counts.items() if bits[-2]=='0')
    wrong=sum(n for bits,n in counts.items() if bits[-2]=='0' and bits[:2]!='00')
    return {'counts':counts,'shots':shots,'heralded_successes':success,
            'successful_outputs_failing_joint_readout':wrong,
            'bit_order':'c3(reference), c2(data), c1(recovery), c0(first)',
            'scope':'SDK software simulator only, not QPU; failures retained'}


if __name__=='__main__':
    result=example()
    if result['successful_outputs_failing_joint_readout']:
        raise RuntimeError('heralded output failed joint reference check')
    print(json.dumps(result,indent=2))
