"""独立SDK线路参考；读取内部态仅限测试。"""
import numpy as np
import pytest
from pyqpanda3.core import CPUQVM,QProg,X
from pyqpanda_alg.QRecovery import weak_z_recovery_block


@pytest.mark.parametrize('s',[.6,.8,.95])
def test_complete_measured_maps_including_failed_branches(s):
    # 可以延后第一次Z测量，是因为后续门只把该标志位用作控制，
    # 不再对该标志位施加任何旋转。
    measured=weak_z_recovery_block(0,1,2,first_cbit=0,recovery_cbit=1,strength=s)
    columns=[]
    for bit in [0,1]:
        p=QProg(3)
        if bit:p<<X(0)
        for g in measured.gate_operations():p<<g
        cpu=CPUQVM();cpu.run(p,1);columns.append(np.array(cpu.result().get_state_vector()))
    for first in [0,1]:
        for recovery in [0,1]:
            k=np.column_stack([v[[4*recovery+2*first,4*recovery+2*first+1]] for v in columns])
            if recovery==0:expected=np.sqrt(1-s)*np.eye(2)
            else:expected=np.diag([np.sqrt(2*s-1),0]) if first==0 else np.diag([0,np.sqrt(2*s-1)])
            np.testing.assert_allclose(k,expected,atol=2e-12)


@pytest.mark.parametrize('args',[(0,0,2),(0,1,1),(-1,1,2),(True,1,2)])
def test_reject_overlapping_or_invalid_registers(args):
    with pytest.raises(ValueError):weak_z_recovery_block(*args,first_cbit=0,recovery_cbit=1)


def test_flag_bits_must_be_separate():
    with pytest.raises(ValueError):weak_z_recovery_block(0,1,2,first_cbit=0,recovery_cbit=0)


@pytest.mark.parametrize('s',[0,.5,1,-.1,1.1,True,'0.8',complex(.8),float('nan'),float('inf')])
def test_invalid_strength(s):
    with pytest.raises(ValueError):
        weak_z_recovery_block(0,1,2,first_cbit=0,recovery_cbit=1,strength=s)


@pytest.mark.parametrize('cbits',[(-1,1),(0,-1),(True,1),(0,1.5)])
def test_invalid_classical_indices(cbits):
    with pytest.raises(ValueError):
        weak_z_recovery_block(0,1,2,first_cbit=cbits[0],recovery_cbit=cbits[1])
