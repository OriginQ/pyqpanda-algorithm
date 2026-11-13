import math
import pytest
from pyqpanda3.core import QCircuit, RY, Z, X
from pyqpanda_alg.Grover import mark_data_reflection
from pyqpanda_alg.QuantumCounting import QuantumCounting
from pyqpanda_alg.QuantumCounting.decoding import decode_phase_distribution
from pyqpanda_alg.QCount import QuantumCount

@pytest.mark.parametrize('n',[1,2,3,4])
def test_all_counts_native(n):
    # Exhaustive cardinalities, including genuine no-mark/full-mark oracles.
    for m in range((1<<n)+1):
        marked=[format((i*5)%(1<<n),f'0{n}b') for i in range(m)]
        result=QuantumCounting(n,phase_bits=n+2,mark_data=marked).run()
        assert result.rounded_count==m
        assert result.estimated_count==result.estimated_probability*(1<<n)
        assert result.count_available
        assert result.phase==int(result.phase_state,2)/(1<<(n+2))
        assert 0<=result.phase_probability<=result.pair_probability<=1

@pytest.mark.parametrize('probability',[0,.125,.25,.5,.75,1])
def test_nonuniform_probability_is_not_cardinality(probability):
    def prep(q): return QCircuit()<<RY(q[0],2*math.asin(math.sqrt(probability)))
    def oracle(q): return QCircuit()<<Z(q[0])
    result=QuantumCounting(1,phase_bits=8,flip_operator=oracle,in_operator=prep).run()
    assert abs(result.estimated_probability-probability)<math.pi/256
    assert result.estimated_count is None and result.rounded_count is None
    assert not result.count_available
    compat=QuantumCount(1,oracle,precision_bits=8,in_operator=prep).run()
    assert compat.marked_count is None and compat.estimated_count is None
    assert compat.marked_fraction==result.estimated_probability

def test_compatibility_uses_same_engine_and_custom_zero_reflection():
    def oracle(q): return mark_data_reflection(q,['01'])
    def zero(q): return mark_data_reflection(q,['00'])
    old_api=QuantumCount(2,oracle,precision_bits=6,zero_flip=zero)
    new_api=QuantumCounting(2,phase_bits=6,flip_operator=oracle,zero_flip=zero)
    a,b=old_api.run(),new_api.run()
    assert a.marked_count==b.rounded_count==1
    assert a.marked_fraction==b.estimated_probability
    _,search,phase=old_api.build_program()
    assert search==[0,1] and phase==list(range(2,8))

def test_oracle_constructed_once_per_build():
    calls=[]
    def oracle(q):
        calls.append(tuple(q))
        return mark_data_reflection(q,['101','111'])
    counter=QuantumCounting(3,phase_bits=8,flip_operator=oracle)
    counter.circuit()
    assert calls==[(0,1,2)]

def test_mirrors_pool_without_counting_endpoints_twice():
    result=decode_phase_distribution({'001':3,'111':3,'010':5})
    assert result.phase==1/8 and result.pair_probability==pytest.approx(6/11)
    assert result.phase_probability==pytest.approx(3/11)
    for state,probability in [('000',0),('100',1)]:
        result=decode_phase_distribution({state:1e308})
        assert result.probability==probability and result.pair_probability==1

@pytest.mark.parametrize('weights',[{}, {'001':0}, {'001':-1}, {'001':math.inf},
    {'001':math.nan}, {'001':True}, {'001':1,'01':1}, {'x':1}])
def test_invalid_histograms(weights):
    with pytest.raises(ValueError): decode_phase_distribution(weights)

@pytest.mark.parametrize('kwargs',[{'qnumber':True},{'phase_bits':True},{'shots':True},
    {'phase_bits':21},{'flip_operator':7,'mark_data':None},{'in_operator':7},
    {'mark_data':['00','00']},{'mark_data':[['0','0']]},{'mark_data':['2']}])
def test_invalid_counter_inputs(kwargs):
    args=dict(qnumber=2,mark_data=['00']);args.update(kwargs)
    with pytest.raises((ValueError,TypeError)): QuantumCounting(**args)


def test_original_result_constructors_and_preparation_attribute():
    from pyqpanda_alg.QuantumCounting import QuantumCountingResult
    from pyqpanda_alg.QCount import QuantumCountResult
    first=QuantumCountingResult(2.0,2,.25,'01',.5,8)
    second=QuantumCountResult(2,.25,.25,'01',.5,1024)
    assert first.pair_probability is None and second.pair_probability is None
    assert first.count_available and second.count_available
    def oracle(q): return QCircuit()<<Z(q[0])
    counter=QuantumCount(1,oracle)
    assert isinstance(counter.in_operator([0]),QCircuit)
    assert counter._counter(10).count_available
    counter.in_operator=lambda q: QCircuit()<<RY(q[0],math.pi/3)
    assert not counter._counter(10).count_available
