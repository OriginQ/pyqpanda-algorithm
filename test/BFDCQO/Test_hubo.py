import importlib.util
import itertools
import numpy as np
import pytest
from scipy.linalg import expm

from pyqpanda_alg.BFDCQO import (
    PolynomialIsingProblem, binary_hubo_to_ising, binary_hubo_energy,
    ExactSolver, BFDCQO, QAOASolver,
)
from pyqpanda_alg.BFDCQO.core.hamiltonian import build_initial_hamiltonian, build_problem_hamiltonian
from pyqpanda_alg.BFDCQO.core.agp import first_order_gauge_potential
from pyqpanda_alg.BFDCQO.core.pauli import matrix, pauli_string
from pyqpanda_alg.BFDCQO.backends.numpy_statevector import apply_pauli_rotation
from pyqpanda_alg.BFDCQO.backends.pyqpanda3_backend import pauli_rotation_recipe, pauli_rotation_resources


def _validation_problem(n: int, degree: int, seed: int, name: str) -> PolynomialIsingProblem:
    """Create a deterministic mixed-order problem used only by this test module."""
    rng = np.random.default_rng(seed)
    terms = {(i,): float(rng.uniform(-0.6, 0.6)) for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.55:
                terms[(i, j)] = float(rng.uniform(-0.8, 0.8))
    candidates = list(itertools.combinations(range(n), degree))
    rng.shuffle(candidates)
    for term in candidates[: max(2, min(5, len(candidates)))]:
        terms[term] = float(rng.uniform(-1.0, 1.0))
    return PolynomialIsingProblem(n=n, terms=terms, name=name)


def cubic5_validation() -> PolynomialIsingProblem:
    return _validation_problem(5, 3, 9, "cubic5")


def quartic6_validation() -> PolynomialIsingProblem:
    return _validation_problem(6, 4, 3, "quartic6")


def test_binary_hubo_to_ising_preserves_every_basis_energy():
    n=4
    terms={(0,):0.7,(1,2):-1.1,(0,1,3):1.8,(0,1,2,3):-0.9}
    offset=0.31
    p=binary_hubo_to_ising(n,terms,offset=offset,name="binary4")
    assert p.degree==4
    for idx in range(1<<n):
        bits=[(idx>>q)&1 for q in range(n)]
        direct=binary_hubo_energy(n,terms,bits,offset=offset)
        assert np.isclose(direct,p.energy_index(idx),atol=1e-12)


def test_polynomial_ising_diagonal_matches_pauli_matrix_all_states():
    p=PolynomialIsingProblem(4,{(0,):.2,(1,3):-.4,(0,1,2):.9,(0,1,2,3):-.3},offset=.17)
    H=matrix(build_problem_hamiltonian(p)) + p.offset*np.eye(1<<p.n)
    assert np.allclose(H,np.diag(np.diag(H)),atol=1e-13)
    for idx in range(1<<p.n):
        assert np.isclose(H[idx,idx].real,p.energy_index(idx),atol=1e-12)


def test_hubo_first_order_agp_matches_dense_commutators():
    p=PolynomialIsingProblem(4,{(0,):.31,(1,2):-.44,(0,1,2):.82,(0,1,2,3):-.57})
    bias=np.array([.15,-.22,.09,.31])
    hi=build_initial_hamiltonian(p.n,bias,-1.0); hf=build_problem_hamiltonian(p); lam=.43
    g=first_order_gauge_potential(hi,hf,lam)
    Hi,Hf=matrix(hi),matrix(hf); O1=Hi@Hf-Hf@Hi; Had=(1-lam)*Hi+lam*Hf; O2=Had@O1-O1@Had
    alpha=-np.trace(O1.conj().T@O1).real/np.trace(O2.conj().T@O2).real
    assert np.isclose(g.alpha1,alpha,rtol=1e-12,atol=1e-12)
    # The native AGP must contain higher-weight strings, not a quadratized surrogate.
    assert max(sum(x!="I" for x in ps) for ps in g.A1)>=4


def _single_gate_matrix(n,q,U):
    mats=[]
    I=np.eye(2,dtype=complex)
    for k in reversed(range(n)): mats.append(U if k==q else I)
    out=np.array([[1.+0j]])
    for m in mats: out=np.kron(out,m)
    return out


def _cnot_matrix(n,c,t):
    U=np.zeros((1<<n,1<<n),dtype=complex)
    for i in range(1<<n):
        j=i ^ ((1<<t) if ((i>>c)&1) else 0)
        U[j,i]=1
    return U


def _recipe_matrix(n,recipe):
    X=np.array([[0,1],[1,0]],complex); Z=np.diag([1,-1]).astype(complex); H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
    U=np.eye(1<<n,dtype=complex)
    for op in recipe:
        if op[0]=="H": G=_single_gate_matrix(n,op[1],H)
        elif op[0]=="RX":
            a=op[2]; R=np.cos(a/2)*np.eye(2)-1j*np.sin(a/2)*X; G=_single_gate_matrix(n,op[1],R)
        elif op[0]=="RZ":
            a=op[2]; R=np.cos(a/2)*np.eye(2)-1j*np.sin(a/2)*Z; G=_single_gate_matrix(n,op[1],R)
        elif op[0]=="CNOT": G=_cnot_matrix(n,op[1],op[2])
        else: raise AssertionError(op)
        U=G@U
    return U


@pytest.mark.parametrize("ops",[
    {0:"Y",1:"Z",3:"Z"},
    {0:"X",1:"Y",2:"Y",3:"Z"},
])
def test_k_body_pyqpanda_rotation_recipe_matches_dense_exponential(ops):
    n=4; theta=.371; ps=pauli_string(n,ops); P=matrix({ps:1.0})
    want=expm(-1j*theta*P); got=_recipe_matrix(n,pauli_rotation_recipe(ps,theta))
    assert np.allclose(got,want,atol=1e-12)
    res=pauli_rotation_resources(ps); w=len(ops)
    assert res["cnot"]==2*(w-1)


def test_exact_solver_accepts_native_hubo():
    p=quartic6_validation(); r=ExactSolver().solve(p)
    e,states=p.exact_ground_states()
    assert np.isclose(r.best_energy,e)
    assert r.best_solutions==states
    assert r.states_evaluated==64


def test_native_cubic_and_quartic_bfdcqo_regression():
    for p,min_factor in [(cubic5_validation(),2.0),(quartic6_validation(),2.5)]:
        r=BFDCQO(p,n_trotter=3,dt=.1,iterations=8,shots=4000,seed=0).solve()
        assert r.success
        p0=r.history[0].ground_state_probability; pf=r.history[-1].ground_state_probability
        assert pf>p0*min_factor
        assert r.best_solution in r.exact_ground_states


def test_qaoa_explicitly_rejects_native_hubo_without_quadratization():
    with pytest.raises(TypeError,match="quadratic"):
        QAOASolver(cubic5_validation(),p=1)

@pytest.mark.skipif(importlib.util.find_spec("pyqpanda3") is None, reason="pyqpanda3 not installed")
def test_pyqpanda_native_hubo_matches_numpy_reference():
    p=cubic5_validation(); kwargs=dict(n_trotter=2,dt=.1,iterations=1,shots=None,seed=0)
    a=BFDCQO(p,backend="numpy",**kwargs).solve(); b=BFDCQO(p,backend="pyqpanda",**kwargs).solve()
    assert np.isclose(a.history[0].expected_energy,b.history[0].expected_energy,atol=1e-8)
    assert np.isclose(a.history[0].ground_state_probability,b.history[0].ground_state_probability,atol=1e-8)
