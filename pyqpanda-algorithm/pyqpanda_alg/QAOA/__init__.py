'''
A Quantum Optimization Algorithm Set, based on pyqpanda3.

The **QMengJi** is a collection of fundamental quantum optimization algorithms and
functions that are commonly used in developer's optimization problems.

The QMengJi provides standardized set of tools and building blocks for writing quantum programs.

Some key functions included in the QMengJi are:

Quantum Approximate Optimization Algorithm : This is a quantum algorithm that can be used to search the lowest energy
eigenstate of a Hamiltonian. It is considered to be one of the candidate algorithms for quantum advantage.
Developer can construct object by QAOA(problem, arg1, arg2…) and directly call QAOA.run(args) to optimize their
user-defined problem.

Overall, the QMengJi provides a standardized set of tools for developers, allowing them to optimize functions with
binary variables, such as combinatorial optimization problems, that may have many practical implications.
Developers can use quantum algorithms directly to obtain results without knowing anything about quantum computing, or
build their own suitable quantum circuits to obtain more customized results according to their requirements.It is an 
important resource for solving optimization problems and advancing research on quantum optimization algorithms.

'''
from . import default_circuits
from . import qaoa
from . import dstate
from . import spsa
from .lightcone import QAOALightCone


__all__ = ['qaoa', 'dstate', 'spsa', 'default_circuits', 'QAOALightCone']

