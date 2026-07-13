'''
A Quantum Optimization Algorithm Set, based on pyqpanda.

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
'''
pyqpanda-algorithm Python\n
Copyright (C) Origin Quantum 2017-2023\n
Licensed Under Apache Licence 2.0
'''

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from . import QAOA
# from . import VQE
# from . import HHL
# from . import QAOA
from . import QARM
from . import QKmeans
# from . import QLuoShu
from . import QPCA
# from . import QSolver
from . import QSVM
# from . import extensions
# import warnings

#QFinance
from . import QUBO
from . import QCmp
from . import QAE
from . import QSVD
from . import QSVR
from . import Grover
from . import QmRMR
from . import QSEncode
from . import QKD

