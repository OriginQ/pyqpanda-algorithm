'''
The QECClosedForm module provides tools for closed-form prediction of
quantum error-correction code parameters (AG complete-code family):
code parameters, encoding rate, loss scaling, zero-loss boundary,
logical-operator counting and detection rate — no circuit, no simulation.
'''

from .QECClosedForm import QECClosedForm

__all__ = ["QECClosedForm"]
