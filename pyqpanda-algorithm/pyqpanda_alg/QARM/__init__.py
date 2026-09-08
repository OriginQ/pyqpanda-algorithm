'''
Association rule mining finds interesting associations or correlations between item sets in
a large amount of data . Mining the implicit relationship between objects from large-scale
data is called association analysis or association rule learning, which can reveal the hidden
association pattern in data and help people to carry out market operation and decision support.
For example, on the same trip to the supermarket, if the customer buys milk, what is the
likelihood that he will also buy bread?

Based on the famous classical association rule mining algorithm Apriori algorithm,
a quantum association rule mining algorithm is proposed to realize the core task.
Specifically, given a quantum black box accessing a trading database,
the algorithm first uses the quantum parallel amplitude estimation algorithm
to estimate the support of all candidate K term sets in a quantum parallel manner,
and store it in a quantum superposition state.  Next, the quantum amplitude
amplification algorithm is used to search the candidate K term sets that are not
less than the predetermined threshold from the superposition quantum states.
'''

from .complete_rules import QuantumAssociationRulesMining

__all__ = ['QuantumAssociationRulesMining']