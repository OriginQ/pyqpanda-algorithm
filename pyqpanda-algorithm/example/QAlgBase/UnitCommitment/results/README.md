# Runtime FakeBackend result

This record is a local, finite-shot rehearsal of the same measured circuit
used by the case5 workflow.  It uses the OriginQ Runtime `FakeBackend` and the
`WK_C180` calibration snapshot; no remote quantum task or real-QPU job was
submitted.  The API key was read from the local environment and its value was
not written to the result.

The classical MILP/LP reference cost is 28439.3074.  With 256 shots, the
Runtime FakeBackend produced a capacity-feasible probability of 0.859375 and
selected a feasible schedule with cost 28453.3074, a 0.0492% gap to the
reference.  The two-qubit Bell smoke check had even-parity probability 0.6875.

The machine-readable values are in [summary.json](summary.json); the cost
comparison figure is [case5_runtime_fakebackend_costs.png](case5_runtime_fakebackend_costs.png).
These values document a local calibration-derived rehearsal, not a real-device
accuracy claim.

