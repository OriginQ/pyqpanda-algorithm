def test_qmrmr_star_exports_feature_selection():
    namespace = {}
    exec("from pyqpanda_alg.QmRMR import *", namespace)
    assert "Feature_Selection" in namespace
