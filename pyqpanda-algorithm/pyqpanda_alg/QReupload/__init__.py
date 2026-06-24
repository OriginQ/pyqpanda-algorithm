# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QReupload: data re-uploading variational quantum learning for pyqpanda3.

Adds the first pyqpanda-algorithm ML estimator whose circuit parameters are
trained against a supervised task loss (sklearn-style fit); the existing
QSVM/QSVR fit only a classical SVM over a parameter-free quantum feature map.
Provides a data re-uploading model with trainable input scaling, an optional
entangling ring, exact adjoint-gradient training, and encoding-expressivity /
barren-plateau diagnostics.
"""
from .qreupload import QReuploadRegressor, QReuploadClassifier
from .expressivity import fourier_spectrum, trainability_scan

__all__ = [
    "QReuploadRegressor",
    "QReuploadClassifier",
    "fourier_spectrum",
    "trainability_scan",
]
