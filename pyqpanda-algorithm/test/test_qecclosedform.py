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

"""QECClosedForm 单元测试：闭式 vs 已发布的精确值（10.30/10.35）。"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pyqpanda_alg.QECClosedForm import QECClosedForm


def test_code_parameters():
    """[[16,6,4]] = CSS(RM(1,4)) 参数闭式。"""
    cf = QECClosedForm(4, 1)
    assert cf.code() == (16, 6, 4)


def test_large_code():
    """[[1024,672,16]]：m=10, r=3。"""
    cf = QECClosedForm(10, 3)
    assert cf.code() == (1024, 672, 16)


def test_encoding_rate():
    """[[1024,1002,4]] 编码率 = 0.978516。"""
    cf = QECClosedForm(10, 1)
    assert abs(cf.encoding_rate() - 0.978516) < 1e-5


def test_fail_w0_15_7_3():
    """[[16,6,4]] 的 fail(w0) = 0.875（与精确枚举一致）。"""
    cf = QECClosedForm(4, 1)
    assert abs(cf.fail - 0.875) < 1e-6


def test_loss_closed_form():
    """[[16,6,4]]: loss(0.01) = 3e-08（c_d = 3, d = 4）。"""
    cf = QECClosedForm(4, 1)
    assert abs(cf.loss(0.01) - 3e-8) < 1e-10
    # [[1024,672,16]]: loss(0.01) ≈ 1.05e-24
    cf2 = QECClosedForm(10, 3)
    assert abs(cf2.loss(0.01) - 1.05e-24) < 1e-26


def test_zero_loss_boundary():
    """零损失边界 k_max = ⌊(d-1)/2⌋：d=16 → 7。"""
    cf = QECClosedForm(10, 3)
    assert cf.zero_loss_boundary() == 7


def test_logical_operator_count():
    """逻辑算符计数（定理 10.30.2.04）: RM(1,5) → 1240, RM(1,6) → 10416。"""
    assert QECClosedForm(5, 1).logical_operator_count() == 1240
    assert QECClosedForm(6, 1).logical_operator_count() == 10416


def test_detection_rate():
    """检测率闭式 sin²(θ/2)（10.29 预言 2a）。"""
    assert abs(QECClosedForm.detection_rate(0.1) - math.sin(0.05)**2) < 1e-12


def test_invalid_params():
    """非正逻辑比特数应报错。"""
    with pytest.raises(ValueError):
        QECClosedForm(5, 2)  # k = 32 - 2·16 = 0
