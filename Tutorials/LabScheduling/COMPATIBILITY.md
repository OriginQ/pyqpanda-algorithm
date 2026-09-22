# Python 安装兼容性

推荐 Python 3.12。2026-09-21 在 Linux x86_64 上补充了 3.11 和 3.13 的
独立 wheel 安装验证，核心代码和量子参数未修改。
下表及机器记录保留该兼容性轮次的提交范围；后续报告预算修复的验证结果见
[VALIDATION.md](VALIDATION.md)，不把历史安装指纹更新为未经重放的新指纹。

| Python | 依赖约束文件 | 应用测试 | 原有算法测试 | 核心覆盖率 |
|---|---|---:|---:|---:|
| 3.11.16 | `constraints-py311.txt` | 108 通过 | 18 通过 | 98.92% |
| 3.12.3 | `constraints-py312.txt` | 108 通过 | 18 通过 | 98.92% |
| 3.13.11 | `constraints-py312.txt` | 108 通过 | 18 通过 | 98.92% |

3.12 的完整发布验证见 [VALIDATION.md](VALIDATION.md)。3.11 / 3.13 各使用干净
worktree 内新建的 `.venv`，安装同一应用 wheel；确认导入来自 `site-packages`，
11 个核心模块与源码字节一致。依赖版本、模块及 wheel 指纹、命令、退出码和测试
摘要见 [compatibility.json](results/compatibility.json)。两个新增环境均通过
`pip check`，安装版本与对应约束文件逐项一致。

## 选择对应依赖

默认安装命令继续使用[主教程](README.md#1-三分钟演示)中的 Python 3.12 路线。
Python 3.11 请在仓库根目录运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install \
  -c pyqpanda-algorithm/example/LabScheduling/constraints-py311.txt \
  -e ./pyqpanda-algorithm \
  -r pyqpanda-algorithm/example/LabScheduling/requirements-dev.txt
```

Python 3.13 使用 `python3.13 -m venv .venv`，随后执行主教程中的安装命令；
`constraints-py312.txt` 的文件名保留历史含义，当前固定版本也在 3.13 上验证通过。
比较多个 Python 版本时，应使用不同 checkout 的 `.venv`，不要覆盖已有环境。

3.11 无法使用原 3.12 约束文件：实际安装预检以 `ResolutionImpossible` 失败。
下列四项固定依赖的原版本要求 Python >=3.12；新文件仅调整这四项，其余锁定
版本保持一致，未修改仓库全局依赖要求。

| 依赖 | 3.11 固定版本 | 3.12 / 3.13 固定版本 |
|---|---|---|
| NumPy | 2.4.6 | 2.5.3 |
| SciPy | 1.17.1 | 1.18.1 |
| ContourPy | 1.3.3 | 1.4.0 |
| Sphinx | 9.0.4 | 9.1.0 |

PyQPanda3 均为 0.4.1；其发布文件含 CPython 3.11 / 3.12 / 3.13 的 Linux x86_64
wheel，可在 [PyPI 发布文件](https://pypi.org/project/pyqpanda3/0.4.1/#files)核对。
wheel 存在或包元数据允许安装本身不代表应用运行成功，上表记录的是实际运行结果。

## 验证范围与失败记录

两个新增环境均执行：medium（目标 4）、含 21 次权重敏感性运行的取舍演示、
AC-3 资源边界案例（目标 3）、不启用 AC-3 的超限案例（退出码 5），以及不可行
案例（退出码 3）。可行案例的实际量子结果经原始约束与经典证书验证。

源码可编辑安装使用主教程的测试命令。验证已安装 wheel 的覆盖率时，应指定
安装目录，避免测到 checkout 中未被导入的副本。例如在 Python 3.11 中：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python -m pytest -c test/LabScheduling/pytest.ini test/LabScheduling \
  --timeout=60 --cov=.venv/lib/python3.11/site-packages/pyqpanda_alg/LabScheduling \
  --cov-fail-under=95
```

本次首次用 `--cov=pyqpanda_alg.LabScheduling` 在两个新增 wheel 环境均出现
`ImportError: cannot load module more than once per process`，各有 9 个收集错误。
改为上述实际安装目录后完整通过；归档保留该失败，不把首次运行算作通过。
尚未确定该工具链行为的上游根因，不据此提交算法缺陷修复或修改业务代码。

CI 已配置 Ubuntu 的 3.11 / 3.12 / 3.13 矩阵，包含安装、质量检查、应用测试、
18 项原有算法测试、成功演示与失败状态检查。远程实际结果单列于 VALIDATION.md；
本地通过不代表官方 PR CI 已获批准。

验证范围限于 Linux x86_64、标准 GIL 构建和上述依赖版本；未验证 Windows、
macOS、ARM 或自由线程 Python。跨版本通过表示接口与业务性质成立，不表示不同
依赖环境会产生完全相同的优化参数、counts 或耗时；严格历史重放仍应使用归档
对应的提交与固定环境。
