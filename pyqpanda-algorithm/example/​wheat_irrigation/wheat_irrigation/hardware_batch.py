"""Prepare, submit once, and collect a bounded batch of real QCloud jobs.

The compatibility collector preserves the cloud's probability results without
manufacturing counts. A submission attempt is persisted before cloud access;
an ambiguous attempt without a job ID is never retried automatically.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import time

from . import hardware
from .cli import main as run_cli
from .model import IrrigationInstance, build_qubo


SCENARIOS = ('jointing', 'heading', 'grain_filling')
TERMINAL = {'completed_probabilities', 'failed'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def _lock(root):
    # Advisory OS locks release automatically if a process is interrupted.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is operating on this batch') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _plan(root, batch, entry):
    config = batch['configuration']
    path = root / 'plans' / entry['scenario'] / 'hardware_job.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    report = plan['report']
    for key in ('shots', 'depth', 'maxiter', 'restarts', 'seed'):
        if report['configuration'][key] != config[key]:
            raise ValueError('Prepared circuit configuration differs from batch configuration')
    if report['hardware']['chip'] != config['chip'] or report['configuration']['scenario'] != entry['scenario']:
        raise ValueError('Prepared circuit belongs to a different chip or scenario')
    model = build_qubo(IrrigationInstance.from_dict(report['input_data']))
    angles = report['qaoa_reference']['parameters']
    rebuilt = hardware.prepare_qaoa_circuit(model, angles['gammas'], angles['betas'])
    if rebuilt != plan['circuit'] or rebuilt['sha256'] != entry['circuit_sha256']:
        raise ValueError('Prepared circuit integrity check failed')
    return plan


def prepare_batch(root, configuration):
    path = root / 'batch.json'
    if path.exists():
        batch = json.loads(path.read_text(encoding='utf-8'))
        if batch['configuration'] != configuration:
            raise ValueError('Existing batch has different settings; use its saved settings or a new directory')
    else:
        batch = {'schema_version': 1, 'created_at': _now(), 'configuration': configuration, 'jobs': []}
        _save(path, batch)
    for scenario in configuration['scenarios']:
        output = root / 'plans' / scenario
        manifest = output / 'hardware_job.json'
        if not manifest.exists():
            args = ['--backend', 'qcloud', '--qcloud-chip', configuration['chip'],
                    '--qcloud-dry-run', '--scenario', scenario, '--no-plot', '--output', str(output)]
            for flag in ('depth', 'shots', 'maxiter', 'restarts', 'seed'):
                args += ['--' + flag, str(configuration[flag])]
            if run_cli(args) not in (0, None):
                raise ValueError('Offline preparation failed')
        plan = json.loads(manifest.read_text(encoding='utf-8'))
        for repeat in range(1, configuration['repeats'] + 1):
            name = f'{scenario}_{repeat:02d}'
            entry = next((item for item in batch['jobs'] if item['name'] == name), None)
            if entry is None:
                entry = {'name': name, 'scenario': scenario, 'repeat': repeat,
                         'state': 'prepared', 'job_id': None,
                         'circuit_sha256': plan['circuit']['sha256']}
                batch['jobs'].append(entry)
                _save(path, batch)
            _plan(root, batch, entry)
    _save(path, batch)
    write_summary(root, batch)
    return batch


def submit_jobs(root, batch):
    if any(not item.get('job_id') and item['state'] in {'submitting', 'submission_unknown'} for item in batch['jobs']):
        raise ValueError('A previous submission has no saved ID; inspect cloud task history before any retry')
    pending = [item for item in batch['jobs'] if not item.get('job_id') and item['state'] == 'prepared']
    if not pending:
        return
    chip = batch['configuration']['chip']
    if hardware.list_hardware_backends().get(chip) is not True:
        raise hardware.HardwareError('Selected batch chip is not currently available')
    # Verify every circuit before any submission consumes machine time.
    plans = {entry['name']: _plan(root, batch, entry) for entry in pending}
    for entry in pending:
        entry.update(state='submitting', submission_attempted_at=_now())
        _save(root / 'batch.json', batch)

        def submitted(job_id):
            entry.update(job_id=job_id, state='submitted', submitted_at=_now())
            try:
                _save(root / 'batch.json', batch)
            except BaseException:
                print(f"Cloud task {job_id} exists but saving its ID failed; retain this ID and do not resubmit.",
                      file=sys.stderr, flush=True)
                raise
            print(f"Submitted {entry['name']}: {job_id}", flush=True)

        try:
            result = hardware.run_hardware(
                plans[entry['name']]['circuit'], chip=chip, shots=batch['configuration']['shots'],
                timeout=0, on_submitted=submitted,
            )
            _save(root / 'jobs' / entry['name'] / 'initial_sdk_result.json', result)
            entry['state'] = 'failed' if result['status'] == 'failed' else 'submitted'
        except Exception as exc:
            entry['last_submission_exception_type'] = type(exc).__name__
            if not entry.get('job_id'):
                entry['state'] = 'submission_unknown'
                _save(root / 'batch.json', batch)
                write_summary(root, batch)
                raise hardware.HardwareError('Submission outcome is unknown; inspect cloud history before retrying') from None
            # The ID is durable. Result parsing errors must never cause a new job.
            entry['state'] = 'submitted'
        _save(root / 'batch.json', batch)
        write_summary(root, batch)


def collect_once(root, batch):
    from .hardware_batch_analysis import analyze_probability_result
    from .hardware_batch_query import query_cloud_response

    for entry in batch['jobs']:
        if not entry.get('job_id') or entry['state'] in TERMINAL:
            continue
        output = root / 'jobs' / entry['name']
        try:
            plan = _plan(root, batch, entry)
            response = query_cloud_response(entry['job_id'])
            obj = response['obj']
            if obj.get('chipId') != batch['configuration']['chip']:
                raise ValueError('Cloud response chip does not match this batch')
            _save(output / 'cloud_response.json', response)
            entry['last_queried_at'] = _now()
            entry['cloud_task_state'] = str(obj.get('taskState'))
            if str(obj.get('taskState')) == '4' or obj.get('errCode', 0) not in (None, 0):
                entry['state'] = 'failed'
            elif str(obj.get('taskState')) == '3':
                analysis = analyze_probability_result(plan, response, entry['job_id'])
                _save(output / 'result_analysis.json', analysis)
                entry.update(state='completed_probabilities', collected_at=_now())
                entry.pop('last_collection_exception_type', None)
                print(f"Completed {entry['name']}: {entry['job_id']}", flush=True)
            else:
                entry['state'] = 'pending'
                print(f"Pending {entry['name']}: cloud state {entry['cloud_task_state']}", flush=True)
        except Exception as exc:
            entry['last_collection_exception_type'] = type(exc).__name__
            print(f"Collection deferred {entry['name']}: {type(exc).__name__}", flush=True)
        _save(root / 'batch.json', batch)
        write_summary(root, batch)


def collect_jobs(root, batch, timeout, interval):
    deadline = time.monotonic() + timeout
    while True:
        collect_once(root, batch)
        remaining = [entry for entry in batch['jobs'] if entry.get('job_id') and entry['state'] not in TERMINAL]
        if not remaining or time.monotonic() >= deadline:
            return
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def write_summary(root, batch):
    rows = []
    for entry in batch['jobs']:
        row = {key: entry.get(key) for key in ('name', 'scenario', 'repeat', 'state', 'job_id')}
        row['requested_shots'] = batch['configuration']['shots']
        result = root / 'jobs' / entry['name'] / 'result_analysis.json'
        if entry['state'] == 'completed_probabilities' and result.exists():
            data = json.loads(result.read_text(encoding='utf-8'))
            best = data['best_observed_budget_feasible_candidate']
            row.update(
                best_objective=best['objective'] if best else None,
                exact_objective=data['exact_physical_objective'],
                relative_gap=best['relative_gap_to_exact'] if best else None,
                feasible_probability=data['normalized_export_metrics']['budget_feasible_probability'],
                raw_feasible_probability=data['raw_export_weighted_metrics']['budget_feasible_probability'],
                ideal_feasible_probability=data['ideal_metrics']['budget_feasible_probability'],
                tvd=data['total_variation_distance_normalized_export_vs_ideal'],
                unique_observed_states=data['unique_observed_states'],
            )
        rows.append(row)
    summary = {'updated_at': _now(), 'configuration': batch['configuration'],
               'execution_environment': batch.get('execution_environment'),
               'planned_jobs': len(rows), 'submitted_jobs': sum(bool(row['job_id']) for row in rows),
               'completed_jobs': sum(row['state'] == 'completed_probabilities' for row in rows),
               'failed_jobs': sum(row['state'] == 'failed' for row in rows),
               'counts_synthesized': False, 'runs': rows}
    _save(root / 'batch_summary.json', summary)
    columns = ['name', 'scenario', 'repeat', 'state', 'job_id', 'requested_shots', 'best_objective',
               'exact_objective', 'relative_gap', 'feasible_probability', 'raw_feasible_probability',
               'ideal_feasible_probability', 'tvd', 'unique_observed_states']
    with (root / 'batch_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 小麦灌溉批量真机测试', '',
             f"芯片：`{batch['configuration']['chip']}`；每次请求 {batch['configuration']['shots']} 次测量。",
             f"计划 {len(rows)} 个任务，已提交 {summary['submitted_jobs']}，已完成 {summary['completed_jobs']}，失败 {summary['failed_jobs']}。", '',
             '| 场景/重复 | 状态 | 最佳目标 | 经典最优 | 预算可行率 | 理想可行率 | TVD |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if 'feasible_probability' in row:
            best = f"{row['best_objective']:.6f}" if row['best_objective'] is not None else '无可行候选'
            cells = [row['name'], row['state'], best, f"{row['exact_objective']:.6f}",
                     f"{100*row['feasible_probability']:.4f}%", f"{100*row['ideal_feasible_probability']:.4f}%", f"{row['tvd']:.6f}"]
        else:
            cells = [row['name'], row['state'], '—', '—', '—', '—', '—']
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '表中真机可行率和 TVD 使用按导出概率总和归一化的值；原始概率和归一化记录均保留。没有从概率合成整数计数。重复实验共享该场景的本地优化参数，仅重新进行独立硬件测量。', '',
              '## 任务证据', '']
    for row in rows:
        lines.append(f"- {row['name']}：`{row['job_id'] or '尚未提交'}`；状态 `{row['state']}`。")
        if row['state'] == 'completed_probabilities':
            lines.append(f"  [结果分析](jobs/{row['name']}/result_analysis.json) · [认证云端响应](jobs/{row['name']}/cloud_response.json)")
    lines += ['', '任务 ID 保存后只续查，不重复提交；无 ID 的不确定提交会阻止自动重试。当前兼容查询路径从旧版 SDK 的认证响应中提取原始概率，区别于要求整数计数的原单任务管线。', '',
              '单次或少量重复实验不能证明量子优势。农业参数为合成数据，结果不代表实际节水或产量效果。', '',
              '[汇总 JSON](batch_summary.json) · [汇总 CSV](batch_summary.csv) · [批量任务清单](batch.json)']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _credential(key_file):
    token = os.environ.get(hardware.TOKEN_ENV_VAR, '').strip() or os.environ.get('OPENAI_API_KEY', '').strip()
    if not token and key_file.is_file():
        token = key_file.read_text(encoding='utf-8').strip()
    if token.startswith(('export ', 'OPENAI_API_KEY=', 'QCLOUD_API_KEY=')):
        parts = shlex.split(token)
        if parts and parts[0] == 'export':
            parts = parts[1:]
        if len(parts) != 1:
            raise ValueError('Credential must contain a single value or environment assignment')
        name, separator, value = parts[0].partition('=')
        if separator and name in {'OPENAI_API_KEY', 'QCLOUD_API_KEY'}:
            token = value.strip()
        elif not separator:
            token = parts[0]
        else:
            raise ValueError('Unsupported credential assignment')
    if not token:
        raise ValueError('No cloud credential is available')
    return token


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'submit', 'collect', 'run', 'status'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chip', default='WK_C180')
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--maxiter', type=int, default=100)
    parser.add_argument('--restarts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--poll-interval', type=float, default=10)
    parser.add_argument('--key-file', type=Path, default=Path('/tmp/wheat-qcloud-api-key'))
    parser.add_argument('--sdk-path', type=Path)
    args = parser.parse_args(argv)
    if any(getattr(args, key) <= 0 for key in ('repeats', 'shots', 'depth', 'maxiter', 'restarts')):
        parser.error('repeats/shots/depth/maxiter/restarts must be positive')
    if not math.isfinite(args.timeout) or args.timeout < 0 or not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error('timeout must be finite and nonnegative; poll interval must be finite and positive')
    if len(args.scenarios) != len(set(args.scenarios)):
        parser.error('scenario names must be unique')
    if args.sdk_path:
        sys.path.insert(0, str(args.sdk_path.resolve()))
    root = args.output.resolve()
    config = {key: getattr(args, key) for key in ('chip', 'scenarios', 'repeats', 'shots', 'depth', 'maxiter', 'restarts', 'seed')}
    previous_token = os.environ.get(hardware.TOKEN_ENV_VAR)
    try:
        with _lock(root):
            if args.mode in {'prepare', 'run'}:
                batch = prepare_batch(root, config)
            else:
                batch = json.loads((root / 'batch.json').read_text(encoding='utf-8'))
            if args.mode in {'submit', 'collect', 'run'}:
                os.environ[hardware.TOKEN_ENV_VAR] = _credential(args.key_file)
                try:
                    sdk_version = version('pyqpanda3')
                except PackageNotFoundError:
                    sdk_version = None
                batch['execution_environment'] = {
                    'python': platform.python_version(), 'platform': platform.platform(),
                    'pyqpanda3': sdk_version, 'collector': 'authenticated_sdk_response_compatibility',
                }
                _save(root / 'batch.json', batch)
            if args.mode in {'submit', 'run'}:
                submit_jobs(root, batch)
            if args.mode in {'collect', 'run'}:
                collect_jobs(root, batch, args.timeout, args.poll_interval)
            write_summary(root, batch)
            print(f"Batch results: {root}", flush=True)
            return 2 if any(entry['state'] in {'failed', 'submission_unknown'} for entry in batch['jobs']) else 0
    except (Exception, KeyboardInterrupt) as exc:
        # Do not expose SDK errors, HTTP headers, or credential file contents.
        print(f'Batch stopped: {type(exc).__name__}; inspect the saved batch status before retrying.', file=sys.stderr)
        return 2
    finally:
        if previous_token is None:
            os.environ.pop(hardware.TOKEN_ENV_VAR, None)
        else:
            os.environ[hardware.TOKEN_ENV_VAR] = previous_token


if __name__ == '__main__':
    raise SystemExit(main())
