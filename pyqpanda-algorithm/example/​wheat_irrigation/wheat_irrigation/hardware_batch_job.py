"""Submit several prepared circuits as ONE native OriginQ cloud batch job.

This uses QCloudBackend.run(list[QProg], shots, options), whereas hardware_batch
orchestrates separate single-circuit jobs. Collection never submits a new job.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import time

from . import hardware
from .hardware_batch import _credential, _lock, _now, _plan, _save


MANIFEST = 'native_batch.json'
TERMINAL = {'completed_probabilities', 'failed'}


def validated_plans(root, batch):
    """Rebuild and check all circuits in the saved submission order."""
    if batch.get('submission_mode') != 'platform_native_batch':
        raise ValueError('Expected a native batch manifest')
    entries = batch['circuits']
    names = [entry['scenario'] for entry in entries]
    if len(entries) < 2 or len(set(names)) != len(names):
        raise ValueError('A native batch needs at least two distinct scenario circuits')
    if names != batch['configuration']['scenarios']:
        raise ValueError('Circuit order differs from saved configuration')
    for index, entry in enumerate(entries):
        if entry['index'] != index:
            raise ValueError('Circuit indices are not in submission order')
    hardware._chip_name(batch['configuration']['chip'])
    hardware._positive_integer(batch['configuration']['shots'], 'shots')
    return [_plan(root, batch, entry) for entry in entries]


def prepare_native_batch(root, source):
    path = root / MANIFEST
    if path.exists():
        batch = json.loads(path.read_text(encoding='utf-8'))
        if batch['source_directory'] != str(source.resolve()):
            raise ValueError('Existing native batch uses a different source')
        validated_plans(root, batch)
        return batch
    source_batch = json.loads((source / 'batch.json').read_text(encoding='utf-8'))
    names = source_batch['configuration']['scenarios']
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('Source needs at least two distinct scenarios')
    entries, plans = [], []
    for index, name in enumerate(names):
        source_entry = next(item for item in source_batch['jobs'] if item['scenario'] == name)
        plans.append(_plan(source, source_batch, source_entry))
        entries.append({'index': index, 'scenario': name,
                        'circuit_sha256': source_entry['circuit_sha256']})
    config = dict(source_batch['configuration'])
    config['repeats'] = 1
    batch = {'schema_version': 1, 'submission_mode': 'platform_native_batch',
             'sdk_submission': 'QCloudBackend.run(list[QProg], shots, options)',
             'created_at': _now(), 'source_directory': str(source.resolve()),
             'configuration': config, 'circuits': entries,
             'state': 'prepared', 'job_id': None,
             'options': {'mapping': True, 'optimization': True,
                         'amend': False, 'is_prob_counts': True}}
    for entry, plan in zip(entries, plans):
        _save(root / 'plans' / entry['scenario'] / 'hardware_job.json', plan)
    validated_plans(root, batch)
    _save(path, batch)
    write_report(root, batch)
    return batch


def submit_native_batch(root, batch):
    if batch.get('job_id'):
        return
    if batch['state'] != 'prepared':
        raise ValueError('Submission already attempted without a saved ID; inspect cloud history')
    plans = validated_plans(root, batch)
    sdk = hardware._load_sdk()
    # Parse every circuit before authentication or starting the submission.
    programs = [hardware._parse_originir(plan['circuit']['originir']) for plan in plans]
    service = hardware._service(sdk, hardware.TOKEN_ENV_VAR)
    chip = batch['configuration']['chip']
    try:
        if service.backends().get(chip) is not True:
            raise hardware.HardwareError('Selected chip is unavailable')
        backend = service.backend(chip)
        if len(set(backend.chip_info().available_qubits())) < max(p['circuit']['n_qubits'] for p in plans):
            raise hardware.HardwareError('Selected chip has too few available qubits')
    except hardware.HardwareError:
        raise
    except Exception:
        raise hardware.HardwareError('Cloud batch preflight failed') from None
    options = sdk.QCloudOptions()
    options.set_mapping(True)
    options.set_optimization(True)
    options.set_amend(False)
    options.set_is_prob_counts(True)
    batch.update(state='submitting', submission_attempted_at=_now(),
                 submitted_circuit_count=len(programs), submission_call_count=1)
    _save(root / MANIFEST, batch)
    try:
        # Exactly one cloud request; list overload creates a native batch job.
        job = backend.run(programs, batch['configuration']['shots'], options)
        job_id = hardware._identifier(job.job_id(), 'job_id')
    except BaseException as exc:
        batch.update(state='submission_unknown', last_submission_exception_type=type(exc).__name__)
        _save(root / MANIFEST, batch)
        if not isinstance(exc, Exception):
            raise
        raise hardware.HardwareError('Unknown submission outcome; do not resubmit') from None
    batch.update(job_id=job_id, state='submitted', submitted_at=_now())
    try:
        _save(root / MANIFEST, batch)
    except BaseException:
        print(f'Cloud batch exists: {job_id}. Saving its ID failed; retain this ID and do not resubmit.',
              file=sys.stderr, flush=True)
        raise
    print(f'Submitted ONE native batch: {job_id}; circuits={len(programs)}', flush=True)
    write_report(root, batch)


def collect_once(root, batch):
    from .hardware_batch_query import query_cloud_response
    from .hardware_native_batch_analysis import analyze_native_batch_results

    if not batch.get('job_id'):
        raise ValueError('No submitted native batch ID exists')
    if batch['state'] in TERMINAL:
        return
    plans = validated_plans(root, batch)
    response = query_cloud_response(batch['job_id'])
    obj = response['obj']
    if obj.get('chipId') != batch['configuration']['chip']:
        raise ValueError('Response chip differs from the saved batch')
    _save(root / 'cloud_response.json', response)
    batch.update(last_queried_at=_now(), cloud_task_state=str(obj.get('taskState')),
                 platform_task_type=obj.get('taskType'), platform_task_cal_method=obj.get('taskCalMethod'))
    if str(obj.get('taskState')) == '4' or obj.get('errCode', 0) not in (None, 0):
        batch['state'] = 'failed'
    elif str(obj.get('taskState')) == '3':
        result = analyze_native_batch_results(plans, response, batch['job_id'],
                                               [entry['scenario'] for entry in batch['circuits']])
        _save(root / 'result_analysis.json', result)
        for circuit in result['circuit_results']:
            _save(root / 'results' / circuit['scenario_name'] / 'result_analysis.json', circuit['analysis'])
        batch.update(state='completed_probabilities', collected_at=_now(), result_count=result['result_count'])
        batch.pop('last_collection_exception_type', None)
        print(f"Completed native batch {batch['job_id']}; results={result['result_count']}", flush=True)
    else:
        batch['state'] = 'pending'
        print(f"Pending native batch {batch['job_id']}; cloud state={batch['cloud_task_state']}", flush=True)
    _save(root / MANIFEST, batch)
    write_report(root, batch)


def collect_batch(root, batch, timeout, interval):
    deadline = time.monotonic() + timeout
    while True:
        try:
            collect_once(root, batch)
        except Exception as exc:
            batch['last_collection_exception_type'] = type(exc).__name__
            _save(root / MANIFEST, batch)
            print(f'Collection deferred: {type(exc).__name__}; saved job ID will be reused.', flush=True)
        if batch['state'] in TERMINAL or time.monotonic() >= deadline:
            return
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def write_report(root, batch):
    lines = ['# 本源量子原生批量真机任务', '',
             f"父任务 ID：`{batch.get('job_id') or '尚未提交'}`；状态：`{batch['state']}`。",
             f"芯片：`{batch['configuration']['chip']}`；{len(batch['circuits'])} 条电路，每条请求 {batch['configuration']['shots']} shots。", '',
             '通过一次 `QCloudBackend.run(list[QProg], shots, options)` 提交一个云端任务。',
             f"云端原始 taskType：`{batch.get('platform_task_type')}`；taskCalMethod：`{batch.get('platform_task_cal_method')}`。", '',
             '电路与返回结果按 SDK 列表顺序对应，索引从 0 开始；电路 SHA-256 保存在 native_batch.json。']
    if batch['state'] == 'completed_probabilities':
        data = json.loads((root / 'result_analysis.json').read_text(encoding='utf-8'))
        lines += ['', '| 索引 | 场景 | 最佳可行目标 | 经典最优 | 预算可行概率 | TVD |',
                  '| ---: | --- | ---: | ---: | ---: | ---: |']
        for item in data['circuit_results']:
            analysis = item['analysis']
            best = analysis['best_observed_budget_feasible_candidate']
            value = f"{best['objective']:.6f}" if best else '无可行候选'
            lines.append(f"| {item['result_index']} | {item['scenario_name']} | {value} | "
                         f"{analysis['exact_physical_objective']:.6f} | "
                         f"{analysis['normalized_export_metrics']['budget_feasible_probability']:.4%} | "
                         f"{analysis['total_variation_distance_normalized_export_vs_ideal']:.6f} |")
        lines += ['', '平台返回截断概率；表中概率经归一化。未获得原始整数计数，未由概率反推或补造 counts。',
                  '最佳可行目标采用原始测量决策，预算可行与完整 QUBO 最优是不同指标。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'submit', 'collect', 'run', 'status'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, default=Path('/tmp/wheat-qcloud-api-key'))
    parser.add_argument('--sdk-path', type=Path)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--poll-interval', type=float, default=10)
    args = parser.parse_args(argv)
    if args.mode in {'prepare', 'run'} and args.source is None:
        parser.error('--source is required for prepare/run')
    hardware._poll_settings(args.timeout, args.poll_interval)
    if args.sdk_path:
        sys.path.insert(0, str(args.sdk_path.resolve()))
    root = args.output.resolve()
    previous_token = os.environ.get(hardware.TOKEN_ENV_VAR)
    try:
        with _lock(root):
            if args.mode in {'prepare', 'run'}:
                batch = prepare_native_batch(root, args.source.resolve())
            else:
                batch = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
                validated_plans(root, batch)
            if args.mode in {'submit', 'collect', 'run'}:
                os.environ[hardware.TOKEN_ENV_VAR] = _credential(args.key_file)
                if 'execution_environment' not in batch:
                    try:
                        sdk_version = version('pyqpanda3')
                    except PackageNotFoundError:
                        sdk_version = None
                    batch['execution_environment'] = {'python': platform.python_version(),
                                                      'platform': platform.platform(), 'pyqpanda3': sdk_version}
                    _save(root / MANIFEST, batch)
            if args.mode in {'submit', 'run'}:
                submit_native_batch(root, batch)
            if args.mode in {'collect', 'run'}:
                collect_batch(root, batch, args.timeout, args.poll_interval)
            write_report(root, batch)
            print(f"Native batch: {batch['state']}; output={root}", flush=True)
            return 2 if batch['state'] in {'failed', 'submission_unknown'} else 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Native batch stopped: {type(exc).__name__}; inspect saved status before retrying.', file=sys.stderr)
        return 2
    finally:
        if previous_token is None:
            os.environ.pop(hardware.TOKEN_ENV_VAR, None)
        else:
            os.environ[hardware.TOKEN_ENV_VAR] = previous_token


if __name__ == '__main__':
    raise SystemExit(main())
