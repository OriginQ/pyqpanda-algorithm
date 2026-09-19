"""Yellow River wheat irrigation example using pyqpanda_alg QAOA.

Adapted from the original agriculture CLI with wheat scenarios, supply-water
accounting, portable package resources and a human-readable planning report.
"""

import argparse
import csv
from datetime import date, timedelta
from importlib.resources import files
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import __version__
from .model import IrrigationInstance, build_qubo, greedy_schedule
from .solvers import MAX_STATEVECTOR_QUBITS, enumerate_spectrum, solve_qaoa


SCENARIOS = {"jointing": "拔节期", "heading": "抽穗扬花期", "grain_filling": "灌浆期"}


def load_scenario(name):
    """Read installed package data without depending on the checkout layout."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown wheat scenario: {name}")
    resource = files("wheat_irrigation").joinpath("data", f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _day_date(instance, day):
    if instance.start_date is None:
        return ""
    return (date.fromisoformat(instance.start_date) + timedelta(days=day)).isoformat()


def _stage_label(stage):
    return SCENARIOS.get(stage, stage)


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _positive_float(text):
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _nonnegative_float(text):
    value = float(text)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return value


def _candidate(model, x, optimum):
    result = model.evaluate(x)
    result["decision_bits_x0_first"] = "".join(str(int(b)) for b in x)
    result["absolute_gap"] = max(0.0, result["objective"] - optimum)
    result["relative_gap"] = result["absolute_gap"] / abs(optimum) if abs(optimum) > 1e-12 else None
    result["events_by_zone"] = {
        z.name: [d + 1 for d, active in enumerate(result["schedule"][i]) if active]
        for i, z in enumerate(model.instance.zones)
    }
    return result


def summarize_distribution(model, bits, energies, probabilities, counts, optimum):
    """Score raw shots; select a budget-feasible schedule only among observed x.

    Slack is not repaired. Agricultural candidate selection uses H_obj(x),
    while raw QUBO metrics retain the observed slack penalty.
    """
    nx = model.n_decisions
    budget = model.instance.budget_events
    events = bits[:, :nx].sum(axis=1).astype(int)
    residual = events + bits[:, nx:] @ model.slack_weights - budget
    budget_feasible = events <= budget
    zero_residual = residual == 0
    optimal = np.isclose(energies, optimum, atol=1e-7, rtol=0)
    frequencies = np.zeros(len(bits), dtype=int)
    for index, count in counts.items():
        frequencies[index] = count
    shots = int(frequencies.sum())
    if shots <= 0:
        raise ValueError("At least one shot is required")
    observed = np.flatnonzero(frequencies)
    sample_distribution = frequencies / shots
    selected = None
    feasible_observed = observed[budget_feasible[observed]]
    if len(feasible_observed):
        # Remove the known penalty only to rank physical schedules, retaining
        # the raw state and residual in the output for auditability.
        physical = energies[feasible_observed] - model.budget_penalty * residual[feasible_observed] ** 2
        index = int(feasible_observed[np.argmin(physical)])
        selected = _candidate(model, bits[index, :nx], optimum)
        selected.update({
            "sampled_state_index": index,
            "sampled_full_bits_x0_first": "".join(str(int(b)) for b in bits[index]),
            "sampled_count": int(frequencies[index]),
            "sampled_slack_residual": int(residual[index]),
            "sampled_qubo_energy": float(energies[index]),
        })
    def stats(distribution):
        return {
            "mean_qubo_energy": float(distribution @ energies),
            "budget_feasible_probability": float(distribution[budget_feasible].sum()),
            "zero_slack_residual_probability": float(distribution[zero_residual].sum()),
            "optimal_qubo_probability": float(distribution[optimal].sum()),
        }
    return {
        "ideal": stats(probabilities), "sampled": stats(sample_distribution),
        "shots": shots, "best_observed_qubo_energy": float(energies[observed].min()),
        "best_budget_feasible_sample": selected,
        "counts_by_state_index": {str(i): int(frequencies[i]) for i in observed},
    }


def run_comparison(model, args):
    started = time.perf_counter()
    bits, energies = enumerate_spectrum(model, args.max_qubits)
    optimum = float(energies.min())
    minimum_indices = np.flatnonzero(np.isclose(energies, optimum, rtol=0, atol=1e-7))
    for index in minimum_indices:
        full = bits[index]
        residual = (int(full[:model.n_decisions].sum())
                    + int(full[model.n_decisions:] @ model.slack_weights) - model.instance.budget_events)
        if residual != 0:
            raise RuntimeError("The enumerated global minimum violates the certified budget encoding")
    exact = _candidate(model, bits[minimum_indices[0], :model.n_decisions], optimum)
    if not np.isclose(exact["objective"], optimum, rtol=0, atol=1e-6):
        raise RuntimeError("Physical and QUBO objectives disagree at the exact solution")
    solutions = {
        "Exact": exact,
        "Greedy": _candidate(model, greedy_schedule(model, args.depletion_fraction), optimum),
        "No irrigation": _candidate(model, np.zeros(model.n_decisions, dtype=int), optimum),
    }
    report = {
        "instance": model.instance.name,
        "source_note": model.instance.source_note,
        "application": "Yellow River Basin wheat irrigation",
        "version": __version__,
        "region": model.instance.region,
        "crop": model.instance.crop,
        "start_date": model.instance.start_date,
        "formulation_reference": "Wheat irrigation QUBO: linear soil-water balance, weighted moisture targets and water-supply budget; see docs/model.md",
        "execution": ("Classical enumeration and heuristic baselines" if args.backend == "classical"
                      else "Small-instance classical enumeration and ideal CPU quantum simulation; no QPU execution"),
        "configuration": {
            "backend": args.backend, "seed": args.seed, "depth": args.depth,
            "maxiter": args.maxiter, "restarts": args.restarts, "shots": args.shots,
            "depletion_fraction": args.depletion_fraction, "max_qubits": args.max_qubits,
        },
        "model": {
            "days": model.instance.days, "zones": len(model.instance.zones),
            "decision_variables": model.n_decisions, "slack_variables": len(model.slack_weights),
            "total_qubits": model.n_variables, "budget_events": model.instance.budget_events,
            "dose_mm": model.instance.dose_mm, "zone_area_ha": model.instance.zone_area_ha,
            "irrigation_efficiency": model.instance.irrigation_efficiency,
            "water_budget_m3": model.instance.water_budget_m3,
            "event_net_water_m3": model.instance.event_net_water_m3,
            "event_gross_water_m3": model.instance.event_gross_water_m3,
            "budget_net_water_m3": model.instance.budget_events * model.instance.event_net_water_m3,
            "effective_budget_gross_water_m3": model.instance.effective_budget_gross_water_m3,
            "penalty_bound": model.penalty_bound, "budget_penalty": model.budget_penalty,
            "slack_weights": model.slack_weights.tolist(),
            "constant": model.constant,
            "nonzero_ising_pairs": int(np.count_nonzero(model.quadratic)),
        },
        "exact": {
            "minimum_qubo_energy": optimum, "maximum_qubo_energy": float(energies.max()),
            "optimal_full_bitstrings": len(minimum_indices),
            "enumerated_states": len(bits),
        },
        "solutions": solutions,
    }
    report["software"] = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "matplotlib", "pyqpanda3", "pyqpanda_alg"):
        try:
            report["software"][package] = version(package)
        except PackageNotFoundError:
            report["software"][package] = None
    uniform = np.full(len(bits), 1.0 / len(bits))
    rng = np.random.default_rng(args.seed)
    sampled_uniform = rng.multinomial(args.shots, uniform)
    uniform_counts = {int(i): int(sampled_uniform[i]) for i in np.flatnonzero(sampled_uniform)}
    report["uniform_sampling"] = summarize_distribution(model, bits, energies, uniform, uniform_counts, optimum)
    if report["uniform_sampling"]["best_budget_feasible_sample"] is not None:
        solutions["Uniform"] = report["uniform_sampling"]["best_budget_feasible_sample"]
    if args.backend != "classical":
        local_backend = args.qcloud_local_backend if args.backend == "qcloud" else args.backend
        result = solve_qaoa(model, depth=args.depth, maxiter=args.maxiter, restarts=args.restarts,
                            shots=args.shots, seed=args.seed, backend=local_backend, max_qubits=args.max_qubits)
        qaoa = summarize_distribution(model, bits, energies, result["probabilities"], result["counts"], optimum)
        qaoa.update({key: result[key] for key in (
            "backend", "engine", "circuit_evaluations", "parameters", "normalization",
            "ising_offset", "evaluations", "optimizer_runs"
        )})
        uniform_popt = report["uniform_sampling"]["ideal"]["optimal_qubo_probability"]
        qaoa["ideal_optimum_enrichment_over_uniform"] = qaoa["ideal"]["optimal_qubo_probability"] / uniform_popt
        if args.backend == "qcloud":
            # Local samples must never be presented as hardware observations.
            report["qaoa_reference"] = {key: qaoa[key] for key in (
                "ideal", "backend", "engine", "circuit_evaluations", "parameters",
                "normalization", "ising_offset", "evaluations", "optimizer_runs",
                "ideal_optimum_enrichment_over_uniform",
            )}
            report["execution"] = "Local ideal QAOA optimization; QPU execution has not completed"
            report["configuration"]["qcloud_local_backend"] = local_backend
        else:
            report["qaoa"] = qaoa
            if qaoa["best_budget_feasible_sample"] is not None:
                solutions["QAOA"] = qaoa["best_budget_feasible_sample"]
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def export_results(model, report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if "input_data" in report:
        (output / "resolved_input.json").write_text(
            json.dumps(report["input_data"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if "forecast" in report:
        from .forecast_output import export_forecast
        export_forecast(report, output)
    h, J, offset = model.to_ising()
    variables = [{"index": i, "type": "irrigation", "zone": model.instance.zones[z].name, "day": d + 1}
                 for i, (z, d) in enumerate(model.decisions)]
    variables += [{"index": model.n_decisions + i, "type": "slack", "weight": int(w)}
                  for i, w in enumerate(model.slack_weights)]
    qubo = {
        "convention": "H(v) = constant + sum_i linear[i]*v[i] + sum_(i<j) quadratic[i,j]*v[i]*v[j]",
        "state_index_convention": "state_index = sum_i(v[i] * 2**i); displayed x0-first bits reverse conventional ket strings",
        "variables": variables, "linear": model.linear.tolist(), "constant": model.constant,
        "quadratic": [[int(i), int(j), float(model.quadratic[i, j])] for i, j in zip(*np.nonzero(model.quadratic))],
        "ising": {"h": h.tolist(), "offset": offset,
                  "J": [[int(i), int(j), float(J[i, j])] for i, j in zip(*np.nonzero(J))]},
    }
    (output / "qubo.json").write_text(json.dumps(qubo, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "schedules.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["solver", "zone", "day", "date", "growth_stage", "window_open", "irrigate",
                         "net_depth_mm", "net_water_m3", "gross_water_m3", "conveyance_loss_m3"])
        for name, solution in report["solutions"].items():
            for zi, zone in enumerate(model.instance.zones):
                for d, active in enumerate(solution["schedule"][zi]):
                    depth = active * model.instance.dose_mm
                    net = active * model.instance.event_net_water_m3
                    gross = active * model.instance.event_gross_water_m3
                    writer.writerow([name, zone.name, d + 1, _day_date(model.instance, d), zone.growth_stage[d],
                                     int(d + 1 in zone.window_days), active, depth, net, gross, gross - net])
    with (output / "moisture.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["solver", "zone", "day", "date", "growth_stage", "moisture_mm", "target_mm",
                         "taw_mm", "moisture_threshold_mm", "below_threshold", "outside_physical_range"])
        for name, solution in report["solutions"].items():
            for zi, zone in enumerate(model.instance.zones):
                for d, value in enumerate(solution["moisture_mm"][zi]):
                    writer.writerow([name, zone.name, d + 1, _day_date(model.instance, d), zone.growth_stage[d],
                                     value, zone.target_mm[d], zone.taw_mm, zone.moisture_threshold_mm[d],
                                     int(value < zone.moisture_threshold_mm[d]),
                                     int(value < 0 or value > zone.taw_mm)])
    export_planning_report(model, report, output)


def export_planning_report(model, report, output):
    """Summarize water allocation and diagnostics without a plotting dependency."""
    instance = model.instance
    lines = [
        "# 黄河流域小麦灌溉计算报告", "", f"场景：{_cell(instance.name)}", "",
        f"区域：{_cell(instance.region)}；作物：{_cell(instance.crop)}；规划：{instance.days} 天。", "",
        f"数据来源：{_cell(instance.source_note)}", "",
        f"取水总预算 {instance.water_budget_m3:.2f} m³，输配水效率 {instance.irrigation_efficiency:.1%}。"
        f"单次净灌水深 {instance.dose_mm:g} mm，单地块 {instance.zone_area_ha:g} ha，"
        f"每次取水 {instance.event_gross_water_m3:.2f} m³、根区净水 {instance.event_net_water_m3:.2f} m³。"
        f"固定剂量下允许至多 {instance.budget_events} 次事件。", "",
        "## 方法比较", "",
        "| 方法 | 加权目标值↓ | 渠首取水 m³ | 根区净水 m³ | 预算可行 | 低于水分阈值（区·天） | 水分越界（区·天） |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, result in report["solutions"].items():
        lines.append(f"| {_cell(name)} | {result['objective']:.3f} | {result['gross_water_m3']:.2f} | "
                     f"{result['net_water_m3']:.2f} | {'是' if result['budget_feasible'] else '否'} | "
                     f"{result['below_threshold_zone_days']} | {len(result['moisture_excursions'])} |")
    lines += ["", "目标值是水分偏差和管理成本的加权和，不能换算为产量或货币。低于阈值及水分越界属于诊断，未作为硬约束。", ""]
    if "forecast" in report:
        forecast = report["forecast"]
        evaluation = forecast["evaluation"]
        lines += ["## 预测—调度", "",
                  f"预测方法：{_cell(forecast['method'])}；历史截止 {evaluation['final_refit_end']}，"
                  f"回看 {forecast['lookback_days']} 天。保留期 {evaluation['holdout_start']} 至 {evaluation['holdout_end']}，"
                  "按时间顺序递归预测，评估后用全部规划前历史重新运算。", "",
                  f"预测数据来源：{_cell(forecast['source_note'])}", "",
                  "| 预测对象 | MAE | 持续性基线 MAE | RMSE | NSE | KGE |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        targets = [("ET₀ (mm/天)", forecast["et0"])]
        targets += [(name + " ψ (kPa)", value) for name, value in forecast["soil"].items()]
        for name, value in targets:
            metrics = value["holdout_metrics"]
            def fmt(number):
                return "未定义" if number is None else f"{number:.4f}"
            lines.append(f"| {_cell(name)} | {fmt(metrics['mae'])} | "
                         f"{fmt(value['persistence_holdout_metrics']['mae'])} | {fmt(metrics['rmse'])} | "
                         f"{fmt(metrics['nse'])} | {fmt(metrics['kge'])} |")
        clipped = sum(len(value["clipped_days"]) for value in forecast["soil"].values())
        lines += ["", "ET₀ 的历史评价参照是 FAO56 计算值，土壤评价参照是输入的日末水势；"
                  "保留期使用当时的气象与灌溉记录，不表示真实天气预报的端到端精度。", "",
                  "水势通过各地块标定表转换成无新增灌溉的可利用水量 B；调度采用 B 加累计净灌水。"
                  "该近似未建模灌溉引起的非线性下渗与反馈。", "",
                  "预测模式的水量调整项替换原物理基线：ET₀、雨量通过张力特征起作用；"
                  "单独修改未来 Kc、毛管上升或有效降雨比例后重新预测，会被调整项抵消，ETc 仅供参考。", "",
                  f"预测水势超出标定表范围的区·天数：{clipped}；这些值在转换前裁至表边界，原始值与标记见 forecast.csv。"
                  "调度后的水量轨迹仍不截断。", "",
                  "forecast.json 保存逐项指标、训练参数和预测；forecast_input.json 保存原始预测输入；"
                  "resolved_input.json 可直接用 --input 重放调度，无须再次训练。", ""]
    if "qaoa" in report:
        qaoa = report["qaoa"]
        stats = qaoa["sampled"]
        lines += [f"QAOA 执行引擎：`{qaoa['engine']}`；电路评估/任务数 {qaoa['circuit_evaluations']}。", "",
                  f"QAOA 共抽样 {qaoa['shots']} 次，预算可行率 {stats['budget_feasible_probability']:.2%}，"
                  f"完整 QUBO 最优态命中率 {stats['optimal_qubo_probability']:.2%}。", ""]
        if "QAOA" not in report["solutions"]:
            lines += ["本次 QAOA 未抽到预算可行方案；以下经典精确方案仅作为对照。", ""]
    if "hardware" in report:
        hardware = report["hardware"]
        reference = report["qaoa_reference"]
        lines += ["## 量子真机测试", "",
                  f"芯片：`{_cell(hardware['chip'])}`；状态：`{hardware['status']}`；"
                  f"任务编号：`{_cell(hardware.get('job_id') or '尚未提交')}`。", "",
                  f"本地参数优化使用 `{reference['engine']}`，评估 {reference['circuit_evaluations']} 次；"
                  "真机仅执行最终参数对应的测量电路。", "",
                  f"理想模拟预算可行率 {reference['ideal']['budget_feasible_probability']:.2%}，"
                  f"完整 QUBO 最优态概率 {reference['ideal']['optimal_qubo_probability']:.2%}。", ""]
        if hardware["status"] == "finished":
            lines += [f"真机原始计数共 {hardware['shots']} 次；读出修正关闭，未修复松弛位或重新抽样。", "",
                      f"真机频率与理想分布的总变差距离：{report['qaoa']['total_variation_distance']:.6f}"
                      "（同时含有限抽样误差和硬件噪声）。", ""]
        else:
            lines += ["尚无已完成的真机测量结果；本报告中的经典方案仅用于对照。"
                      "任务信息与电路保存在 hardware_job.json 和 qaoa.originir。", ""]
    for method in ("QAOA", "Exact"):
        if method not in report["solutions"]:
            continue
        result = report["solutions"][method]
        lines += [f"## {method} {'抽样候选' if method == 'QAOA' else '经典精确对照'}", "",
                  f"取水 {result['gross_water_m3']:.2f} m³，预算剩余 {result['remaining_water_m3']:.2f} m³，"
                  f"输配水损失 {result['conveyance_loss_m3']:.2f} m³。", "",
                  "| 地块 | 日期 / 规划日 | 生育期 | 净灌水深 mm | 取水 m³ |",
                  "| --- | --- | --- | ---: | ---: |"]
        for zi, zone in enumerate(instance.zones):
            active_days = [d for d, active in enumerate(result["schedule"][zi]) if active]
            if not active_days:
                lines.append(f"| {_cell(zone.name)} | 本周期未安排灌溉 | — | 0 | 0 |")
            for d in active_days:
                label = _day_date(instance, d)
                lines.append(f"| {_cell(zone.name)} | {label} / 第 {d + 1} 天 | "
                             f"{_cell(_stage_label(zone.growth_stage[d]))} | {instance.dose_mm:g} | "
                             f"{instance.event_gross_water_m3:.2f} |")
        lines.append("")
    lines += ["## 模型与运行边界", "",
              f"{model.n_decisions} 个灌溉决策 + {len(model.slack_weights)} 个预算松弛变量 = {model.n_variables} 比特。"
              "窗口以外禁止灌溉；相邻地块同日灌溉、同地块连续灌溉为软惩罚。", "",
              f"执行方式：{report['execution']}。本报告不证明量子优势。"
              "模型使用未截断的线性根区水量平衡，合成场景结果未经过田间验证。", "",
              "完整输入、参数、原始抽样和优化状态见 summary.json；逐日轨迹见 moisture.csv。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(model, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    panels = len(model.instance.zones) + 1
    rows = (panels + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(12, 3.5 * rows), squeeze=False, layout="constrained")
    axes = axes.flatten()
    solutions = report["solutions"]
    names = list(solutions)
    axes[0].bar(names, [solutions[n]["objective"] for n in names], color="#447ea2")
    axes[0].set_title("Physical objective (lower is better)", loc="left", weight="bold")
    axes[0].set_ylabel("Weighted cost, not currency")
    axes[0].tick_params(axis="x", rotation=20)
    colors = {"Exact": "#216b48", "Greedy": "#ca7932", "QAOA": "#6655b7"}
    if "forecast" in report:
        colors["No irrigation"] = "#9b5555"
    for zi, zone in enumerate(model.instance.zones):
        axis = axes[zi + 1]
        axis.axhspan(0, zone.taw_mm, color="#e4eee8", alpha=0.6)
        for name, color in colors.items():
            if name in solutions:
                values = [zone.initial_moisture_mm] + solutions[name]["moisture_mm"][zi]
                axis.plot(range(model.instance.days + 1), values, marker="o", markersize=3,
                          color=color, label=name, linestyle="--" if name == "QAOA" else "-")
        axis.plot(range(1, model.instance.days + 1), zone.target_mm, color="#404a50", linestyle=":", label="Target")
        axis.plot(range(1, model.instance.days + 1), zone.moisture_threshold_mm,
                  color="#b97033", linestyle="-.", linewidth=1, label="Depletion threshold")
        axis.axhline(zone.taw_mm, color="#718078", linewidth=0.8, linestyle="--", label="TAW")
        axis.axhline(0, color="#718078", linewidth=0.8)
        axis.set_title(f"{zone.name}: root-zone available water", loc="left", weight="bold")
        axis.set_xlabel("Day (0 = initial state)")
        axis.set_ylabel("Available water (mm)")
        axis.legend(fontsize=8, ncol=3)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[panels:]:
        axis.set_visible(False)
    mode = ("QPU: " + report["hardware"]["status"] if "hardware" in report else
            "classical baselines" if report["configuration"]["backend"] == "classical" else "ideal CPU simulation")
    figure.suptitle(f"Yellow River wheat irrigation | {mode}", fontsize=14, weight="bold")
    figure.savefig(output / "comparison.svg")
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description="黄河流域缺水地区小麦灌溉：QUBO / QAOA / 经典对照")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="自定义场景 JSON 路径")
    source.add_argument("--scenario", choices=list(SCENARIOS), default="jointing", help="内置小麦生育期场景，默认拔节期")
    parser.add_argument("--list-scenarios", action="store_true", help="列出内置合成场景并退出")
    parser.add_argument("--export-input", type=Path, help="将生效的输入保存为可修改的 JSON，同时继续求解")
    parser.add_argument("--water-budget-m3", type=float, help="覆盖渠首总取水预算（m³），含输配水损失")
    parser.add_argument("--efficiency", type=float, help="覆盖输配水效率，取值 (0,1]")
    forecasting = parser.add_mutually_exclusive_group()
    forecasting.add_argument("--forecast-input", type=Path, help="每日历史气象/水势、未来天气及土壤标定 JSON，先预测再调度")
    forecasting.add_argument("--forecast-demo", action="store_true", help="生成匹配场景的合成历史与未来天气，演示预测—调度")
    parser.add_argument("--forecast-model", choices=["ridge", "vqc"], default="ridge", help="ET₀/土壤预测模型，与调度 --backend 独立")
    parser.add_argument("--forecast-maxiter", type=_positive_int, default=30, help="VQC 预测训练迭代上限，与 QAOA --maxiter 独立")
    parser.add_argument("--output", type=Path, help="结果目录；默认当前目录 outputs/<场景名>")
    parser.add_argument("--backend", choices=["pyqpanda_alg", "numpy", "pyqpanda", "classical", "qcloud"],
                        default="pyqpanda_alg",
                        help="调度后端，默认 pyqpanda_alg；pyqpanda 是兼容别名")
    parser.add_argument("--depth", type=_positive_int, default=2)
    parser.add_argument("--maxiter", type=_positive_int, default=100, help="每个起点的 COBYLA 评估预算")
    parser.add_argument("--restarts", type=_positive_int, default=2, help="总起点数：一个线性初值及其余随机初值")
    parser.add_argument("--shots", type=_positive_int, default=4096, help="优化完成后的抽样次数")
    parser.add_argument("--seed", type=_nonnegative_int, default=7)
    parser.add_argument("--depletion-fraction", type=float, help="覆盖贪心基线的允许耗水比例，默认使用各地块生育期参数")
    parser.add_argument("--max-qubits", type=_positive_int, default=MAX_STATEVECTOR_QUBITS, help="枚举/态矢量上限，最大 18")
    parser.add_argument("--no-plot", action="store_true", help="导出 JSON/CSV/Markdown，不生成图表")
    cloud = parser.add_argument_group("本源 QCloud 真机测试：本地优化参数，真机测量最终电路")
    cloud.add_argument("--qcloud-chip", help="账户可用的真实芯片后端名称，不接受云模拟器")
    cloud.add_argument("--qcloud-local-backend", choices=["numpy", "pyqpanda_alg"], default="numpy",
                       help="真机测试前的本地参数优化后端，默认 numpy")
    cloud.add_argument("--qcloud-token-env", default="QCLOUD_API_KEY", help="保存 API Key 的环境变量名")
    cloud.add_argument("--qcloud-timeout", type=_nonnegative_float, default=300, help="轮询等待秒数；0 为仅查询一次，超时保存任务供续查")
    cloud.add_argument("--qcloud-poll-interval", type=_positive_float, default=5, help="任务查询间隔（秒）")
    action = cloud.add_mutually_exclusive_group()
    action.add_argument("--qcloud-dry-run", action="store_true", help="仅本地优化和导出电路，不连接云端")
    action.add_argument("--qcloud-resume", type=Path, help="读取 hardware_job.json 续查已提交任务，不重新提交")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.list_scenarios:
        for key, label in SCENARIOS.items():
            print(f"{key:<14} {label}｜黄河流域关中缺水麦田｜3 地块 × 7 天｜合成数据")
        return 0
    provided = {option.split("=", 1)[0] for option in argv if option.startswith("--")}
    if args.qcloud_resume:
        allowed = {"--qcloud-resume", "--backend", "--output", "--no-plot", "--qcloud-token-env",
                   "--qcloud-timeout", "--qcloud-poll-interval"}
        if provided - allowed or ("--backend" in provided and args.backend != "qcloud"):
            parser.error("续查使用已保存的场景、参数和 shots；只可覆盖输出、绘图、凭据变量及查询等待设置")
        args.backend = "qcloud"
        if args.output is None:
            args.output = args.qcloud_resume.parent
        try:
            from .hardware_workflow import resume_hardware_test
            model, report = resume_hardware_test(args)
            export_results(model, report, args.output)
            if not args.no_plot:
                plot_comparison(model, report, args.output)
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            parser.exit(2, f"错误：{exc}\n")
        return _print_results(report, args.output)
    if args.backend == "qcloud":
        if not args.qcloud_chip:
            parser.error("--backend qcloud 需要 --qcloud-chip 指定真实芯片")
    elif any(option.startswith("--qcloud-") for option in provided):
        parser.error("--qcloud-* 参数需要 --backend qcloud")
    if args.output is None:
        suffix = f"_forecast_{args.forecast_model}" if args.forecast_demo or args.forecast_input else ""
        if args.backend == "qcloud":
            suffix += "_qcloud_dry_run" if args.qcloud_dry_run else "_qcloud"
        args.output = Path.cwd() / "outputs" / ((args.input.stem if args.input else args.scenario) + suffix)
    try:
        if args.backend == "qcloud" and (args.output / "hardware_job.json").exists():
            raise ValueError("输出目录已有 hardware_job.json；请用 --qcloud-resume 续查，或选择新的 --output")
        input_snapshot = (json.loads(args.input.read_text(encoding="utf-8")) if args.input
                          else load_scenario(args.scenario))
        if not isinstance(input_snapshot, dict):
            raise ValueError("Input JSON must contain an object")
        if args.water_budget_m3 is not None:
            input_snapshot["water_budget_m3"] = args.water_budget_m3
        if args.efficiency is not None:
            input_snapshot["irrigation_efficiency"] = args.efficiency
        forecast_report = None
        if args.forecast_demo or args.forecast_input:
            from .forecasting import prepare_forecast
            if args.forecast_demo:
                from .forecast_demo import make_forecast_demo
                forecast_input = make_forecast_demo(input_snapshot, seed=args.seed)
            else:
                forecast_input = json.loads(args.forecast_input.read_text(encoding="utf-8"))
            scheduling_input = input_snapshot
            print(f"预测：{args.forecast_model}；按时间保留评估，再用规划前历史训练 ET₀ / 土壤水势模型。")
            input_snapshot, forecast_report = prepare_forecast(input_snapshot, forecast_input,
                                                              method=args.forecast_model, seed=args.seed,
                                                              maxiter=args.forecast_maxiter)
        model = build_qubo(IrrigationInstance.from_dict(input_snapshot))
        print(f"实例：{model.instance.name}\n数据说明：{model.instance.source_note}")
        print(f"变量：{model.n_decisions} 决策 + {len(model.slack_weights)} 松弛 = {model.n_variables} 量子比特；预算 {model.instance.budget_events} 次")
        print(f"渠首取水预算：{model.instance.water_budget_m3:.2f} m³；效率：{model.instance.irrigation_efficiency:.1%}；"
              f"单次取水：{model.instance.event_gross_water_m3:.2f} m³")
        report = run_comparison(model, args)
        report["input_data"] = input_snapshot
        report["configuration"]["scenario"] = None if args.input else args.scenario
        if forecast_report is not None:
            report["forecast"] = forecast_report
            report["forecast_input"] = forecast_input
            report["scheduling_input_before_forecast"] = scheduling_input
            report["configuration"].update(forecast_model=args.forecast_model, forecast_maxiter=args.forecast_maxiter)
        if args.backend == "qcloud":
            from .hardware_workflow import start_hardware_test
            start_hardware_test(model, report, args)
        if args.export_input:
            args.export_input.parent.mkdir(parents=True, exist_ok=True)
            args.export_input.write_text(json.dumps(input_snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        export_results(model, report, args.output)
        if not args.no_plot:
            plot_comparison(model, report, args.output)
            if forecast_report is not None:
                from .forecast_output import plot_forecast
                plot_forecast(forecast_report, args.output)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"错误：{exc}\n请核对输入与 requirements.txt；无需图表时可加 --no-plot。\n")
    return _print_results(report, args.output)


def _print_results(report, output):
    print("\n方案                 目标值       取水(m³)    净水(m³)   预算可行  低于阈值(区·天)")
    for name, solution in report["solutions"].items():
        print(f"{name:<18} {solution['objective']:11.3f} {solution['gross_water_m3']:10.1f} {solution['net_water_m3']:10.1f}"
              f"   {str(solution['budget_feasible']):<7} {solution['below_threshold_zone_days']:5}")
    if "qaoa" in report:
        print(f"\nQAOA 执行引擎：{report['qaoa']['engine']}")
        stats = report["qaoa"]["sampled"]
        print(f"\nQAOA 原始样本预算可行率：{stats['budget_feasible_probability']:.2%}；"
              f"松弛等式满足率：{stats['zero_slack_residual_probability']:.2%}；"
              f"完整 QUBO 最优态命中率：{stats['optimal_qubo_probability']:.2%}")
        candidate = report["qaoa"]["best_budget_feasible_sample"]
        if candidate is None:
            print("本次 QAOA 抽样没有预算可行方案；未用经典解替换。")
        else:
            print("QAOA 抽样候选：" + json.dumps(candidate["events_by_zone"], ensure_ascii=False))
    if "hardware" in report:
        hardware = report["hardware"]
        print(f"\nQCloud 芯片：{hardware['chip']}；状态：{hardware['status']}；任务：{hardware.get('job_id') or '尚未提交'}")
        if hardware["status"] == "pending":
            print(f'续查：wheat-irrigation --qcloud-resume "{output / "hardware_job.json"}" --no-plot')
        mode = "真机测量完成" if hardware["status"] == "finished" else "尚无已完成的真机测量结果"
    else:
        mode = "经典基线" if report["configuration"]["backend"] == "classical" else "经典 CPU 上的理想量子模拟"
    print(f"\n结果：{output.resolve()}\n执行类型：{mode}；结果不代表真实农田验证。")
    return 2 if report.get("hardware", {}).get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
