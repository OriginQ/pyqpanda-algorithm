import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from .circuit import resource_profile
from .config import ExperimentConfig
from .data import make_mnist_like_binary_dataset
from .metrics import classification_metrics
from .models import LogisticBaseline, NoiseAwareVQC
from .preprocessing import preprocess_splits, stratified_split
from .reporting import write_bar_svg, write_report


def run_experiment(output_dir: Path, config: ExperimentConfig) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    x_rows, y = make_mnist_like_binary_dataset(config.n_samples, config.seed)
    splits = stratified_split(x_rows, y, config.train_ratio, config.val_ratio, config.seed)
    processed = preprocess_splits(splits)

    models = [
        ("logistic_8feature", LogisticBaseline(config.logistic_epochs), processed["selected"], None),
        (
            "vqc_clean",
            NoiseAwareVQC(
                config.n_qubits,
                config.ansatz_layers,
                config.vqc_epochs,
                config.shots,
                train_noise_prob=0.0,
                angle_jitter=0.0,
                seed=config.seed,
            ),
            processed["angle"],
            0.0,
        ),
        (
            "vqc_noise_aware",
            NoiseAwareVQC(
                config.n_qubits,
                config.ansatz_layers,
                config.vqc_epochs,
                config.shots,
                train_noise_prob=config.train_noise_prob,
                angle_jitter=config.angle_jitter,
                seed=config.seed + 17,
            ),
            processed["angle"],
            config.train_noise_prob,
        ),
    ]

    metric_rows: List[Dict[str, object]] = []
    resource_rows: List[Dict[str, object]] = []
    robustness_rows: List[Dict[str, object]] = []
    test_accuracy = {}

    for model_name, model, data_splits, model_noise in models:
        train_x, train_y = data_splits["train"]
        start = time.perf_counter()
        model.fit(train_x, train_y)
        train_time = time.perf_counter() - start
        infer_time = 0.0

        for split_name in ("train", "val", "test"):
            x_split, y_split = data_splits[split_name]
            start = time.perf_counter()
            if model_name.startswith("vqc"):
                probs = model.predict_proba(x_split, noise_prob=model_noise or 0.0, angle_jitter=0.0)
            else:
                probs = model.predict_proba(x_split)
            infer_time += time.perf_counter() - start
            row = {"model": model_name, "split": split_name}
            row.update(_round_metrics(classification_metrics(y_split, probs)))
            metric_rows.append(row)
            if split_name == "test":
                test_accuracy[model_name] = row["accuracy"]

        if model_name.startswith("vqc"):
            profile = resource_profile(
                config.n_qubits,
                config.ansatz_layers,
                config.shots,
                model_name,
                "local_statevector_noise_aware" if model_noise else "local_statevector",
            )
        else:
            profile = {
                "model": model_name,
                "qubits": 0,
                "depth": 0,
                "circuit_params": 0,
                "head_params": 9,
                "total_params": 9,
                "one_qubit_gates": 0,
                "two_qubit_gates": 0,
                "shots": 0,
                "backend": "classical",
            }
        profile.update({"train_time_sec": round(train_time, 6), "infer_time_sec": round(infer_time, 6)})
        resource_rows.append(profile)

    for model_name, model, data_splits, _ in models:
        if not model_name.startswith("vqc"):
            continue
        x_test, y_test = data_splits["test"]
        for noise in (0.0, 0.02, config.eval_noise_prob, 0.08):
            probs = model.predict_proba(x_test, noise_prob=noise, angle_jitter=config.angle_jitter if noise else 0.0)
            row = {"model": model_name, "noise_prob": noise}
            row.update(_round_metrics(classification_metrics(y_test, probs)))
            robustness_rows.append(row)

    split_info = split_summary(splits)
    write_csv(output_dir / "metrics.csv", metric_rows, list(metric_rows[0].keys()))
    write_csv(output_dir / "resource_table.csv", resource_rows, list(resource_rows[0].keys()))
    write_csv(output_dir / "robustness.csv", robustness_rows, list(robustness_rows[0].keys()))
    write_json(output_dir / "config.json", config.__dict__)
    write_json(output_dir / "split_summary.json", split_info)
    write_report(output_dir / "report.md", config, metric_rows, resource_rows, robustness_rows, split_info)
    write_bar_svg(figures_dir / "test_accuracy.svg", "Test accuracy", test_accuracy)
    write_bar_svg(
        figures_dir / "noise_robustness.svg",
        "F1 at evaluation noise",
        {row["model"] + "@" + str(row["noise_prob"]): row["f1"] for row in robustness_rows},
    )
    return {"metrics": metric_rows, "resources": resource_rows, "robustness": robustness_rows}


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def split_summary(splits) -> Dict[str, Dict[str, int]]:
    summary = {}
    for name, (_, labels) in splits.items():
        summary[name] = {
            "samples": len(labels),
            "class_0": sum(1 for label in labels if label == 0),
            "class_1": sum(1 for label in labels if label == 1),
        }
    return summary


def _round_metrics(metrics):
    return {key: round(float(value), 6) for key, value in metrics.items()}

