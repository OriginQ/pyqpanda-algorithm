from __future__ import annotations
import argparse
import json
from pathlib import Path

from pyqpanda_alg.BFDCQO.benchmark.family import run_family_benchmark


def main():
    ap = argparse.ArgumentParser(description="Basic/CVaR/Adaptive BF-DCQO matched-shot benchmark")
    ap.add_argument("--output-dir", default="results/bfdcqo_family")
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds starting at zero")
    ap.add_argument("--cvar-alpha", type=float, default=0.10)
    ap.add_argument("--budgets", default="1536,3072,6144")
    ap.add_argument("--backend", choices=["numpy", "pyqpanda"], default="numpy")
    ap.add_argument(
        "--quick", action="store_true",
        help="run one seed and the smallest budget as a smoke test",
    )
    args = ap.parse_args()
    budgets = tuple(int(x) for x in args.budgets.split(",") if x.strip())
    if args.quick:
        budgets = budgets[:1]
        args.seeds = 1
    result = run_family_benchmark(
        budgets=budgets,
        seeds=tuple(range(args.seeds)),
        cvar_alpha=args.cvar_alpha,
        backend=args.backend,
        output_dir=args.output_dir,
    )
    print("BF-DCQO family benchmark")
    print(json.dumps(result["config"], indent=2))
    for r in result["summary"]:
        print(
            f"{r['problem']:24s} budget={r['shot_budget']:4d} {r['variant']:8s} "
            f"Popt={r['mean_p_opt']:.6f}  "
            f"E[ratio]={r['mean_expected_approx_ratio']:.6f}  "
            f"success={r['success_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
