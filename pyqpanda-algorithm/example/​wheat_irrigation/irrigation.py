"""Source-checkout entry point for the standalone wheat application."""

from wheat_irrigation.cli import (
    export_results, main, plot_comparison, run_comparison, summarize_distribution,
)

if __name__ == "__main__":
    raise SystemExit(main())
