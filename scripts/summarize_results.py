"""Generate tables and plots from eval/harness.py JSON outputs.

Examples:
  python scripts/summarize_results.py
  python scripts/summarize_results.py results/t0_regex.json results/t1_regex.json
  python scripts/summarize_results.py --output-dir results/final_summary results/*.json
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path


RATE_COLUMNS = [
    "grounder_success_rate",
    "policy_success_rate",
    "e2e_success_rate",
]


def _import_pyplot():
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            import matplotlib.pyplot as plt
        return plt, None
    except Exception as e:  # noqa: BLE001
        details = stderr.getvalue().strip().splitlines()
        detail = details[-1] if details else str(e)
        return None, f"{type(e).__name__}: {detail}"


def _run_label(path, data):
    grounder = data.get("grounder", "unknown")
    variant = data.get("prompt_variant")
    tiers = "-".join(data.get("tiers") or [])
    if grounder == "llm" and variant:
        return f"{grounder}_{variant}_{tiers}"
    return f"{grounder}_{tiers}"


def load_rows(paths):
    summary_rows = []
    detail_rows = []

    for path in paths:
        data = json.loads(path.read_text())
        run_label = _run_label(path, data)
        run_timestamp = data.get("run_timestamp")
        grounder = data.get("grounder")
        variant = data.get("prompt_variant")
        skip_policy = data.get("skip_policy", False)

        for tier, stats in (data.get("summary") or {}).items():
            row = {
                "source_file": str(path),
                "run": run_label,
                "run_timestamp": run_timestamp,
                "grounder": grounder,
                "prompt_variant": variant,
                "tier": tier,
                "skip_policy": skip_policy,
                **stats,
            }
            row["failure_mode_counts"] = json.dumps(
                row.get("failure_mode_counts") or {},
                sort_keys=True,
            )
            summary_rows.append(row)

        for result in data.get("results") or []:
            detail_rows.append({
                "source_file": str(path),
                "run": run_label,
                "grounder": grounder,
                "prompt_variant": variant,
                "tier": result.get("tier"),
                "id": result.get("id"),
                "instruction": result.get("instruction"),
                "skipped": result.get("skipped", False),
                "skip_reason": result.get("skip_reason"),
                "grounder_success": result.get("grounder_success"),
                "grounder_distance_m": result.get("grounder_distance_m"),
                "grounder_failure_mode": result.get("grounder_failure_mode"),
                "policy_success_rate": result.get("policy_success_rate"),
                "policy_mean_distance_m": result.get("policy_mean_distance_m"),
                "e2e_success_rate": result.get("e2e_success_rate"),
                "predicted_goal": json.dumps(result.get("predicted_goal")),
                "ground_truth_goal": json.dumps(result.get("ground_truth_goal")),
            })

    return summary_rows, detail_rows


def write_csv(rows, path):
    if not rows:
        path.write_text("")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_missing(value):
    return value is None or value == ""


def write_markdown(summary_rows, path):
    if not summary_rows:
        path.write_text("No summary rows found.\n")
        return

    cols = [
        "run",
        "tier",
        "n",
        "grounder_success_rate",
        "policy_success_rate",
        "e2e_success_rate",
        "mean_grounder_distance_m",
        "failure_mode_counts",
    ]
    cols = [c for c in cols if any(c in row for row in summary_rows)]

    formatted_rows = []
    for row in summary_rows:
        formatted = {}
        for col in cols:
            value = row.get(col)
            if col in RATE_COLUMNS and not _is_missing(value):
                formatted[col] = f"{100 * float(value):.1f}%"
            elif col == "mean_grounder_distance_m" and not _is_missing(value):
                formatted[col] = f"{float(value):.4f}"
            else:
                formatted[col] = "" if _is_missing(value) else str(value)
        formatted_rows.append(formatted)

    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(row[col] for col in cols) + " |"
        for row in formatted_rows
    ]
    path.write_text("\n".join([header, separator, *body]) + "\n")


def plot_rates(summary_rows, output_dir):
    plt, err = _import_pyplot()
    if err:
        print(f"Skipping success_rates.png: matplotlib unavailable ({err})")
        return

    if not summary_rows:
        return

    available_rates = [
        c for c in RATE_COLUMNS
        if any(row.get(c) is not None for row in summary_rows)
    ]
    if not available_rates:
        return

    labels = [f"{row.get('run')} / {row.get('tier')}" for row in summary_rows]
    x_positions = list(range(len(labels)))
    width = 0.8 / len(available_rates)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    for idx, rate_col in enumerate(available_rates):
        offsets = [
            x + (idx - (len(available_rates) - 1) / 2) * width
            for x in x_positions
        ]
        values = [
            row.get(rate_col) if row.get(rate_col) is not None else 0
            for row in summary_rows
        ]
        ax.bar(offsets, values, width=width, label=rate_col)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_xlabel("")
    ax.set_title("Grounding, policy, and end-to-end success")
    ax.legend(loc="lower right")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "success_rates.png", dpi=160)
    plt.close(fig)


def plot_failure_modes(detail_rows, output_dir):
    plt, err = _import_pyplot()
    if err:
        print(f"Skipping failure_modes.png: matplotlib unavailable ({err})")
        return

    if not detail_rows:
        return

    failures = [
        row for row in detail_rows
        if row.get("grounder_failure_mode")
    ]
    if not failures:
        return

    labels = sorted({f"{row.get('run')} / {row.get('tier')}" for row in failures})
    modes = sorted({row.get("grounder_failure_mode") for row in failures})
    counts = {
        (label, mode): 0
        for label in labels
        for mode in modes
    }
    for row in failures:
        label = f"{row.get('run')} / {row.get('tier')}"
        counts[(label, row.get("grounder_failure_mode"))] += 1

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    bottoms = [0] * len(labels)
    x_positions = list(range(len(labels)))
    for mode in modes:
        values = [counts[(label, mode)] for label in labels]
        ax.bar(x_positions, values, bottom=bottoms, label=mode)
        bottoms = [a + b for a, b in zip(bottoms, values)]

    ax.set_ylabel("count")
    ax.set_xlabel("")
    ax.set_title("Grounder failure modes")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "failure_modes.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Harness JSON files. Defaults to results/*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/summary"),
        help="Directory for CSV, Markdown, and PNG outputs.",
    )
    args = parser.parse_args()

    inputs = args.inputs or sorted(Path("results").glob("*.json"))
    inputs = [p for p in inputs if p.is_file()]
    if not inputs:
        raise SystemExit("No result JSON files found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, detail_rows = load_rows(inputs)

    summary_csv = args.output_dir / "summary.csv"
    detail_csv = args.output_dir / "details.csv"
    summary_md = args.output_dir / "summary.md"

    write_csv(summary_rows, summary_csv)
    write_csv(detail_rows, detail_csv)
    write_markdown(summary_rows, summary_md)
    plot_rates(summary_rows, args.output_dir)
    plot_failure_modes(detail_rows, args.output_dir)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {summary_md}")
    print(f"Plot step complete; PNGs are written when matplotlib is available.")


if __name__ == "__main__":
    main()
