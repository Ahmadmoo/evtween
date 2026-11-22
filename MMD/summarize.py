"""Table and plot of an MMD run: one row per (sequence, generator), then the mean per generator.

    python -m MMD.summarize MMD/runs/eds-test-<hash>

Reads rows/<sequence>/<generator>.json, so it works while a run is still going.
Averages only scores that compare across sequences: prob_distance (normalized MMD)
and count_relative_to_empty. Raw count_distance grows with a sequence's event count.
Writes summary.png and summary.csv in the run folder. Lower is better.
"""
import argparse
import json
from pathlib import Path
import sys

METRICS = (("prob_distance", "MMD, normalized (shape)"),
           ("count_relative_to_empty", "Count distance / empty prediction"))
SHOWN = ("sequence", "generator", "status", "n_gt", "n_gen", "prob_distance",
         "count_relative_to_empty", "seconds")


def load(run):
    rows = []
    for path in sorted((run / "rows").glob("*/*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            print(f"Skipping unreadable {path}", file=sys.stderr)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run folder (MMD/runs/<dataset>-<split>-<hash>)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    import pandas as pd

    rows = load(args.run)
    if not rows:
        raise SystemExit(f"No finished pairs in {args.run / 'rows'}")
    frame = pd.DataFrame(rows)
    print(frame.reindex(columns=SHOWN).to_string(index=False), "\n")

    ok = frame[frame.status == "ok"]
    metrics = [(c, t) for c, t in METRICS if c in ok and ok[c].notna().any()]
    if ok.empty or not metrics:
        raise SystemExit("No 'ok' rows with scores yet")
    summary = ok.groupby("generator").agg(
        sequences=("sequence", "nunique"),
        **{column: (column, "mean") for column, _ in metrics})
    summary = summary.sort_values(metrics[0][0])
    total = ok.sequence.nunique()
    print(f"Mean over sequences per generator (lower is better); {total} sequence(s) scored:")
    print(summary.to_string(float_format="%.4f"))
    if (summary.sequences < total).any():
        print("\nWarning: generators were scored on different sequences, so their means are "
              "not directly comparable.")
    others = frame[frame.status != "ok"]
    if len(others):
        print(f"\n{len(others)} pair(s) not ok:")
        print(others.reindex(columns=("sequence", "generator", "status", "detail"))
              .to_string(index=False))
    summary.to_csv(args.run / "summary.csv")
    print(f"\nTable: {args.run / 'summary.csv'}")
    if args.no_plot:
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(summary.index)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 0.6 * len(order) + 1.8),
                             squeeze=False)
    for ax, (column, title) in zip(axes[0], metrics):
        ax.barh(range(len(order)), summary.loc[order, column], color="#9db4d3", label="mean")
        for i, generator in enumerate(order):
            values = ok.loc[ok.generator == generator, column].dropna()
            ax.scatter(values, [i] * len(values), s=18, color="#1f3b63", zorder=3,
                       label="per sequence" if i == 0 else None)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_title(f"{title}  (lower is better)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=8,
               frameon=False)
    fig.suptitle(f"{args.run.resolve().name}  ({total} sequences)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.run / "summary.png", dpi=150)
    plt.close(fig)
    print(f"Plot: {args.run / 'summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
