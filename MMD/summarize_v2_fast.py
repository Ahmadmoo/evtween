"""Summary of an MMD run with similarity, per-sequence ranks and head-to-head wins.

    python -m MMD.summarize_v2_fast MMD/runs/eds-test-v2-<hash>
    python -m MMD.summarize_v2_fast MMD/runs/eds-test-v2-<hash> --metric prob_distance

Works on run.py and run_v2_fast.py folders (similarity only exists in v2_fast runs with
output_version 2). Prints, per generator: the mean of each comparable score over
sequences, the mean rank per sequence (1 = best) and, for --metric (default: similarity
when present, else prob_distance), how many sequences each generator beats each other
one. Close means are only a real difference when one generator wins on most sequences.
Writes summary_v2.csv, wins_<metric>.csv and summary_v2.png in the run folder.
"""
import argparse
from pathlib import Path

from .summarize import load

# column, title, lower is better
METRICS = (("similarity", "Similarity to GT (cosine)", False),
           ("prob_distance", "MMD, normalized (shape)", True),
           ("count_relative_to_empty", "Count distance / empty prediction", True))
SHOWN = ("sequence", "generator", "status", "n_gt", "n_gen", "similarity", "prob_distance",
         "count_relative_to_empty", "seconds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Run folder (MMD/runs/...)")
    parser.add_argument("--metric", choices=[m[0] for m in METRICS],
                        help="Score for ranks and wins (default: similarity if present)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    import pandas as pd

    rows = load(args.run)
    if not rows:
        raise SystemExit(f"No finished pairs in {args.run / 'rows'}")
    frame = pd.DataFrame(rows)
    print(frame.reindex(columns=SHOWN).to_string(index=False), "\n")
    ok = frame[frame.status == "ok"]
    metrics = [m for m in METRICS if m[0] in ok and ok[m[0]].notna().any()]
    if ok.empty or not metrics:
        raise SystemExit("No 'ok' rows with scores yet")
    metric = args.metric or metrics[0][0]
    lower = {c: low for c, _, low in METRICS}
    if metric not in [m[0] for m in metrics]:
        raise SystemExit(f"{metric} is not in this run (old run folder?)")

    table = ok.pivot_table(index="sequence", columns="generator", values=metric)
    ranks = table.rank(axis=1, ascending=lower[metric])
    summary = ok.groupby("generator").agg(
        sequences=("sequence", "nunique"), **{c: (c, "mean") for c, _, _ in metrics})
    summary[f"mean_rank_{metric}"] = ranks.mean()
    summary = summary.sort_values(f"mean_rank_{metric}")
    total = ok.sequence.nunique()
    print(f"Per generator over {total} sequence(s); similarity: higher is better, "
          f"distances and ranks: lower is better")
    print(summary.to_string(float_format="%.4f"))
    if (summary.sequences < total).any():
        print("\nWarning: generators were scored on different sequences.")

    order = list(summary.index)
    wins = pd.DataFrame(index=order, columns=order, dtype=object)
    for a in order:
        for b in order:
            if a == b:
                wins.loc[a, b] = "-"
                continue
            both = table[[a, b]].dropna()
            better = both[a] > both[b] if not lower[metric] else both[a] < both[b]
            wins.loc[a, b] = f"{int(better.sum())}/{len(both)}"
    print(f"\nSequences where the row beats the column on {metric}:")
    print(wins.to_string())

    others = frame[frame.status != "ok"]
    if len(others):
        print(f"\n{len(others)} pair(s) not ok:")
        print(others.reindex(columns=("sequence", "generator", "status", "detail"))
              .to_string(index=False))
    summary.to_csv(args.run / "summary_v2.csv")
    wins.to_csv(args.run / f"wins_{metric}.csv")
    print(f"\nTables: {args.run / 'summary_v2.csv'}, {args.run / f'wins_{metric}.csv'}")
    if args.no_plot:
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics),
                                                       0.6 * len(order) + 1.8), squeeze=False)
    for ax, (column, title, low) in zip(axes[0], metrics):
        ax.barh(range(len(order)), summary.loc[order, column], color="#9db4d3", label="mean")
        for i, generator in enumerate(order):
            values = ok.loc[ok.generator == generator, column].dropna()
            ax.scatter(values, [i] * len(values), s=18, color="#1f3b63", zorder=3,
                       label="per sequence" if i == 0 else None)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_title(f"{title}  ({'lower' if low else 'higher'} is better)", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=8,
               frameon=False)
    fig.suptitle(f"{args.run.resolve().name}  ({total} sequences, sorted by mean rank on "
                 f"{metric})", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.run / "summary_v2.png", dpi=150)
    plt.close(fig)
    print(f"Plot: {args.run / 'summary_v2.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
