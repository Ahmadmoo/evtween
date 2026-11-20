# Gaussian kernel distance (MMD) between event streams

Self-contained replacement for `work_jsd/JSD_v2`. It has no imports from `jsd_work/` or
`work_jsd/`; data loading and metadata checks come from `benchmarks/run_metric.py`, like
every other benchmark metric.

| File | Role |
| --- | --- |
| `kernel.py` | The distance: `exact` reference and `fast` tiled/threaded algorithm, `count` and `probability` weightings |
| `benchmark_io.py` | GT geometry, prediction alignment (`input.json`), prediction reading |
| `events.py` | Event validation and clock conversion (copied from `jsd_work/histogram/events.py`) |
| `run.py` | Parallel, resumable runner |
| `test_mmd.py` | Analytical cases, dense O(n²) reference, fast/exact agreement, runner and resume |

## Definition

For each polarity, with signed weights w (GT positive, generated negative) and
`k = exp(-0.5 ((dx/σs)² + (dy/σs)² + (dt/σt)²))`, the squared distance is
`Σ_ij w_i w_j k_ij`. Opposite polarities have zero kernel, so the joint distance is
`hypot(on, off)`.

| Weighting | Weights | Columns | Measures |
| --- | --- | --- | --- |
| `count` | +1 / −1 per event | `count_*` | event placement **and** count (van Rossum style; same as JSD_v2) |
| `probability` | +1/N_gt / −1/N_gen (N over both polarities) | `prob_*` | shape of the distribution only (standard biased MMD, ≤ √2) |

Both are computed in one pass by default (`--weighting count probability`).
`count_relative_to_empty` divides by the distance of an empty prediction; it can exceed 1.
`prob_*` is blank when either stream is empty. Pair it with a count metric.

## Algorithms

- `exact`: every pair; stops only where the Gaussian underflows (40 σt). Same numbers as JSD_v2.
- `fast` (default): skips pairs farther than `--cutoff-sigma` (default 10) sigmas along an
  axis, using spatial tiles and a time sweep, and runs tiles on threads. Each skipped pair has
  kernel < e^(−50); `*_truncation_bound` is a guaranteed bound on the resulting change of the
  distance. On synthetic 640×480 data it matched `exact` to 1e-14 and was ~28× faster on one
  thread.

`--tile-px`, `--block-size`, `--chunk-size`, `--jobs` and `--threads` change speed and memory
only, never the score, so changing them keeps resume working.

## Run

From the repository root:

```bash
python -m unittest MMD.test_mmd -v

# Check which GT and prediction folders will be used, without scoring:
python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1 --dry-run

python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1
python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1 \
  --sequence 02_rocket_earth_light --methods v2e evtween_v7_student
```

GT comes from the `source` in each prediction's `input.json` (as in `run_metric.py`), else
`--data-root` / `$EVTWEEN_DATA`. The start of a run prints the GT folder of every sequence,
and every CSV row records `gt_source` and `prediction`.

Output goes to `MMD/runs/<dataset>-<split>-<hash>/` (or `--output`): `config.json`,
`rows/<sequence>/<generator>.json` written as each pair finishes, and `results.csv`. Rerun
the same command to resume. A pair is rescored only if it did not finish or its GT or
prediction files changed. A folder with different settings is refused.

### CPUs

The runner sizes itself from the CPUs this process is allowed to use (Slurm allocation or
`taskset`), not the machine's core count. By default it runs about one sequence per 8 CPUs and
splits the CPUs among them as threads; a single sequence gets all CPUs. Override with
`--jobs` and `--threads` (keep `jobs × threads ≤ CPUs`). Memory per thread is roughly
`block_size × chunk_size × 16` bytes (≈ 32 MB by default) plus the loaded events.

Slurm example for 32 CPUs:

```bash
#!/bin/bash
#SBATCH --job-name=mmd
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
cd /path/to/evtween
python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1
```

Without Slurm, `taskset -c 0-31 python -m MMD.run ...` gives the same limit.

## Choosing σ

The score depends strongly on σs and σt. Report a small grid (for example σs ∈ {1, 2, 4} px,
σt ∈ {0.5, 1, 5} ms) and check that the ranking of generators is stable. Cost grows with
σs² · σt · local event density.
