# Gaussian kernel distance (MMD) between event streams

Self-contained replacement for `work_jsd/JSD_v2`. It has no imports from `jsd_work/` or
`work_jsd/`; data loading and metadata checks come from `benchmarks/run_metric.py`, like
every other benchmark metric.

| File | Role |
| --- | --- |
| `kernel.py` | The distance: `exact` reference, `fast` tiled algorithm on CPU threads or a GPU, `count` and `probability` weightings |
| `benchmark_io.py` | GT geometry, prediction alignment (`input.json`), prediction reading |
| `events.py` | Event validation and clock conversion (copied from `jsd_work/histogram/events.py`) |
| `run.py` | Parallel, resumable runner |
| `summarize.py` | Mean per generator over sequences, `summary.csv` and `summary.png` |
| `check_device.py` | GPU vs CPU on synthetic events: same scores? how much faster? |
| `*_v2_fast.py` | Faster version of the same metric (see v2_fast below) |
| `test_mmd.py` | Analytical cases, dense O(n²) reference, fast/exact/GPU-path agreement, runner, resume, GPU scheduling |

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
  thread. With `--device cuda` the same blocks run in float64 on a GPU through PyTorch.

`--device`, `--tile-px`, `--block-size`, `--chunk-size`, `--jobs` and `--threads` change
speed and memory only, never the score, so changing them keeps resume working. Run folders
are keyed on the settings and on `METRIC_VERSION` in `kernel.py`, which is bumped only when
a change alters scores.

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

While a pair runs, a line is printed every `--progress-every` seconds (default 60; 0 turns
it off) with the stage, the estimated percentage, elapsed and remaining time and kernel pairs
per second:

```
  [02_rocket_earth_light/v2e] start: generator 1/7 pending here
  [02_rocket_earth_light/v2e] ON merge: 434,112,318 events
  [02_rocket_earth_light/v2e] ON:  24.5%  6m10s elapsed, ~19m01s left, 3.1e+10 pairs/s
```

Stages per pair: `ON merge`, `ON`, `OFF merge`, `OFF`, and once per sequence the same for
`GT baseline`. The percentage is estimated from event counts per tile, so the time left is
approximate.

Output goes to `MMD/runs/<dataset>-<split>-<hash>/` (or `--output`): `config.json`,
`rows/<sequence>/<generator>.json` written as each pair finishes, and `results.csv`. Rerun
the same command to resume. A pair is rescored only if it did not finish or its GT or
prediction files changed. A folder with different settings is refused.

### Summary and plot

```bash
python -m MMD.summarize MMD/runs/eds-test-<hash>
```

Prints every (sequence, generator) row, then the mean per generator over sequences, sorted
best first, and writes `summary.csv` and `summary.png` (bars = mean, dots = sequences) in the
run folder. It reads the `rows/` files, so it also works while a run is still going. Only
`prob_distance` and `count_relative_to_empty` are averaged; raw `count_distance` grows with
a sequence's event count. Compare generators only when the `sequences` column is the same
for all of them.

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

### GPU

Needs PyTorch with CUDA in the environment. Everything stays float64, so A100 and H100 give
the CPU's scores (H100 has about 3× the FP64 rate, so it is faster). The GPU model is chosen
in the job request, not in the code. Check a new GPU first:

```bash
python -m MMD.check_device                    # cuda vs cpu on 2M synthetic events
python -m MMD.run eds test --sigma-spatial-px 2 --sigma-temporal-ms 1 --device cuda
```

`--device cuda` uses every visible GPU, one sequence per GPU at a time (`--jobs` defaults to
the GPU count); `--device cuda:1` uses one. CPU threads (`--threads`) prepare the next tiles
while the GPU works; loading and merging events still runs on the CPU, so ask for a few CPUs
and the same RAM as a CPU run. Each GPU block holds 2 × `block_size × chunk_size` doubles
(2 GiB by default). The GPU path has no expm1 split for blocks within one sigma, which only
matters for displacements below about 1e-7 σ.

```bash
#SBATCH --gres=gpu:h100:1      # or gpu:a100:1; names depend on the cluster (sinfo -o "%P %G")
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
```

## v2_fast

`kernel_v2_fast.py`, `run_v2_fast.py`, `check_device_v2_fast.py` and `test_v2_fast.py`
compute the same distance with less work; the files above are unchanged.

- Per polarity, D² = S_gg + S_pp − 2 S_gp (count) or S_gg/N² + S_pp/M² − 2 S_gp/(NM)
  (probability), with S the sums of kernel values GT-GT, prediction-prediction and
  GT-prediction. S_gg is computed once per sequence, so each generator costs about half the
  pairs. When a prediction shares most of its events exactly with GT, the channel is scored
  the kernel.py way instead (identical events cancel first); each channel takes the cheaper
  way. Identical channels give exactly 0.
- On a GPU, `torch.compile` fuses distance, exp and the weighted sum into one kernel
  (`--no-compile` turns it off; if compiling fails it falls back and the row's `device`
  says so).
- Scores equal kernel.py's to ~1e-15 in the tests. Runs go to
  `MMD/runs/<dataset>-<split>-v2-<hash>`, never mixed with run.py's.
- Each row also stores the three sums per polarity (`sum_gt_gt_*`, `sum_gen_gen_*`,
  `sum_gt_gen_*`, count-weighted) and `similarity` = S_gp / sqrt(S_gg S_pp) (joint over
  polarities, plus `_on` / `_off`): the cosine between the GT and prediction densities
  smoothed by the kernel, in [0, 1], independent of event counts. 1 = same shape, 0 = no
  overlap. These cost nothing extra. (Rows from before output_version 2 lack them; such
  runs get a new folder.)
- `python -m MMD.summarize_v2_fast <run folder>` adds the mean similarity, the mean rank per
  sequence and a head-to-head table (in how many sequences each generator beats each
  other one). Close means are a real difference only if one generator wins on most
  sequences.

```bash
python -m unittest MMD.test_v2_fast -v
python -m MMD.check_device_v2_fast        # fused / element-wise GPU vs kernel.py CPU
python -m MMD.run_v2_fast eds test --sigma-spatial-px 2 --sigma-temporal-ms 1 --device cuda
python -m MMD.summarize_v2_fast MMD/runs/eds-test-v2-<hash>
```

Resources for a GPU run: one GPU (A100 80GB or H100), 16 CPUs (loading and merging events
run on the CPU; at most 4 threads prepare tiles), 128 GB RAM for the largest EDS sequence.

## Choosing σ

The score depends strongly on σs and σt. Report a small grid (for example σs ∈ {1, 2, 4} px,
σt ∈ {0.5, 1, 5} ms) and check that the ranking of generators is stable. Cost grows with
σs² · σt · local event density.
