# Event distribution histograms

Evaluate saved benchmark event streams against the corresponding GT. The CSV contains one row per dataset, split, sequence and generator. Use those four columns to join downstream evaluation results.

## Run one sequence

From the repository root in the benchmark environment:

```bash
python -m unittest jsd_work.histogram.test_histogram -v

python -m jsd_work.histogram.run_benchmark eds test \
  --sequence 02_rocket_earth_light \
  --methods v2e evtween_v7_student \
  --config jsd_work/histogram/config.example.json
```

Use `--data-root` for the root containing converted `dataset/split/sequence` directories and `--pred-root` for the root containing saved `dataset/split/generator/sequence/events.npz` files. The defaults follow the existing benchmark and honor `EVTWEEN_DATA` for the GT root. Omitting `--sequence` discovers all sequences. Omitting `--methods` discovers prediction folders. The runner uses NumPy, SciPy and the benchmark's Pillow dependency. SciPy 1.13 or newer is needed for optional spatial Wasserstein. Enable `save_plots` in an environment with Matplotlib to export the figure.

Each run writes `config.json` with effective parameters and library versions. By default it also saves `results.csv` under a unique directory in `jsd_work/histogram/runs/`. Use `--output` to supply a new output path. With `save_distributions`, each pair exports occupied bin IDs, raw counts and logical shapes as NPZ. With `save_plots`, each nonempty pair exports a PNG with spatial probability maps, temporal and polarity comparisons, and x–time projections summed across y. Plots use a shared scale within each pair of maps or projections.

## Common evaluation domain

The adapter uses the benchmark's sequence geometry, metadata verification and prediction loader. It checks the prediction sidecar against the converted GT sequence, including source identifiers, frame indices, dimensions, time origin and duration. The v3 format stores predictions as seconds relative to the first RGB frame, and converted GT as seconds on the original sequence clock. It subtracts the recorded frame timestamp from GT and uses the same half-open `[start, stop)` for both streams. The default window spans the first through the last frame. To score a shorter interval, set `time_window_s` to two seconds values relative to the first frame.

Full frame dimensions are read from the source. For archived cropped predictions, set `allow_benchmark_crop` only if you want the recorded crop applied to GT too. CSV rows record the scoring dimensions and crop origin. Compare models under the same geometry and interval. Sidecar verification establishes the declared coordinate frame and timing; the method assumes the data were correctly synchronized by the dataset and generation pipelines.

Event arrays must have matching 1D shape. Coordinates must be integer pixel indices within the frame. Timestamps must be finite and nondecreasing. The GT supports either signed or binary polarity as used by converted data. Predictions follow the signed v3 convention. Exact duplicate tuples in the scored interval are counted in the CSV and retained as events. Missing inputs, metadata mismatches and native-voxel-only predictions receive explicit statuses. If either stream is empty, count errors remain available and distribution scores are blank with an empty-stream status.

## Histogram definitions

Set `spatial_bin_size` in pixels and `temporal_bin_s` in seconds independently. All events in the accepted domain contribute one count per requested distribution. Grid dimensions use ceiling division. The last spatial or temporal cell can be smaller when the domain is not evenly divisible. Floating-point time boundaries are shared between GT and predictions.

| Distribution | Bin axes | Meaning |
| --- | --- | --- |
| Spatial | y, x | Pixel occupancy over the selected interval |
| Temporal | t | Activity per time bin |
| Spatiotemporal | t, y, x | Where events occur at each time bin |
| Polarity | OFF, ON | Proportion of negative and positive events |
| Joint | t, y, x, polarity | Spatiotemporal activity by polarity. Add `joint` to enable it. |

All distributions use sparse arrays of occupied linear bin IDs and integer counts in NumPy row-major order. A bin's mass is its count divided by the total scored event count. CSV reports the logical bin count, occupied bins per stream and the bytes of the stored IDs and counts. Peak memory also includes loaded events, transient sorting arrays and the occupied-support union.

## Scores

- **JSD** is `(KL(P || M) + KL(Q || M)) / 2`, with `M = (P + Q) / 2`. Zero-mass terms contribute zero. The default logarithm is natural. `normalize_jsd` scales the result to `[0, 1]`. `log_base: "2"` expresses unnormalized information scores in bits. SciPy's `jensenshannon` returns the square root of the divergence and is used as an independent test reference. [SciPy Jensen–Shannon distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html)
- **KL** is `sum(P * log(P / Q))` over bins where `P > 0`, with `P = GT` and `Q = generated`. The default `kl_epsilon` mixes each distribution with a uniform distribution across the full grid using weight epsilon. Unoccupied bins in both streams receive equal mass and contribute zero, so the calculation stays sparse. Set `reverse_kl` to also report generated-to-GT KL. Set `kl_epsilon` to zero for unsmoothed KL, which is infinite when the second distribution assigns zero mass to a bin used by the first. [SciPy relative entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html)
- **Wasserstein-1** is optional. Temporal transport uses the absolute difference between time-bin centers in seconds. Spatial transport uses Euclidean distances between 2D bin centers in pixels. Spatial work is limited by the product of the two occupied-support sizes. When that product exceeds `max_spatial_transport_pairs`, the CSV leaves the score blank and reports `spatial_transport_limit`. Spatiotemporal, joint and polarity transport return `not_applicable`. [SciPy 1D Wasserstein](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) · [SciPy multidimensional Wasserstein](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance_nd.html)
- **Count error** reports both event totals, the absolute difference and `abs(n_gt - n_gen) / max(n_gt, count_epsilon)`. It is independent of the probability scores.

## Validation and next experiments

The tests check analytical cases, sparse storage, agreement with SciPy, cutoff behavior and the saved-stream benchmark entry point. The normalized JSD can saturate when very fine bins make sets of occupied cells nearly disjoint. Changing bin size changes what the score measures, and coarser bins can hide motion-timing errors. As a first experiment, compare spatial bin sizes of 1, 4 and 8 pixels and bin times of 1, 5 and 10 ms on the same saved streams. Repeat with controlled time jitter, spatial shifts, drops and injections, and inspect support overlap and count error alongside JSD. Then test associations with downstream performance using sequence-aware analysis.
