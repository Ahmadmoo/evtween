# Student v8: test-set event generation and metrics

Method: `evtween_v8_student`. This integration targets branch **v8**, inspected at
`348c41f2d99b69266dbe0d3ffba652bc90a6e9bf` (not v8.1 or v8.2).

The shared method registry enables v2, v3, v3_fast and the metric runners.
Metric formulas and generation/sensor logic are unchanged. v3_fast processes all
adjacent frame pairs of every sequence at original resolution, without resizing
or the old center crop. It resumes existing complete outputs.

## Setup on Rails

Run from the benchmark repository with the existing model environment activated:

```bash
conda activate tides
git switch event-evaluation-v52-v53
git pull --ff-only origin event-evaluation-v52-v53
git fetch origin v8

# One-time isolated model checkout; do not switch the benchmark to v8.
export EVTWEEN_V8_ROOT="$(dirname "$PWD")/evtween_bnchmark-v8-348c41f"
if [ ! -e "$EVTWEEN_V8_ROOT" ]; then
  git worktree add --detach "$EVTWEEN_V8_ROOT" 348c41f2d99b69266dbe0d3ffba652bc90a6e9bf
fi
test "$(git -C "$EVTWEEN_V8_ROOT" rev-parse HEAD)" = 348c41f2d99b69266dbe0d3ffba652bc90a6e9bf
```

Weights follow the existing convention:
`$EVTWEEN_WEIGHTS/v8/student.pt`.
The root defaults to
`../evtween-tides-compare/benchmarks/event_generation/checkpoints/evtween_versions`
relative to the benchmark repository. If your weights are elsewhere, explicitly set
`EVTWEEN_WEIGHTS` to the directory **containing v8**, not to the weight file.

```bash
export EVTWEEN_WEIGHTS="${EVTWEEN_WEIGHTS:-$(dirname "$PWD")/evtween-tides-compare/benchmarks/event_generation/checkpoints/evtween_versions}"
test -f "$EVTWEEN_WEIGHTS/v8/student.pt"
```

Use the real student checkpoint containing `cfg` and `model`, with stage
`student` or `joint`; renaming a teacher checkpoint does not turn it into a student.
The worker checks stage, enabled dataset and non-frozen weight compatibility.
Keep existing `EVTWEEN_DATA`, `EVTWEEN_PYTHON`, backbone cache and hub environment
settings when required by your training environment.

## Generate every test sequence

```bash
for dataset in hqevfi bsergb eds; do
  python -u benchmarks/event_generation/generate_event_streams_v3_fast.py \
    "$dataset" test --methods evtween_v8_student --gpu-jobs 4 || break
done
```

Outputs:
`data_test_evt/<dataset>/test/evtween_v8_student/<sequence>/events.npz`.
This uses the existing v3_fast worker reuse and frame cache. Four GPU workers is a
starting setting, not a speed guarantee; tune for the actual model's memory use.
Run metrics after generation completes successfully.

## General metrics

```bash
for dataset in hqevfi bsergb eds; do
  python -u benchmarks/run_metric_fast.py "$dataset" test \
    --methods evtween_v8_student --metric all \
    --device cpu --jobs 4 --workers 4 || break
done
```

This example fits 16 allocated CPU cores. `all` selects event count, count ratio,
ON percentage, event rate, ISI distribution, ISI median, ISI CV, IEI distance,
same-timestamp fraction, Fano factor, and **chamfer_full only** among Chamfer variants.
Outputs: `reports/<dataset>/<metric>/<config_id>/results.csv`.

## Fixed-duration voxel metrics

Use the same temporal bin as your previous comparison; 10 ms below is an example.

```bash
for dataset in hqevfi bsergb eds; do
  python -u benchmarks/run_voxel_metric_v2_fast.py "$dataset" test \
    --methods evtween_v8_student --metric all \
    --voxel-bin-ms 10 --workers 4 || break
done
```

This computes voxel MAE, RMSE, Relative L1 and PMSE 2/4 adapted.
Outputs: `reports_voxel_v2/<dataset>/<metric>/<config_id>/results.csv`.

## Validation scope

The v8 generation source matches all five exact source boundaries required by
the existing worker adapter. Both fast metric parsers use the shared registry.
Real checkpoint loading and GPU generation must be validated on Rails; the
checkpoint and Rails datasets are not available in the development workspace.
