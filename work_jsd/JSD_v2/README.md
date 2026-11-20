# Gaussian event-stream distance

This evaluator takes a saved event stream and its aligned ground truth. Each event has unit weight. Exact duplicate events retain their multiplicity. ON and OFF are compared independently using a Gaussian kernel on pixel coordinates and event time.

## Run on saved streams

From the repository root in the Evtween benchmark environment:

```bash
python -m work_jsd.JSD_v2.run_benchmark eds test \
  --sequence 02_rocket_earth_light --methods v2e \
  --sigma-spatial-px 2 --sigma-temporal-ms 1 \
  --time-window-s 0 0.05
```

The bandwidths above are example settings. Use the same bandwidths and time interval for all generators being compared. Pass `--data-root` for the root of the converted GT tree and `--pred-root` for the root of the saved prediction tree. List multiple sequence and generator names as needed. Omitting `--methods` discovers all generator directories in the selected dataset and split. With no explicit time window, scoring uses the full shared frame interval. The output CSV goes under `work_jsd/JSD_v2/runs/` by default. Pass `--output` to choose a new CSV path.

The runner reads `ev_{x,y,t,p}.npy` and `events.npz` using the existing v3 benchmark alignment and checks `input.json` before scoring. The interval is half-open `[start, stop)`, in seconds relative to the first RGB frame. Full-frame geometry is required by default. Pass `--allow-benchmark-crop` to apply the prediction's verified crop to GT as well.

## Scores

For same-polarity events, the kernel is `exp(-0.5 * ((dx/sigma_s)^2 + (dy/sigma_s)^2 + (dt/sigma_t)^2))`. For opposite polarities it is zero. For each polarity, form the union of event tuples and assign each tuple a weight equal to its GT count minus its generated count. The squared distance is the weighted sum of kernel values over all ordered pairs of tuples, including self-pairs. The joint distance is the square root of the sum of the squared ON and OFF distances.

| CSV column | Meaning |
| --- | --- |
| `distance` | Joint distance. Zero for identical streams. Lower is closer to GT. |
| `distance_on`, `distance_off` | Distance for each polarity. |
| `empty_baseline` | Distance from GT to an empty prediction. |
| `relative_to_empty` | `distance / empty_baseline`. One equals the empty-prediction error. Blank when GT is empty. |
| `n_gt`, `n_gen` | Event counts in the scored domain. |

Computation uses float64, cancels exact matches, and evaluates kernel pairs in bounded-memory blocks. It uses every input event, with no sampling or binning. Dense streams can require quadratic work in the number of distinct events. Start with a short shared interval, especially for dense scenes. `--block-size` controls the size of the temporary pairwise matrices. Per sequence and crop the denominator is computed once and reused across predictions.

## Python API

```python
from work_jsd.JSD_v2 import KernelReference

# gt and pred are prepared Events with timestamps in seconds and signed polarity.
scorer = KernelReference(gt, sigma_spatial_px=2, sigma_temporal_ms=1)
score = scorer.score(pred)
```
