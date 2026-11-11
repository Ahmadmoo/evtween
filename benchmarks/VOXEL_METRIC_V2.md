# Fixed-duration voxel metrics v2

Files: `voxel_metric_v2.py`, `run_voxel_metric_v2.py`. The existing v1 metric,
runner and watchboard are unchanged. V2 reports default to `reports_voxel_v2`.

## Time domain

`--voxel-bin-ms` is required and means actual nominal milliseconds per bin,
not a count. Start defaults to zero relative to the first source frame; stop
defaults to the last source frame. Optional `--voxel-start-s` and
`--voxel-stop-s` use the same relative seconds for GT and prediction. Windows
outside the shared source domain are rejected. Internal RGB frame edges do
not determine bin edges. Events outside the selected half-open window are
excluded, including events exactly at stop.
`scoring_window` explicitly requires source edges starting at zero relative
to the first frame, as produced by `sequence_info`. Nonzero-origin or
pretrimmed source edges are rejected rather than interpreted as a new origin.

For 100 ms and 30 ms bins, edges are 0, 30, 60, 90, 100 ms. The final bin is
10 ms, not a rescaled 25 ms bin. CSV records actual bounds, bin count, final
width and H/W per sequence. Config records nominal width and protocol so
different widths/windows have different report directories.

## Representation and boundaries

The full saved H×W geometry is used without resizing or configurable spatial
averaging. Existing generation crops remain; missing source pixels cannot be
reconstructed. Native signed V2V arrays are not applicable.

Legacy MAE/RMSE/relative L1 retain hard counts and their signed/ON/OFF meanings.
Counts are not rates: the shorter final bin has fewer expected counts. Every
original voxel has equal metric weight, including the final bin; there is no
duration normalization, per-stream normalization or nonzero-only reduction.
MAE/RMSE are streamed across bins after one validation per stream.

Adapted PMSE uses separate polarity channels and forward temporal weights.
For a nominal bin width delta, an event contributes `(next_edge-t)/delta` to
its current full bin and the remainder to the next bin. In the final bin,
all of an event's mass remains there; overflow beyond the scoring window is
folded into the last bin. This conserves one unit per included event but is
an explicit edge adaptation, not author-code reproduction. No events before
start are borrowed as interpolation context.

## Full-coverage PMSE pooling

V2CE III-A specifies 3D average pooling with kernel=stride 2 or 4 followed
by MSE: https://arxiv.org/html/2309.08891v2 . Its II-B motivates temporal
weighting. This v2 code is an original implementation with additional
benchmark conventions, not a copied official evaluator.

No row, column or temporal tail is dropped. Each partial block is averaged
over its actual original-cell count. Squared errors of pooled means are
then weighted by that count, and normalized by total original cells across
both polarities. Thus a one-cell edge block does not receive the same weight
as a full 64-cell block. When all dimensions divide the kernel, this becomes
the ordinary mean of pooled squared errors. Small dimensions are supported.

Public names are `pmse_2_v2ce_adapted` and `pmse_4_v2ce_adapted`, deliberately
distinguishing these results from v1 and the paper's published tables.
PMSE still needs dense grids; one grid is pooled before allocating the next,
but large resolutions with fine time bins may require substantial memory.
No blanket speed claim is made for bincount versus add.at.

## Usage

From repository root (existing event generation outputs are reused):

```sh
python benchmarks/run_voxel_metric_v2.py bsergb test --metric voxel_mae --voxel-bin-ms 30
python benchmarks/run_voxel_metric_v2.py bsergb test --metric voxel_rmse --voxel-bin-ms 30
python benchmarks/run_voxel_metric_v2.py bsergb test --metric voxel_relative_l1 --voxel-bin-ms 30
python benchmarks/run_voxel_metric_v2.py bsergb test --metric pmse_2_v2ce_adapted --voxel-bin-ms 30
python benchmarks/run_voxel_metric_v2.py bsergb test --metric pmse_4_v2ce_adapted --voxel-bin-ms 30
```

Optional `--methods v2ce evtween_v5.2_student`, `--sequence NAME`,
`--voxel-start-s 0.1 --voxel-stop-s 0.5`, and the existing `--data-root`,
`--pred-root`, `--report-root` path overrides are available. Omitting methods
evaluates all registered methods; absent predictions have explicit statuses.
The v2 runner writes results incrementally and recomputes on rerun; it does
not claim v1's SQLite resume/locking behavior. Do not run two writers into
the same v2 report folder simultaneously. V1 reports are never overwritten.

V2 changes representations at the boundary and PMSE pooling, so new scores
must be computed. Compare only scores with the same v2 time/protocol settings.
Open `benchmarks/notebook/watch_voxel_metrics_v2.ipynb` to watch these reports.
Its tables compare nominal bin widths in milliseconds; its 2D/3D views use
the same mass-preserving edges. Set Bin ms and start/stop to match the run.
The v1 notebook remains unchanged.

## Offline interactive metric-versus-bin HTML

Calculate scores at several widths using the same code version, source data,
window, and method selection, then run:

```sh
python -m pip install plotly
python benchmarks/plot_voxel_bins_v2.py --datasets bsergb eds
```

Default inputs: `reports_voxel_v2`. Output: `reports_voxel_v2/html`, one
standalone offline HTML per dataset/protocol group. The exporter never runs
metrics. Optional `--report-root`, `--output-dir`, `--split`, `--methods` and
`--runs` (exact report directory names) control selection. Different evaluator
versions/windows/selections are separated into different files. Ambiguous
duplicate reports are rejected, not selected by modification time.

Within a dataset, choose metric/polarity, width in milliseconds or mean bin
count, and linear/log error axis. Click model legends, zoom and hover. Hover
shows matched sequence count, mean duration, and mean/min/max actual bin count.
Embedded Plotly needs no CDN/network after export. Refresh by rerunning export
after new scores finish; HTML is a snapshot, not a live monitor.
Alternatively, use **Export bin curves** at the bottom of
`watch_voxel_metrics_v2.ipynb`; it uses that notebook's REPORT_ROOT, DATASETS,
SPLIT and METHODS settings and displays links to the generated files. The
chart exporter uses all widths it finds, independently of the table's width
filter, and does not automatically apply the table's RUN_OVERRIDE selection.
Use the exporter CLI with `--runs` if exact run selection is required.

Each metric/channel uses the same methods and sequence cohort across all
available widths, with equal sequence weighting. Sequences with mismatching
actual windows or H/W are excluded. Incomplete models are listed in coverage.
This is not a comparison of raw scales across different metrics. Smaller
errors at coarse widths can hide temporal differences; they do not establish
which metric is intrinsically better at measuring fidelity. A GT zero-error
reference is shown; relative L1 is defined only with nonzero GT mass. Log mode
cannot display zeros. Fine-bin mass-preserving edge behavior still follows v2.

## Checks

`python -m unittest benchmarks.test_voxel_metric_v2 benchmarks.test_voxel_metric -v`

Tests cover 30 ms physical bins and short tails, shifted origins, half-open
endpoints, conserved event mass, polarity, corner-only errors, partial-block
weights, identity, small dimensions, invalid coordinates, analytical values,
and a synthetic actual-CLI run for all five metrics. Real-data accuracy and
paper-table reproduction are not claimed.
