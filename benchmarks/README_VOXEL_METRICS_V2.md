# Voxel metrics v2: complete user guide

This guide describes the implementation in `voxel_metric_v2.py`, its runner,
and its notebook/interactive plots. **Lower error is better within the same
metric, time grid, geometry and protocol.** Voxelization is preprocessing,
not an additional metric.

## 1. Run every voxel metric for every registered method

From the repository root, for `bsergb/test` and 10 ms bins:

```bash
python benchmarks/run_voxel_metric_v2.py bsergb test --metric all --voxel-bin-ms 10 --report-root reports_voxel_v2_ab
```

Omit `--methods` to evaluate every registered method. Existing saved event
predictions are read; this does not train models, run generation, or generate
missing predictions. `--metric all` means the **five v2 voxel metric families**;
it does not include Chamfer or other non-voxel metrics from `run_metric.py`.

For 5, 10 and 30 ms (Bash/Linux):

```bash
for bin_ms in 5 10 30; do
  python benchmarks/run_voxel_metric_v2.py bsergb test \
    --metric all \
    --voxel-bin-ms "$bin_ms" \
    --report-root reports_voxel_v2_ab || exit 1
done
```

PowerShell equivalent:

```powershell
foreach ($binMs in 5, 10, 30) {
    python benchmarks/run_voxel_metric_v2.py bsergb test --metric all --voxel-bin-ms $binMs --report-root reports_voxel_v2_ab
    if ($LASTEXITCODE -ne 0) { throw "Voxel evaluation failed for $binMs ms" }
}
```

All registered methods are currently `v2e`, `iebcs`, `dvs_voltmeter`, `v2ce`,
`v2v`, `evis_core`, `fracevent`, and teacher/student variants of Evtween v5.2
and v5.3. Missing event files produce `missing_prediction`. Native V2V signed
frame arrays produce `not_applicable`: their separate polarity channels and
individual timestamps cannot be reconstructed for this protocol.

To evaluate a subset, append for example:

```bash
--methods v2ce evtween_v5.2_student evtween_v5.3_student
```

Optional path and window arguments:

```bash
--data-root /path/to/data --pred-root /path/to/data_test_evt
--sequence SEQUENCE_NAME
--voxel-start-s 0.1 --voxel-stop-s 0.5
```

Dataset directories are expected beneath `DATA_ROOT/dataset/split/sequence`;
predictions beneath `PRED_ROOT/dataset/split/method/sequence/events.npz`, with
the corresponding `input.json`. Start/stop are seconds relative to the first
source frame, not absolute timestamps or offsets from the first event.
Without custom bounds, evaluate the entire first-to-last-frame interval.

## 2. Shared spatial and temporal domain

Let the scoring interval be $[a,z)$ seconds, duration $D=z-a$, image dimensions
$H,W$, and nominal bin duration $\Delta=\texttt{bin\_ms}/1000$ seconds.
The bin edges are anchored at $a$, spaced by $\Delta$, and terminate at $z$:

$$B=\left\lceil\frac{D}{\Delta}\right\rceil,\qquad
\tau_b=a+b\Delta\ (0\le b<B),\qquad \tau_B=z.$$

Floating-point roundoff at exact multiples is handled to avoid spurious tiny
trailing bins. If $D$ is not a multiple of $\Delta$, the final bin is shorter.
For 100 ms with 30 ms bins: `[0,30)`, `[30,60)`, `[60,90)`, `[90,100)` ms.
These are not rescaled to four equal 25 ms bins.

GT and every prediction use identical bounds, edges and saved H×W geometry.
No stream is independently normalized to its first/last event. Internal RGB
frame gaps do not choose the bin edges. Source-frame times must already be
relative to the first frame, with first edge exactly zero. Nonzero-origin
source edges are rejected. Endpoint comparisons allow only source-origin
floating-point subtraction roundoff, not a different time origin.

Every temporal bin retains the full **available saved image**. No configurable
spatial block averaging or image resizing is used. Existing generation crops
remain: this does not recover original-sensor pixels outside saved predictions.
Inactive pixels are included. The last short bin has equal cell weight; counts
are not divided by its duration. Consequently these are count errors, not
event-rate errors.

### FPS versus bin duration

$$\text{frame gap (ms)}=\frac{1000}{FPS},\qquad
\text{bin width (ms)}=\frac{1000}{FPS\,b}$$

where $b$ is the number of bins per frame gap in a frame-based protocol. V2
instead takes physical bin duration directly. Changing it changes $B$, not
the number of RGB frames selected. For 1 s, 10 ms gives 100 bins. Different
sequence durations yield different bin counts at the same nominal width.

## 3. Hard-count voxel representation

For polarity $c\in\{+,-\}$, construct nonnegative counts
$C_c\in\mathbb{R}^{B\times H\times W}$:

$$C_c(b,y,x)=\sum_j\mathbf{1}[c_j=c]\,
\mathbf{1}[(x_j,y_j)=(x,y)]\,
\mathbf{1}[\tau_b\le t_j<\tau_{b+1}].$$

ON is `p=1`; OFF is `p=-1` or `p=0`. Coordinates and polarities are validated.
Events outside the selected interval, including events exactly at stop, are
excluded. Let $P_+,P_-$ denote predicted counts, $G_+,G_-$ GT counts, and

$$e_+=P_+-G_+,\quad e_-=P_--G_-,\quad N=BHW.$$

All sums below range over the $N$ temporal/spatial cells of one sequence.

## 4. MAE and RMSE: A, B, Signed, ON and OFF

The two CLI families `voxel_mae` and `voxel_rmse` each export five views.
A/B do not require separate CLI passes.

### ON and OFF

$$MAE_{ON}=\frac1N\sum_i|e_{+,i}|,\qquad
MAE_{OFF}=\frac1N\sum_i|e_{-,i}|.$$

$$RMSE_{ON}=\sqrt{\frac1N\sum_i e_{+,i}^2},\qquad
RMSE_{OFF}=\sqrt{\frac1N\sum_i e_{-,i}^2}.$$

### A: polarity-aware count error (primary count view)

$$MAE_A=\frac{\sum_i|e_{+,i}|+\sum_i|e_{-,i}|}{2N}
=\frac{MAE_{ON}+MAE_{OFF}}2.$$

$$RMSE_A=\sqrt{\frac{\sum_i e_{+,i}^2+\sum_i e_{-,i}^2}{2N}}
=\sqrt{\frac{RMSE_{ON}^2+RMSE_{OFF}^2}{2}}.$$

A is calculated from ON/OFF outputs **for each sequence**, then sequence
scores are averaged for the dataset. Do not combine dataset-average RMSE_ON
and RMSE_OFF: square roots and averaging do not commute. Likewise, do not
take the arithmetic average of ON/OFF RMSEs as A RMSE.

### B: polarity-agnostic total-count error (complementary view)

$$P_{total}=P_++P_-,\quad G_{total}=G_++G_-,\quad
e_{total}=e_++e_-.$$

$$MAE_B=\frac1N\sum_i|e_{+,i}+e_{-,i}|,\qquad
RMSE_B=\sqrt{\frac1N\sum_i(e_{+,i}+e_{-,i})^2}.$$

B compares total counts in **each voxel**, not just one total for the entire
sequence. It is not equivalent to `event_count_ratio`. It ignores polarity
swaps if local total counts remain correct.

### Signed: net signed-count error (diagnostic view)

$$P_s=P_+-P_-,\quad G_s=G_+-G_-,\quad e_s=e_+-e_-.$$

$$MAE_s=\frac1N\sum_i|e_{+,i}-e_{-,i}|,\qquad
RMSE_s=\sqrt{\frac1N\sum_i(e_{+,i}-e_{-,i})^2}.$$

Signed is deliberately retained; it is not the combined polarity-aware error.
Same-direction ON/OFF errors can cancel. Its ranking need not match A or the
separate ON/OFF columns. Conversely B can hide wrong-polarity substitutions.
These views are complementary, not interchangeable. A averages over $2N$
channel cells while B/Signed average over $N$ cells; do not compare their raw
numbers as if they had the same interpretation.

### One-voxel examples

| GT (ON,OFF) | Prediction (ON,OFF) | A MAE/RMSE | B MAE/RMSE | Signed MAE/RMSE |
|---|---|---:|---:|---:|
| (10,5) | (12,7) | 2 | 4 | 0 |
| (10,0) | (0,10) | 10 | 0 | 20 |

MAE and RMSE coincide in these examples because both channel-error
magnitudes are equal and each view contains a single spatial/temporal cell.

## 5. Signed relative L1

CLI: `voxel_relative_l1`.

$$L1_{relative}=\frac{\sum_i|P_{s,i}-G_{s,i}|}{\sum_i|G_{s,i}|}.$$

Zero denominator produces an undefined result (`value` empty, status
`undefined`), not zero or an arbitrary epsilon replacement. The denominator
is **absolute signed GT mass**, not the total number of GT events; ON/OFF
cancellation can make it small or zero even in a nonempty stream. Values
greater than 1 are possible. This remains the existing Signed definition;
no A/B relative-L1 variants are currently implemented.

## 6. Adapted V2CE pooling MSE

CLI: `pmse_2_v2ce_adapted` and `pmse_4_v2ce_adapted`.

These use a **different preprocessing representation** from hard-count
MAE/RMSE: separate nonnegative polarity channels with forward temporal
weights. For an event in bin $b<B-1$:

$$w_b=\operatorname{clip}\left(\frac{\tau_{b+1}-t}{\Delta},0,1\right),
\qquad w_{b+1}=1-w_b.$$

Its contribution is $w_b$ to the current bin and $w_{b+1}$ to the next bin at
the same pixel/polarity. In the final bin, all mass remains in that bin.
Thus each included event contributes one unit. No events before start are
borrowed as context. Final-boundary folding is an explicit v2 convention.

Let $V_p,V_g$ be these tensors, of shape $(2,B,H,W)$. Partition time, height
and width into non-overlapping blocks with kernel=stride $k\in\{2,4\}$.
Polarity is never pooled. For a block $q$ with $n_q$ original cells:

$$\bar V_{c,q}=\frac1{n_q}\sum_{i\in q}V_{c,i}.$$

Retain incomplete edge blocks and average over their actual cell count:

$$PMSE^{adapted}_k=
\frac{\sum_{c\in\{+,-\}}\sum_q n_q
(\bar V_{p,c,q}-\bar V_{g,c,q})^2}{2\sum_q n_q}.$$

No image rows, columns or time bins are discarded. Cell-count weighting means
a tiny boundary block does not weigh as much as a full block. When all
dimensions divide $k$, weights are equal and this reduces to ordinary MSE
after average pooling. Small dimensions are supported. There is no subsequent
square root: units are squared events per voxel.

### Provenance and adaptations

[V2CE, arXiv v2, Section III-A](https://arxiv.org/html/2309.08891v2)
defines PMSE using 3D average pooling with kernel/stride 2 or 4, followed by
MSE. II-B motivates temporal contributions; II-A reports 10 bins per frame
pair at 30 FPS (about 3.33 ms per bin).
[Official V2CE repository](https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox).

The fixed-duration full-sequence grid, mass-preserving terminal folding,
partial-block retention and cell-weighted reduction are explicit benchmark
adaptations. Numerical equivalence to author code or published tables is not
claimed. Hence the `_adapted` suffix. A/B/Signed count errors are benchmark
definitions; they are not attributed to V2CE or another paper.

V2V voxelization is preprocessing, not a new pairwise stream error. No direct
V2E/EVIS stream-error metric is invented here. EQS is not implemented. See
[the provenance audit](VOXEL_METRIC_PROVENANCE.md) and
[the v2 protocol notes](VOXEL_METRIC_V2.md) for source-access limitations.

## 7. Outputs and dataset averaging

Each family writes `config.json` and `results.csv` to:

```text
REPORT_ROOT/DATASET/METRIC/test-v2-CONFIG_HASH/
```

For MAE/RMSE, CSV fields map as follows:

| Field | Meaning |
|---|---|
| `value` | Signed error, retained for backward compatibility |
| `value_on`, `value_off` | Separate polarity errors |
| `value_aware` | A, derived per sequence from ON/OFF |
| `value_total` | B, total local-count error |
| `temporal_bins` | Actual number of bins in that sequence |
| `bin_ms`, `last_bin_ms` | Nominal duration and actual final-bin duration |
| `start_s`, `stop_s` | Actual relative scoring bounds |
| `height`, `width` | Available saved scoring geometry |

PMSE and relative L1 use `value` only; empty unrelated columns are intentional.
`ok` rows are valid completed scores. `error` rows include an explanation;
`missing_prediction`, `not_applicable` and `undefined` are not zero errors.

The notebook/HTML uses finite completed rows and matched sequence cohorts.
For a set of $S$ matched sequences, each metric's dataset value is:

$$\text{dataset score}=\frac1S\sum_{s=1}^{S}\text{sequence score}_s.$$

Each sequence has equal weight, regardless of duration/event count. Sequence
RMSE is computed before this average; it is not a globally pooled dataset
RMSE. Different metric/channel cohorts may differ; check `n`. Curve cohorts
are fixed across available widths for each metric/channel. Missing models
or mismatching windows/geometry are excluded with coverage information.

## 8. View tables, images and interactive curves

Open `notebook/watch_voxel_metrics_v2.ipynb` and set:

```python
REPORT_ROOT = ROOT / 'reports_voxel_v2_ab'
DATASETS = ['bsergb']
BIN_WIDTHS_MS = [5, 10, 30]
```

Run All, Refresh reports, then Show bin curves. A/B/Signed are separately
selectable in the chart. ON/OFF and both PMSE variants remain available.
The four best distinct rank levels are shaded blue in the benchmark tables.
White-background voxel views overlay ON red, OFF blue and overlap purple;
3D views place time vertically. Rendering caps affect pictures only.

The optional plotting dependency is `plotly`. Export offline HTML either with
the notebook's Save standalone HTML checkbox, or:

```bash
python -m pip install plotly
python benchmarks/plot_voxel_bins_v2.py --datasets bsergb --report-root reports_voxel_v2_ab --output-dir reports_voxel_v2_ab/html
```

Plots can switch x between nominal width (ms) and actual mean bin count.
Hover shows mean/min/max bin counts, duration and matched sequence count.
Files are snapshots; regenerate after new scores finish. The chart exporter
uses all widths found, independently of the table width filter. Incompatible
code/protocol groups are separated; duplicate ambiguous runs require `--runs`.

## 9. Interpretation, reproducibility and limitations

- Use the same version, window, geometry and width settings across methods.
- A measures polarity-specific counts; B measures local total counts; Signed
  measures net counts. None establishes individual-event correspondence.
- Coarse bins and pooling can hide temporal misalignment. A smaller error at
  a different resolution does not by itself prove improved event fidelity.
- PMSE, MAE, RMSE and relative L1 have different scales/representations.
  Raw values across metrics are not comparable quality scores.
- The final short bin is not duration-normalized; do not call these rate errors.
- Older ON/OFF CSVs can supply A without rescoring. B needs event rescoring.
  Do not synthesize B from separate aggregate polarity errors.
- The runner recomputes on rerun and is not the v1 SQLite resume runner.
  Avoid concurrent writers into an identical report folder. Input changes
  during a score are detected; hashes/settings distinguish configurations.
- Dense PMSE grids can consume substantial memory at fine widths over long
  sequences. Count MAE/RMSE views are streamed; A/B add no separate passes.
- Neither scoring nor this documentation regenerates predictions, expands
  crops or modifies multi-dataset-loader.

## 10. Validation

```bash
python -m unittest benchmarks.test_voxel_metric benchmarks.test_voxel_metric_v2 benchmarks.test_plot_voxel_bins_v2 -v
```

With optional Plotly installed, all 21 tests passed at this revision. Tests
include analytical A/B/Signed cases, RMSE derivation before dataset averaging,
time-origin validation, short bins, boundary mass, corner coverage, pooling
weights, all-family CLI forwarding, legacy compatibility and offline HTML.
The notebook was also exercised with synthetic reports/events. Real-dataset
results and reproduction of paper tables are not claimed by these checks.
