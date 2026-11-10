# Event-stream error metrics: provenance and protocol

Audit: 2026-11-10. Target branch: `event-evaluation-v52-v53`, base commit
`179a71f98f1f54e1d415f7c8adaacde35b72b233`. Implementation is original NumPy
code from the cited definitions, not copied official evaluation code. No
multi-dataset-loader or synced reference material was modified.

## Implemented: V2CE Pooling MSE

- Paper/version: https://arxiv.org/html/2309.08891v2 (2024-04-26), II-B and III-A.
- Official release: https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox
- Functions: `pmse_2_v2ce`, `pmse_4_v2ce` in `voxel_metric.py`.
- Formula: `mean((AvgPool3D_k(pred) - AvgPool3D_k(gt))**2)`, stride `k`.

The paper explicitly specifies **3D** average pooling with kernel and stride
2 or 4. Pool time and the two spatial axes, preserving polarity as a channel.
The representation follows the forward box-integration description in II-B:
for normalized time `u`, contribute `1-frac(u)` to `floor(u)` and `frac(u)` to
the following bin. This helper is private preprocessing, not a metric.
No official evaluator was recovered/pinned to a commit; numerical parity with
the authors' release or published tables is **not claimed**.

### Explicit benchmark conventions and differences

The paper uses 10 bins per frame pair. This benchmark instead uses **one uniform
grid over the entire common scoring interval**, independent of internal frame
timestamps. Default: 160 bins, a benchmark choice, not a universal paper setting.
Increasing sequence duration with fixed bin count changes temporal resolution.
Both streams use identical bounds, bin count, geometry and crop. Bounds come
from source frame timestamps, never from each stream's first/last event.

Convention: `[start, stop)`. Events outside this interval are ignored by the
private voxelizer; the runner additionally rejects predictions outside its
declared scoring domain. Forward contributions crossing `stop` are discarded;
there is no context before `start`, extra terminal bin or boundary renormalization.
These edge choices are explicit benchmark conventions, not verified author-code
behavior. ON is `p>0`; OFF is `p=0` or `p=-1`; channels never cancel.

Pooling is unpadded, floor/valid pooling. Trailing incomplete blocks are dropped,
including temporal blocks. Dimensions smaller than the kernel raise an error.
Mean reduction weights all retained pooled cells and both polarities equally,
including empty cells. No per-stream, nonzero-only, maximum, count or variance
normalization occurs. Paper visualization normalization is not used as metric
normalization. Empty-vs-empty is zero. Output units are squared events/voxel.
Dense float64 grids require memory proportional to `2*T*H*W` per stream; this
implementation is not an out-of-core evaluator.

## Other methods audited; no unsupported additions

| Method | Primary paper and official code | Finding / decision |
|---|---|---|
| V2V | https://arxiv.org/html/2505.16797v1 ; https://github.com/HYLZ-2019/V2V | Sections 4.2/4.3 and appendix A evaluate reconstructed-image MSE/SSIM/LPIPS and flow AEE/3PE. They are downstream errors. Interpolated event voxels are preprocessing, not a stream error. No `*_v2v` error invented. |
| V2E | https://arxiv.org/html/2006.07722 ; https://github.com/SensorsINI/v2e | Sections 5/6 discuss representation, recognition and detection. These do not justify attribution of a direct GT/generated voxel error to V2E. |
| EVIS | https://arxiv.org/html/2607.08098v2 ; https://github.com/spikelab-jhu/isaac-sim-event-camera-plugin | September 2026 version evaluates reconstruction, matching, tracking, rotation speed and throughput. Those are not direct pairwise stream errors. Repository internals were not exhaustively audited. |
| EQS | https://arxiv.org/html/2504.12515v1 ; https://github.com/eventbasedvision/EQS | Sections 4/5 compare RVT feature patches. The paper includes cosine distance `1-cos` but also describes larger similarity scores as better. Exact checkpoint, recurrent-state, temporal preprocessing and aggregation conventions were not verified in an executable evaluator. Not implemented. |

EQS official file inspected:
https://github.com/eventbasedvision/EQS/blob/main/eqs.py — at audit it contains
a VGG LPIPS image optimization example, not the described RVT event evaluator;
README says it will be updated. Generic LPIPS or voxel MSE is not substituted
for EQS. Published CVF PDF retrieval returned 403, but the arXiv HTML was read.
V2V conference PDF exceeded the retrieval size limit; arXiv v1 HTML was read.
Some V2CE raw/branch URLs failed; its official repository landing page and
paper were accessible. These limitations preclude author-code parity claims.

V2CE's occupancy F1/accuracy and timestamp/count metrics were identified, but
not added: additional averaging, empty-case and matching conventions need
confirmation. Training losses are not registered as evaluation methods.

## Existing APIs and runner migration

`voxel_mae`, `voxel_rmse`, `voxel_relative_l1` and the two polarity functions
retain their signatures and array arithmetic. They are legacy benchmark
metrics, not attributed to any of the papers above. For event-stream input,
the runner now bins the full shared interval uniformly for all `voxel_*`
metrics (histogram signed counts, as before, without PMSE temporal interpolation).
Spatial `--voxel-block-size` retains its previous partial-block averaging.

Native V2V files contain frame-interval signed arrays, without recoverable
event timestamps or separate polarities. Default full-interval metrics return
`not_applicable` for them. PMSE always returns `not_applicable` for native V2V.
For historical reproduction only, `--legacy-frame-voxels` restores frame-pair
binning for legacy `voxel_*` metrics, including native V2V. The flag is rejected
for PMSE. Do not compare scores across these protocols.

The runner records temporal bins, protocol, normalization, PMSE representation
and pooling conventions in `config.json`; existing code hashes and settings
digest create separate report/cache folders. Resume cannot reuse an old
frame-pair report as a new full-interval report.

Example (from repository root):

```sh
python benchmarks/run_metric.py eds test --metric pmse_2_v2ce --voxel-temporal-bins 160 --methods v2ce
python benchmarks/run_metric.py eds test --metric pmse_4_v2ce --voxel-temporal-bins 160 --methods v2ce
python -m unittest benchmarks.test_voxel_metric -v
```

## Validation

No pre-existing test suite/configuration was found in this branch. Added nine
unittest cases covering analytical PMSE values, polarity separation, fractional
temporal contributions, endpoint clipping, temporal pooling, valid-pool tails,
invalid inputs, identity/empty inputs, quadratic count scaling (no normalization),
fixed shared alignment, invariance to internal frame segmentation and legacy
array APIs. A synthetic three-frame dataset exercises the actual CLI subprocess,
CSV/status output, PMSE-2/4, full-interval legacy MAE, resume and explicit native
V2V legacy scoring. All nine passed. CLI help and `git diff --check` were also
checked. Real dataset performance and published table reproduction were not run.
