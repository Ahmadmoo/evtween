# evtween

Continuous-time event generation from RGB video, **learned from real events**.

Frames show the endpoints of a motion; events show the path between them.
A teacher learns a small **path code** `z` from real events. A video world model (V-JEPA 2.1 or LeVJEPA) learns to predict that code from frames alone.
A physics decoder turns the code into a log-intensity trajectory `L(u, τ)` with an analytic `dL/dτ`, and a crossing solver emits events with continuous timestamps.
The pixels see the **least-event path** inside a tube around `L` as wide as the frame noise, so noise in the frames makes no events.
At test time only RGB is needed.

```
TRAIN
 I0, I1 + real events in the gap ──> event encoder (teacher) ──> z        (per 16×16 patch)
 context frames, gap never shown  ──> V-JEPA 2.1 + predictor (student) ──> ẑ ≈ z
 z or ẑ + I0, I1 ──> physics decoder ──> L(u,τ) ──> taut path in the noise tube ──> exact-time event likelihood

TEST
 video ──> student ──> ẑ ──> physics decoder ──> L(u,τ) ──> taut path ──> crossing solver + sensor ──> events (t, x, y, p)
```

### This version (`v2-tube`)

Two changes against `v2-sensor-teacher`, both for the salt-and-pepper noise in static regions, which was strongest at the end of the gap:

| Change | Why |
|---|---|
| **Timing warp pinned at the frames** (`losses.event_nll`) | The arrival-time uncertainty used to delay the whole path by `d`, so the last `d` of the path fell outside the window and was never scored. A change there was almost free (≤ 0.8 nats per pixel at `sig` 0.03), and the teacher learned to put its leftover brightness changes in the last few percent of the gap. Now events at `t` see `L(φ(t))` with `φ(t) = t − d·sin(πt)`: off by `d` mid-gap, exact at both frames, so every `d` scores the whole path. `d = 0` is the old loss |
| **Tube** (`model.tube`) | The path must end at frame 1, so frame noise between I0 and I1 had to be crossed somewhere and made events. The pixels now see the taut string in the tube `|g − L| ≤ ε`: the path with the fewest threshold crossings that stays within `ε` of `L`, with free ends. Changes smaller than the noise make no events, real edges stay |

The two go together: with the hole closed but no tube, the frame noise has to be paid for somewhere, and the threshold `c` would absorb it.

Tube width, no hand-set value:
- **Measured noise:** during training, at pixels with no event in the gap, `L1 − L0` is pure frame noise. A running table (like BatchNorm statistics, saved in the checkpoint) keeps `σ_L = median|L1 − L0| / 0.6745 / √2` per dataset and per brightness bin (16 bins of `log(Y + eps)`).
- **Width:** `ε = 1.96/√2 · σ_L`, so 95% of pixels that only see noise can stay flat (`L1 − L0 ~ N(0, 2σ²)` fits in `2ε`).
- **Learned factor:** the decoder gives one more output, a factor `exp(g)` on `ε` (×0.05 to ×2.7, ×1 at `z = 0`). The events narrow the tube where they show a real small change.
- **Taut path:** the mean of the forward and backward play-operator paths (a DVS pixel is itself a play operator). Each value is one value of `L ± ε`, so backward routes every gradient to that value and keeps only an int8 per step.

---

## 1. Parts

| Part | What it does |
|---|---|
| Teacher | ConvNeXt encoder: time-binned real events + I0, I1 → `z` on the 16×16 patch grid. Training only |
| Student | Frozen video backbone on frames before I0 and after I1 (each side encoded alone) + transformer predictor with real frame times and gap length → `ẑ` |
| Decoder | UNet features **gated by `z`**: every path coefficient is multiplied by a function of `z`, so `z = 0` gives exactly the plain SloMo path and the decoder cannot ignore the code |
| Path | Warp both frames with time-polynomial flows (RAFT + SloMo base + learned corrections), blend with a time-varying visibility mask. `L(0)`, `L(1)` equal the frames by construction |
| Tube | `model.taut`: the least-event path within `ε` of `L` (free ends), `ε` = measured frame noise × learned factor. Used by both the likelihood and the sampler (`model.tube`) |
| Sensor | One model for training and generation. After each event the pixel sets `ref = L` and draws thresholds `X·C` with `X ~ IG(1, k)`. ON fires when the running max of `(L − ref)/C_on` reaches `X_on` (OFF likewise). Background events at rate `ν` (per second, half per polarity) also reset the pixel. `C_on`, `r = C_off/C_on` and refractory `R` are learned **per dataset** (each dataset is its own camera); the noise (`k` via `model.sigma_event`, `ν` via `model.nu`) is fixed in the config, because when learned it soaks up the path's errors and the sampler spreads them as noise; brightness per dataset: learned RGB weights, then `PNG^gamma` (gamma from `data.sets`); under a Bayer filter (CED) each pixel uses its own channel |
| Events | `physics.Simulator` samples that same model with the learned parameters: exact crossing of the local quadratic of `L` with the random thresholds, plus background events. With the tube: the taut path on a grid of at least 128 steps, exact crossing of its linear pieces; between gaps the pixel reference moves with the path's free start, so the jump fires nothing |

### Losses

| Loss | Meaning |
|---|---|
| `nll` | Likelihood of the real event **times** under the sensor model above: signal hazard + background rate at each event, survival of both thresholds between events, after the last event and in silent pixels; the first wait per pixel starts from the stationary state. With `model.uncertainty` it is averaged over timing errors `d ~ N(0, sig²)` through a warp pinned at both frames (the whole path is always scored) |
| `photo` | `L(τ)` must match the real hidden frames |
| `cmax` | Contrast maximization: real events moved along the model flow to τ=0 and τ=1 must stack into sharp edges (scale-normalized) |
| `sigreg` | LeJEPA SIGReg on `z`: keeps the code isotropic Gaussian (no collapse, easy to model later) |
| `jepa` | Student code vs teacher code (`stopgrad` in the student stage, joint in the joint stage) |
| `smooth` | Edge-aware smoothness of the path coefficients |

---

## 2. Files

| File | Content |
|---|---|
| `model.py` | Backbones, teacher, student, gated decoder, flows, `render(s, τ) → L, dL/dτ` |
| `losses.py` | Likelihood, photometric, contrast max, SIGReg, stage logic, validation diagnostics |
| `physics.py` | Quadratic root solver and event `Simulator` |
| `data.py` | Sequence reader, training samples, collate, joint sampling over the datasets in `data.sets` |
| `train.py` | Stages `probe / teacher / student / joint`, single GPU or `torchrun` DDP, bf16, resume |
| `generate.py` | RGB video → events `.npz` (student path) |
| `HQ-EVFI/hqevfi.py`, `BS-ERGB/bsergb.py`, `EDS/eds.py`, `CED/ced.py` | One converter per dataset: the download (or its extracted folder) → the training layout |
| `check_data.py` | Checks every converted sequence, and the event / frame alignment per dataset (no model needed) |
| `toy_data.py` | Synthetic dataset for smoke tests |
| `config.yaml` | All settings |

### Code used as-is from other projects

| Part | Source | How |
|---|---|---|
| RAFT | torchvision | `raft_large(weights=Raft_Large_Weights.DEFAULT)` |
| V-JEPA 2.1 ViT-L | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | `torch.hub.load(..., "vjepa2_1_vit_large_384", pretrained=False)`, then the official checkpoint (`ema_encoder`). The repo's hub file points downloads to `localhost`, so the URL is set in `model.py` |
| LeVJEPA ViT-L | [galilai-group/LeVJEPA-VideoMix-Large](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large) | `AutoModel.from_pretrained(..., trust_remote_code=True)` |
| SIGReg | [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa) | `SlicingUnivariateTest(EppsPulley(n_points=17), num_slices=256)` |

---

## 3. Install

```bash
pip install -r requirements.txt
```

---

## 4. Data

Four RGB + event datasets, trained together (`data.sets` in `config.yaml`). One script per dataset turns the download into the training layout.

| Dataset | Event camera | Frames | Test split | Download | Script |
|---|---|---|---|---|---|
| HQ-EVFI (TimeLens-XL, ECCV'24) | Prophesee EVK4-HD, beam splitter, pixel-aligned | RGB, 142 fps | official (TimeLens-XL `dataset_dict.py`) | [Google Drive zip](https://drive.google.com/file/d/104ZMJ-M_frImOOCGfLk_HDb2FV1trveT) | `HQ-EVFI/hqevfi.py` |
| BS-ERGB (Time Lens++, CVPR'22) | Prophesee Gen4M, beam splitter, aligned | RGB, 970x625, ~20-28 fps | official (train / val / test) | [timelens-pp](https://github.com/uzh-rpg/timelens-pp) | `BS-ERGB/bsergb.py` |
| EDS (CVPR'22) | Prophesee Gen3 640x480, beam splitter | RGB, 640x480, 75 Hz | sequences 02, 06, 10, 14 | ["Archive file" per sequence](https://rpg.ifi.uzh.ch/eds.html) | `EDS/eds.py` |
| CED (CVPRW'19) | Color DAVIS346: same pixels as the frames, RGBG Bayer | `image_raw` demosaiced (rggb), linear | every 5th sequence of each category | [zip per category of ROS bags](https://rpg.ifi.uzh.ch/CED.html) | `CED/ced.py` |

```bash
python HQ-EVFI/hqevfi.py --src HQ-EVFI/HQ-EVFI.zip --out data/hqevfi   # also fetches TimeLens-XL dataset_dict.py (or --lists)
python BS-ERGB/bsergb.py --src BS-ERGB/bs_ergb.zip --out data/bsergb
python EDS/eds.py --src EDS/*.tgz --out data/eds                       # --mid-exposure if check_data.py says so
python CED/ced.py --src CED/*.zip --out data/ced
```

- `--src` takes the downloaded archives, or the folders they were extracted to (frames are then hard-linked: no extra disk).
- Zips are read in place. EDS archives and the HQ-EVFI inner zips are unpacked one at a time to `<out>/.tmp` and removed.
- A finished sequence gets a `done` file: a second run skips it and redoes unfinished ones. `--dry` only lists sequences and splits.
- BS-ERGB: Time Lens++ Evaluation License (non-commercial internal evaluation, no derivatives).

Layout per sequence (`<out>/{train,val,test}/<sequence>/`, plus `<out>/info.json` with camera, frames, split and source):

```
seq/
  frames/000000.png ...   # RGB, aligned with the event pixels
  frame_ts.npy            # (N,) float64, seconds, 0 at the first frame
  ev_t.npy                # (M,) float64, seconds, same clock, sorted
  ev_x.npy  ev_y.npy      # (M,) int16
  ev_p.npy                # (M,) int8, +1 / -1
  bayer.txt               # CED only: color filter of the event pixels (top-left 2x2 in reading order)
  done                    # conversion finished
```

Check the data before training:

```bash
python check_data.py data/hqevfi data/bsergb data/eds data/ced   # --full reads every event (slow on 1e9-event sequences)
```

- Every sequence: frames, times, events (order, pixel range, polarity, events per frame interval, cover of the frames). Problems are marked `!`.
- Every dataset, without the model: time offset of the frames, pixel shift and gamma, from the net events vs the log change between neighbouring frames. The time offset uses pixels that change the same way over 3 frame intervals (a pixel that turns around delays its events). `check.py` does the same time check with a trained model.

Joint training, `data.sets` in `config.yaml`:

- Each entry: `root` (holds `train/` and `val/` or `test/`), `skip`, `weight`.
- `skip`: frames hidden between I0 and I1. Choose `(skip+1)/fps` close to the frame interval of the videos you will convert. Defaults give 25-50 ms gaps (HQ-EVFI 4, BS-ERGB 0, EDS 2, CED 0).
- Every batch comes from one dataset (same number of hidden frames); dataset d with probability ∝ `weight`.
- Every sample carries `ds` (position in `data.sets`; stays the same when an entry is set to `null`) and `cfa` (2x2 color filter of the crop, -1 without a filter).
- Validation: fixed samples per dataset (`val/`, else `test/`), one line per dataset.
- Gaps (I0 → I1) without a single event are left out: holes in the recording, not still scenes.

---

## 5. Train

Run the stages in order. Every command accepts dotted overrides; add `torchrun --nproc_per_node 4` for multi-GPU.

```bash
# 0. probe (1-2 days): can the backbone predict event counts in the gap? compare backbones
python train.py train.stage=probe model.backbone=vjepa2_1 train.out=runs/probe_vjepa train.steps=20000
python train.py train.stage=probe model.backbone=levjepa  train.out=runs/probe_levjepa train.steps=20000
python train.py train.stage=probe model.backbone=none     train.out=runs/probe_none train.steps=20000

# 1. teacher + decoder on real events
python train.py train.stage=teacher train.out=runs/teacher

# 2. student (teacher and decoder frozen)
python train.py train.stage=student train.init=runs/teacher/last.pt train.out=runs/student

# 3. optional joint fine-tune (small lr)
python train.py train.stage=joint train.init=runs/student/last.pt train.lr=2e-5 train.out=runs/joint

# fine-tune a teacher trained without the tube: the decoder gets one more output, the old ones are kept
python train.py train.stage=teacher train.init=runs/teacher/last.pt train.out=runs/teacher_tube
```

### What to watch (validation line)

| Value | Meaning | Healthy |
|---|---|---|
| `nll` | Teacher code | Lowest |
| `nll_shuffled` | Codes swapped between samples | Clearly above `nll`, else `z` carries nothing |
| `nll_zero` | Plain SloMo path | Above `nll` |
| `nll_student` | Student code | Between `nll` and `nll_zero` |
| `gap` | `nll_student − nll`: in-between information the video cannot predict | Small; report it vs gap length |

The run resumes from `train.out/last.pt`. Checkpoints leave out the frozen RAFT and backbone weights.

### Smoke test (CPU, a few minutes)

```bash
python toy_data.py data/toy 24
S="data.sets={\"toy\":{\"root\":\"data/toy\",\"skip\":3}} data.crop=64 data.min_events=50 data.workers=2 data.bins=8 \
   model.flow_prior=none model.backbone=none model.width=16 model.depth=[1,1,1] model.pred_dim=64 model.pred_depth=2 \
   loss.grid=16 train.batch=4 train.lr=1e-3 train.warmup=20 train.log_every=100 train.val_batches=8"
python train.py $S train.stage=teacher train.steps=1000 train.save_every=500 train.out=runs/teacher
python train.py $S train.stage=student train.init=runs/teacher/last.pt train.steps=600 train.save_every=300 train.out=runs/student
python generate.py --ckpt runs/student/last.pt --seq data/toy/val/seq100 --out runs/student/events.npz
```

---

## 6. Generate

```bash
python generate.py --ckpt runs/student/last.pt --seq my_video/ --out events.npz --dataset bsergb   # whose sensor / brightness mapping
python generate.py --ckpt ... --seq ... --set noise=0 mismatch=0.05 refractory=5e-4
```

`my_video/` needs only `frames/` and `frame_ts.npy`. Output: `t` (s, float64), `x`, `y` (int16), `p` (int8 ±1), sorted by time.

---

## 7. Main knobs

| Key | Effect |
|---|---|
| `data.sets` | Datasets trained together: `root`, `skip` (hidden frames in the gap; `(skip+1)/fps` close to the frame interval of the videos you will convert), `weight` (share of batches), `gamma` (PNG → linear light, from `check_data.py`) |
| `data.context` | Frames per side for the student, taken at the gap's own stride, so training sees the same frame spacing as generation (even for V-JEPA 2.1) |
| `model.backbone` | `vjepa2_1`, `levjepa`, or `none` (no world knowledge) |
| `model.z_dim` | Size of the path code per patch |
| `model.sigma_event`, `model.nu` | Sensor noise, fixed (threshold spread per event as a fraction of c, background events / px / s); `null` = learned |
| `model.sig_max` | Cap on the per-pixel arrival-time uncertainty used by the likelihood (training only) |
| `model.tube` | Events from the least-event path in the frame-noise tube; the width is measured, not set. `false` = the old path (old configs build without it) |
| `model.K` | Polynomial order of the path in time |
| `loss.grid` | τ steps for the likelihood clocks |
| `sensor.*` | Generation only: `noise` scales the learned background rate; `mismatch`, `refractory` are optional extras; step limits |

---

## 8. Status

Verified here (CPU, toy data):
- `dL/dτ` matches finite differences of the model, also when samples leave the image; inverse-Gaussian terms match numerical integration.
- Simulator and likelihood are the same model: on simulated events the likelihood is lowest at the true `k`, `ν`, `C_on`, `r` and the true motion.
- Teacher generalizes on held-out toy scenes: `nll` < `nll_zero` < `nll_shuffled`. Student lands in between.
- All four stages, generation, V-JEPA 2.1 wrapper (official code, random weights), 2-process DDP.
- `v2-tube`:
  - Taut path: values equal, and gradients agree to 1e-14, with a plain autograd version; `gradcheck` passes; the path stays in the tube.
  - Old configs (`model.uncertainty: false`, no tube) give the same loss, gradients and sampled events as `v2-sensor-teacher`.
  - Loss hole: a pixel with no events whose path rises by 1.5 c now costs the same whether the rise is mid-gap or in the last 2%. The old loss made the late rise 5.2 nats cheaper at `sig` 0.03.
  - Teacher fine-tune from a checkpoint without the tube, generation with the tube, 2-process DDP.

Not tested here: real data on GPU, pretrained weight downloads, the LeVJEPA wrapper (Hugging Face is blocked in this sandbox).

Known limits:
- The sensor model ignores refractory time and low-light bandwidth; the likelihood reads `L` on a τ grid (`loss.grid`) with linear interpolation.
- The noise table uses pixels without events; tiny real changes below the threshold are counted as noise, so `σ_L` is slightly too large. The learned factor can narrow the tube.
- The tube adds a loop over the τ grid (forward and backward in time) per likelihood call.
- The student is deterministic, so ambiguous gaps (same endpoints, different timing) get an average code. A stochastic student `p(z | video)` is the next step.
