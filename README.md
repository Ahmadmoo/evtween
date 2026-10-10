# evtween

Continuous-time event generation from RGB video, **learned from real events**.

Frames show the endpoints of a motion; events show the path between them.
A teacher learns a small **path code** `z` from real events. A video world model (V-JEPA 2.1 or LeVJEPA) learns to predict that code from frames alone.
A physics decoder turns the code into a log-intensity trajectory `L(u, τ)` with an analytic `dL/dτ`, and a crossing solver emits events with continuous timestamps.
At test time only RGB is needed.

```
TRAIN
 I0, I1 + real events in the gap ──> event encoder (teacher) ──> z        (per 16×16 patch)
 context frames, gap never shown  ──> V-JEPA 2.1 + predictor (student) ──> ẑ ≈ z
 z or ẑ + I0, I1 ──> physics decoder ──> L(u,τ), dL/dτ ──> exact-time event likelihood

TEST
 video ──> student ──> ẑ ──> physics decoder ──> crossing solver + sensor ──> events (t, x, y, p)
```

---

## 1. Parts

| Part | What it does |
|---|---|
| Teacher | ConvNeXt encoder: time-binned real events + I0, I1 → `z` on the 16×16 patch grid. Training only |
| Student | Frozen video backbone on frames before I0 and after I1 (each side encoded alone) + transformer predictor with real frame times and gap length → `ẑ` |
| Decoder | UNet features **gated by `z`**: every path coefficient is multiplied by a function of `z`, so `z = 0` gives exactly the plain SloMo path and the decoder cannot ignore the code |
| Path | Warp both frames with time-polynomial flows (RAFT + SloMo base + learned corrections), blend with a time-varying visibility mask. `L(0)`, `L(1)` equal the frames by construction |
| Sensor | One model for training and generation. After each event the pixel sets `ref = L` and draws thresholds `X·C` with `X ~ IG(1, k)`. ON fires when the running max of `(L − ref)/C_on` reaches `X_on` (OFF likewise). Background events at rate `ν` (per second, half per polarity) also reset the pixel. `C_on`, `r = C_off/C_on`, `k`, `ν`, refractory `R` are learned **per dataset** (each dataset is its own camera); brightness per dataset: learned RGB weights, then `PNG^gamma` (gamma from `data.sets`); under a Bayer filter (CED) each pixel uses its own channel |
| Events | `physics.Simulator` samples that same model with the learned parameters: exact crossing of the local quadratic of `L` with the random thresholds, plus background events. Mismatch and refractory are optional extras (off by default, not learned) |

### Losses

| Loss | Meaning |
|---|---|
| `nll` | Likelihood of the real event **times** under the sensor model above: signal hazard + background rate at each event, survival of both thresholds between events, after the last event and in silent pixels; the first wait per pixel starts from the stationary state |
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

Not tested here: real data on GPU, pretrained weight downloads, the LeVJEPA wrapper (Hugging Face is blocked in this sandbox).

Known limits:
- The sensor model ignores refractory time and low-light bandwidth; the likelihood reads `L` on a τ grid (`loss.grid`) with linear interpolation.
- The student is deterministic, so ambiguous gaps (same endpoints, different timing) get an average code. A stochastic student `p(z | video)` is the next step.
