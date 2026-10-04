# evtween

Continuous-time event generation from RGB video, **learned from real events**.

Given two frames, the model predicts the full log-intensity trajectory `L(u, τ)` between them.
It is trained only on real RGB + event pairs (no simulator in the loop).
At test time it needs **only RGB**. Events come out with continuous timestamps from an exact crossing solver, similar to TIDES.

---

## 1. Idea

| Part | What it does |
|---|---|
| Trajectory | Warp both frames with time-polynomial flows, then blend them with a time-varying visibility mask |
| Flows | Super SloMo base flows from a frozen RAFT prior, plus learned corrections `Σ a_k τ^(k+1)` and `Σ b_k (1-τ)^(k+1)` |
| Exact `dL/dτ` | Closed-form chain rule through warp and blend. Uses the exact bilinear gradient, so it matches finite differences to ~1e-9 |
| Endpoints | `L(0)` and `L(1)` equal the input frames exactly, by construction |
| Sensor | Contrast threshold `C_on` and ratio `r = C_off / C_on` are learned from data |
| Events | Per pixel, solve where the local quadratic of `L` crosses `ref ± C`. Adds threshold mismatch, refractory period and background noise |

### Losses

| Loss | Meaning |
|---|---|
| `cos` | Per patch, the direction of the predicted `ΔL` must match the event counts `n_on - r·n_off`. Does not depend on `C` |
| `fit` | `|ΔL - C·(n_on - r·n_off)|` is free below one threshold (quantization), Huber above it |
| `photo` | `L(τ)` must match the real skipped frames |
| `smooth` | Edge-aware smoothness of the predicted coefficients |

`ΔL` is always measured against a known frame (from `τ=0` forward, and to `τ=1` backward), with a small Gaussian blur.
This keeps quantization noise bounded while the signal grows. On toy data it gave 4× more learning signal than short independent intervals.

---

## 2. Files

| File | Content |
|---|---|
| `model.py` | UNet (ConvNeXt blocks), RAFT prior, trajectory `render(s, τ) → L, dL/dτ` |
| `losses.py` | Event, photometric and smoothness losses |
| `physics.py` | Quadratic root solver and event `Simulator` |
| `data.py` | Sequence reader and training dataset |
| `train.py` | Training: single GPU, `torchrun` DDP, bf16, resume |
| `generate.py` | RGB video → events `.npz` |
| `convert.py` | Converts per-interval npz datasets (HS-ERGB, BS-ERGB, TimeLens style) |
| `toy_data.py` | Small synthetic dataset for smoke tests |
| `config.yaml` | All settings |

---

## 3. Install

```bash
pip install -r requirements.txt
```

---

## 4. Data

One folder per sequence:

```
seq/
  frames/000000.png ...   # RGB, aligned with the event sensor
  frame_ts.npy            # (N,) float64, seconds
  ev_t.npy                # (M,) float64, seconds, sorted
  ev_x.npy  ev_y.npy      # (M,) int16
  ev_p.npy                # (M,) int8, +1 / -1
```

`data.root` points to a folder that holds many `seq/` folders.

### Convert HS-ERGB (example)

```bash
python convert.py \
  --images  hsergb/close/test/baloon_popping/images_corrected \
  --timestamps hsergb/close/test/baloon_popping/images_corrected/timestamp.txt \
  --events  hsergb/close/test/baloon_popping/events_aligned \
  --out     data/hsergb/train/baloon_popping \
  --xy_scale 32 --t_scale 1e-6
```

> Check the printed time ranges. Frames and events must both be in seconds. Fix with `--t_scale` / `--ts_scale`.

Good training sources: HS-ERGB, BS-ERGB, ERF-X170FPS, HQ-EVFI (aligned RGB + events, high frame rate).

---

## 5. Train

```bash
# one GPU
python train.py config.yaml

# 4 GPUs
torchrun --nproc_per_node 4 train.py config.yaml

# override any key
python train.py config.yaml train.lr=1e-4 data.skip=15 model.K=4 train.out=runs/k4
```

The run resumes from `train.out/last.pt` if it exists.

Log line: `cos fit photo smooth` losses, the learned `c` and `r`, grad norm, lr.

### Smoke test (CPU, ~1 min)

```bash
python toy_data.py data/toy
python train.py data.root=data/toy/train data.crop=64 data.skip=3 data.min_events=50 data.workers=0 \
  model.flow_prior=none model.width=16 "model.depth=[1,1,1]" train.batch=4 train.steps=100 train.warmup=10 train.out=runs/toy
python generate.py --ckpt runs/toy/last.pt --seq data/toy/val/seq100 --out runs/toy/events.npz
```

---

## 6. Generate

```bash
python generate.py --ckpt runs/default/last.pt --seq my_video/ --out events.npz
python generate.py --ckpt ... --seq ... --set noise_rate=0 mismatch=0.05 refractory=5e-4
```

`my_video/` needs only `frames/` and `frame_ts.npy`.
Output: `t` (s, float64), `x`, `y` (int16), `p` (int8 ±1), sorted by time.

---

## 7. Main knobs

| Key | Effect |
|---|---|
| `data.skip` | Gap between keyframes. Larger = harder motion, more supervision per sample |
| `data.n_tau` | Random time cuts per sample for the event loss |
| `model.K` | Polynomial order of the motion and visibility in time |
| `model.flow_prior` | `raft` (default) or `none` |
| `model.eps`, `model.gamma` | Log offset and display gamma. Must match how the sensor sees light |
| `loss.blur`, `loss.margin` | Quantization handling in the event losses |
| `sensor.*` | Generation only: mismatch, refractory, noise, step limits |

---

## 8. Verified

- `dL/dτ` matches finite differences (rel. error ~1e-9 inside the interval, ~1e-6 at the endpoints).
- After every interval, each pixel's level stays within its threshold band, so no crossing is missed.
- Toy run: the loss goes down and the learned `c` and `r` move toward the true values.
- RAFT path, 2-process DDP (gloo, CPU) and resume all run.

Not tested here: a full GPU run on real data.
