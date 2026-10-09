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
| Sensor | One model for training and generation. After each event the pixel sets `ref = L` and draws thresholds `X·C` with `X ~ IG(1, k)`. ON fires when the running max of `(L − ref)/C_on` reaches `X_on` (OFF likewise). Background events at rate `ν` (per second, half per polarity) also reset the pixel. `C_on`, `r = C_off/C_on`, `k`, `ν` are learned |
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
| `data.py` | Sequence reader, training samples, collate, dataset list (`data.sets`) and the weighted mixing sampler |
| `train.py` | Stages `probe / teacher / student / joint`, single GPU or `torchrun` DDP, bf16, resume (the one training loop) |
| `run_training.py` | Command line for `train.py`: datasets, data fraction, weights, seeds, steps, logging, resume, final test |
| `experiment.py` | Experiment folder, dataset / batch reports, metric CSVs, checkpoints, provenance |
| `scripts/train.sbatch` | Slurm job for Rails: one GPU, saves before the time limit, requeues and resumes |
| `notebooks/inspect_training.ipynb` | Plots of an experiment folder (losses, per-dataset losses, validation, sampling, lr, memory) |
| `generate.py` | RGB video → events `.npz` (student path) |
| `HQ-EVFI/hqevfi.py` | Extracts the HQ-EVFI zip and converts it to the training layout |
| `BS-ERGB/bsergb.py` | Same for BS-ERGB (zip or extracted folders) |
| `CED/ced.py` | Same for CED (ROS1 bags, read without ROS; zips of bags accepted) |
| `EDS/eds.py` | Same for EDS (per-sequence archives); aligns the RGB frames to the event camera |
| `toy_data.py` | Synthetic dataset for smoke tests |
| `dataloader_visualization.ipynb` | Per dataset, chosen loader samples next to the original files (frames, events, timing) with exact checks; needs `jupyter` and `matplotlib` |
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

Four datasets, each converted once to the same layout, then mixed in training (`data.sets` in `config.yaml`).
Every converter takes the downloaded archive directly (`--archive`), extracts it once to `--raw`, and writes
`data/<name>/{train,test}/<sequence>/`. **Check the printed fps and event counts.**

| Dataset | Sensors | RGB | Split | Command |
|---|---|---|---|---|
| **HQ-EVFI** (TimeLens-XL, ECCV'24) | EVK4-HD, beam splitter, pixel-aligned | 142 fps | official (TimeLens-XL) | `python HQ-EVFI/hqevfi.py --archive HQ-EVFI.zip --out data/hqevfi` |
| **BS-ERGB** (Time Lens++, CVPR'22) | Gen4M, beam splitter, aligned 970x625 | ≈28 fps | official train / val / test | `python BS-ERGB/bsergb.py --archive bs_ergb.zip --out data/bsergb` |
| **CED** (CVPRW'19) | Color DAVIS346, same pixels | APS rate | every 5th per category | `python CED/ced.py --archive ced_*.zip --out data/ced` (or `--raw <folder of .bag>`) |
| **EDS** (CVPR'22) | Gen3.1 VGA + FLIR, beam splitter, **not** aligned | up to 75 fps | every 5th sequence | `python EDS/eds.py --archive <folder of .tgz> --out data/eds` |

Notes per dataset:

- **HQ-EVFI**: zip from [Google Drive](https://drive.google.com/file/d/104ZMJ-M_frImOOCGfLk_HDb2FV1trveT) (or put it in `HQ-EVFI/` and leave out `--archive`). Fetches the official ranges and test split from TimeLens-XL (`dataset_dict.py`, `--lists` for a local copy), uses the 3 ms corrected event folders with the one-frame image shift where TimeLens-XL does. Frame times come from the boundaries between event files.
- **BS-ERGB**: event x/y stored in 1/32 px are detected and rounded; event files TimeLens-XL marks as broken split their sequence. Without `--archive`, `--raw` (default `BS-ERGB/`) must hold `1_TEST`, `2_VALIDATION`, `3_TRAINING`.
- **CED**: frames from `image_raw` when it is a Bayer mosaic (linear light, `gamma.txt` = 1.0, pattern in `bayer.txt`), else from `image_color` (sRGB, `gamma.txt` = 2.2). `--dry` lists topics and encodings without converting.
- **EDS**: events stay at their own sensor pixels; RGB frames are warped into the event camera (undistort, rotate, distort with the [dataset calibration](https://github.com/uzh-rpg/bundles-eds/tree/master/config/data/dual_setup/03_calib), built in; `--calib` for another Kalibr camchain) and cropped to the region the RGB lens covers (≈495x360). The flip of the stored RGB images is picked from the data and printed with the residual misalignment; it should read `best residual shift dy,dx = (0, 0)`. Details in `<seq>/align.yaml`. Needs `h5py` and `hdf5plugin` for `events.h5`. `--dry` lists what was found in each sequence (it never extracts or writes anything). Already extracted sequence folders (`<seq>/events.h5`, `images/`, `images_timestamps.txt`): pass `--raw <their parent folder>` instead of `--archive`; `--only <name>` converts single sequences (the train/test split stays the same).

Layout per sequence:

```
seq/
  frames/000000.png ...   # RGB, aligned with the event sensor (symlinks, --copy to copy)
  frame_ts.npy            # (N,) float64, seconds
  ev_t.npy                # (M,) float64, seconds, sorted
  ev_x.npy  ev_y.npy      # (M,) int16
  ev_p.npy                # (M,) int8, +1 / -1
  gamma.txt               # optional: frame gamma of this sequence (CED)
```

### Mixing the datasets

```yaml
data:
  sets:
    - {name: hqevfi, root: data/hqevfi/train, val_root: data/hqevfi/test}
    - {name: bsergb, root: data/bsergb/train, val_root: data/bsergb/test, weight: 2}
    ...
  use: null        # e.g. [hqevfi, eds] to train on a subset
  gap: 0.035       # seconds between I0 and I1
```

- **Gap in seconds, not frames.** The datasets run at 28-142 fps, so `data.gap` sets the time between I0 and I1 and each sequence gets `skip = round(gap * fps) - 1` hidden frames (142 fps → 4, 75 → 2, 28 → 0). Pick it close to the frame interval of the videos you will convert. `data.gap: null` falls back to a fixed `data.skip`. Hidden frames are padded per batch (`mid_mask`); a dataset with skip 0 adds no photometric term.
- **Weights.** Each step draws a dataset by `weight` (default 1 each), then a sample inside it, so the large sets do not drown the small ones. Works with DDP.
- **Per set overrides:** `weight`, `gap` or `skip`, `gamma`, `min_events` (event density differs a lot between an EVK4-HD and a DAVIS346).
- **Gamma** of the frames: the set's `gamma`, else the sequence's `gamma.txt`, else `model.gamma`. It is sent with every sample, so one batch can mix linear and sRGB frames.
- **Validation** prints one line per dataset: `val[hqevfi] nll ...`.
- `data.crop` must fit the smallest frames of every set (CED 346x260, so at most 256).
- `data.root=<folder>` on the command line still trains on that one folder (the list is ignored).

---

## 5. Train

### `run_training.py` (recommended)

A command line over the same training loop (`train.run`); every run gets its own experiment folder.

```bash
# dry run: dataset report + one real batch per dataset + one mixed batch; no model, no optimizer step
python run_training.py --stage teacher --dry-run --exp runs/dryrun --raw hqevfi=HQ-EVFI/raw eds=EDS/EDS
# 1% of the data, 20 steps
python run_training.py --stage teacher --fraction 1% --steps 20 --batch 4 --log-every 1 --val-every 10 --save-every 10 \
    --val-batches 2 --exp runs/smoke_1pct
# 10% pilot, EDS at 25%, BS-ERGB sampled twice as often
python run_training.py --stage teacher --fraction 10% --dataset-fraction eds=25% --weights bsergb=2 --steps 10000 --exp runs/pilot_10pct
# full data, then the student from the teacher's best checkpoint
python run_training.py --stage teacher --steps 200000 --exp runs/teacher_full
python run_training.py --stage student --init runs/teacher_full/checkpoints/best.pt --exp runs/student_full
# continue an interrupted run (refused if learning settings or the data selection changed; --force-resume to accept)
python run_training.py --exp runs/teacher_full --stage teacher --steps 200000 --resume
# held-out test split, once, on request (never used during training)
python run_training.py --final-test runs/teacher_full/checkpoints/best.pt
# Slurm (Rails): submits, saves before the time limit, requeues and resumes by itself
sbatch scripts/train.sbatch runs/teacher_full --stage teacher --steps 200000
```

`python run_training.py --help` lists every flag; anything else: `--set section.key=value`.

| Topic | Behaviour |
|---|---|
| Splits | train = `root`; validation = the official `val_root` (BS-ERGB) or whole training **recordings** held out (`val_frac`, 10%, picked by `--data-seed`; pieces of one recording stay together); test = `test_root`, read only by `--final-test` |
| `--fraction` | share of the **eligible** training samples per dataset (after the split and after dropping sequences too short for one sample). Each sample gets a hash of (data seed, dataset, sequence, frame) and the lowest `ceil(f*N)` are kept: reproducible, and 1% ⊂ 5% ⊂ 10% … `--select sequence` keeps whole sequences instead. Only indices are kept; nothing is loaded |
| `--weights` | share of drawn samples per dataset (default equal); `size` = proportional to the selected samples. Batches are mixed: each sample picks its dataset independently |
| Seeds | `--seed`: augmentation, sampling order, initialization. `--data-seed`: validation split and data selection |
| Augmentation | online only: random crop (re-drawn up to 10 times to reach `min_events`), horizontal flip; validation: centre crop. The number of samples does not change |
| Logs | console + `training.log`: every `--log-every` steps the interval means of every loss term, lr, grad norm, GPU memory, s/step, samples/s, and per dataset the share of samples and its losses (`N/A` where a term does not apply, e.g. `photo` without hidden frames; `sigreg` is a whole-batch statistic, no per-dataset value) |
| Guards | non-finite loss or gradient: the step is not taken, `nonfinite_step*_rank*.json` names the batch's sequences and frames, the run stops. An existing experiment folder is never overwritten (use `--resume`) |
| Checkpoints | `checkpoints/last.pt` (every `--save-every` steps, written atomically), `step_*.pt` (last `--keep`), `best.pt` (lowest mean validation `nll`, or `nll_student` for student / joint). They hold the trainable weights, optimizer, step, sample counts, config, data selection and the random states of every GPU; resume continues the same sample order |

Experiment folder: `config.yaml`, `dataset_report.json`, `train_metrics.csv`, `val_metrics.csv`, `training.log`,
`checkpoints/`, `summary.json`, `env_*.json` (host, GPUs, versions, Slurm job, git commit / branch / status) and
`git_diff_*.patch` if the code had uncommitted changes. `notebooks/inspect_training.ipynb` plots them (no training).

Monitor on Rails: `squeue -u $USER`, `tail -f runs/<exp>/training.log`, `sacct -j <job> -o JobID,State,Elapsed,ExitCode`,
`nvidia-smi` on the node (`srun --jobid <job> --pty nvidia-smi`).

### `train.py` with dotted overrides

The same loop, configured only from `config.yaml` and `key=value`; the experiment folder is `train.out`.
Run the stages in order; add `torchrun --nproc_per_node 4` for multi-GPU.

```bash
# 0. probe (1-2 days): can the backbone predict event counts in the gap? compare backbones
python train.py train.stage=probe model.backbone=vjepa2_1 train.out=runs/probe_vjepa train.steps=20000
python train.py train.stage=probe model.backbone=levjepa  train.out=runs/probe_levjepa train.steps=20000
python train.py train.stage=probe model.backbone=none     train.out=runs/probe_none train.steps=20000

# 1. teacher + decoder on real events
python train.py train.stage=teacher train.out=runs/teacher

# 2. student (teacher and decoder frozen)
python train.py train.stage=student train.init=runs/teacher/checkpoints/best.pt train.out=runs/student

# 3. optional joint fine-tune (small lr)
python train.py train.stage=joint train.init=runs/student/checkpoints/best.pt train.lr=2e-5 train.out=runs/joint
```

### What to watch (validation line)

| Value | Meaning | Healthy |
|---|---|---|
| `nll` | Teacher code | Lowest |
| `nll_shuffled` | Codes swapped between samples | Clearly above `nll`, else `z` carries nothing |
| `nll_zero` | Plain SloMo path | Above `nll` |
| `nll_student` | Student code | Between `nll` and `nll_zero` |
| `gap` | `nll_student − nll`: in-between information the video cannot predict | Small; report it vs gap length |

An existing `train.out` is not overwritten: continue it with `train.resume=true`. Checkpoints leave out the frozen RAFT and backbone weights.

### Smoke test (CPU, a few minutes)

```bash
python toy_data.py data/toy 24
S="data.root=data/toy/train data.val_root=data/toy/val data.crop=64 data.gap=null data.skip=3 data.min_events=50 data.workers=2 data.bins=8 \
   model.flow_prior=none model.backbone=none model.width=16 model.depth=[1,1,1] model.pred_dim=64 model.pred_depth=2 \
   loss.grid=16 train.batch=4 train.lr=1e-3 train.warmup=20 train.log_every=100 train.val_batches=8"
python train.py $S train.stage=teacher train.steps=1000 train.save_every=500 train.out=runs/teacher
python train.py $S train.stage=student train.init=runs/teacher/checkpoints/last.pt train.steps=600 train.save_every=300 train.out=runs/student
python generate.py --ckpt runs/student/checkpoints/last.pt --seq data/toy/val/seq100 --out runs/student/events.npz
```

---

## 6. Generate

```bash
python generate.py --ckpt runs/student/checkpoints/best.pt --seq my_video/ --out events.npz
python generate.py --ckpt ... --seq ... --set noise=0 mismatch=0.05 refractory=5e-4
```

`my_video/` needs only `frames/` and `frame_ts.npy`. Output: `t` (s, float64), `x`, `y` (int16), `p` (int8 ±1), sorted by time.

---

## 7. Main knobs

| Key | Effect |
|---|---|
| `data.gap` | Seconds between I0 and I1 (hidden frames per sequence follow from its fps). Choose it close to the frame interval of the videos you will convert |
| `data.sets`, `data.use` | Datasets and their sampling weights; train on a subset with `data.use=[hqevfi,eds]` |
| `data.skip` | Hidden frames in the gap when `data.gap` is null |
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

Multi-dataset branch: all four converters and the archive input checked on small fake data in each dataset's format
(EDS: flip and alignment recovered exactly through the dataset calibration); the loader, mixing sampler and padded
collate checked against those outputs without torch. A real `train.py` run over the four sets is the first GPU check.

Not tested here: real data on GPU, pretrained weight downloads, the LeVJEPA wrapper (Hugging Face is blocked in this sandbox).

Known limits:
- The sensor model ignores refractory time and low-light bandwidth; the likelihood reads `L` on a τ grid (`loss.grid`) with linear interpolation.
- The student is deterministic, so ambiguous gaps (same endpoints, different timing) get an average code. A stochastic student `p(z | video)` is the next step.
