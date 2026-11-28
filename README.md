# evtween

Continuous-time event generation from RGB video, **learned from real events**.

Frames show the endpoints of a motion; events show the path between them.
A teacher learns a small **path code** `z` from real events. A video world model (V-JEPA 2.1 or LeVJEPA) learns to predict that code from frames alone.
A physics decoder turns the code into a log-intensity trajectory `L(u, τ)` (the frames warped along learned flows and blended), and a crossing solver emits events with continuous timestamps.
The pixels see the **least-event path** inside a tube around `L` as wide as the measured frame noise, so noise in the frames makes no crossings.
**Residual events** come on top at a learned rate per pixel and polarity: a part tied to the slope of the raw path, which fires like a pixel does (each time its running count passes a whole number), and a random background part. The thresholds and the tube width are **measured** from the data, not learned.
At test time only RGB is needed.

```
TRAIN
 I0, I1 + real events in the gap ──> event encoder (teacher) ──> z        (per 8×8 patch)
 context frames, gap never shown  ──> V-JEPA 2.1 + predictor (student) ──> ẑ ≈ z
 z or ẑ + I0, I1 ──> physics decoder ──> L(u,τ) ──> taut path in the noise tube ──> crossings  ─┐
                                    └──> κ, ν maps ──> residual rate (raw slope of L + background) ┴─> likelihood, x-t / y-t, earth mover's, PMSE

TEST
 video ──> student ──> ẑ ──> physics decoder ──> taut path + residual rate ──> crossing solver + sensor ──> events (t, x, y, p)
```

### This version (`v8.4`)

Built from `v8.3`. On real data the v8.3 teacher (real events in) draws sharp edges and little noise; the student draws thick, blurry edges, about twice the noise in quiet scenes, and its ON % moves 1–5 points between versions. Three causes: the student had no pixels (only 16 px video tokens), the decoder was frozen on teacher codes, and training crops were always busy. The teacher is v8.3's; only stage 2 changes.

| Change | v8.3 | v8.4 | Why |
|---|---|---|---|
| Student input | Video tokens only (16 px) | + I0, I1 at full resolution through a conv encoder (the teacher's class, frames only, 2.3M), added on the code grid (8 px), then 2 ConvNeXt blocks there (2.4M) | Edges need 8 px detail; the tokens have 16 px. The encoder's last layer starts at 0, so training starts from a v8.3 student |
| Decoder in stage 2 | Frozen | Trained with the student; the teacher stays frozen and still gives `z_t` for `jepa` | It only ever learned teacher codes; at test time it only reads student codes |
| Stage 2 losses | `jepa`, `nll_student`, `emd_student`, `pmse_student` | + `xt_student`: the teacher's event losses, all on the student's events | The student had no curve-level (x-t) loss |
| Training crops | Retried until ≥ `data.min_events` events | 1 in 4 is a plain random crop | The model never saw quiet, noisy crops: noise in static scenes, ON / OFF balance fit on busy crops only |

- With the decoder moving to the student, the teacher's own validation numbers (`nll`, `mask`, `pmse`, `gap`) get worse in stage 2; that is expected. The student's count: `nll_student`, `xt_student`, `emd_student`, `pmse_student`.
- Start stage 2 from the v8.3 student: a random student head would feed noise codes to the decoder at the start.
- The probe stage now also gets the pixel branch; for a backbone-only probe, use v8.3.

```bash
torchrun --nproc_per_node 4 train.py train.stage=student train.init=runs/student_v8.3/last.pt model.pred_dim=384 train.steps=20000 train.lr=1e-4 train.out=runs/student_v8.4
```

Checks (CPU, toy):
- A v8.3 student loads; the new parts start at (almost) nothing: the code changes by 0.4 % (max 0.007).
- When no crop can pass, 102 of 400 samples (26 %) take a single crop.
- 400 more student steps from the same v8.3 toy student, v8.3 code vs v8.4 code, 3 val sequences × 10 frame pairs:

| | Chamfer | ON % (GT 45.6) | IEI | Fano | PMSE-2 | PMSE-4 | count (× GT) |
|---|---|---|---|---|---|---|---|
| v8.3 | **1.87** | 46.6 | 0.275 | 0.545 | 0.0139 | 0.0052 | **0.95** |
| v8.4 | 1.94 | **46.2** | **0.147** | **0.528** | **0.0120** | **0.0034** | 1.22 |

- On toy the v8.4 student makes more events (1.22× GT); watch the count on real data. The toy has no video backbone and its random crops are all busy (no quiet crops), so it tests the code, not the gain.
- All five stages and `generate.py` run.

### `v8.3` (separate branch)

Built from `v8.2`, for scattered, broken edges. On real data most generated events were drawn per pixel at random (slope + background: 61–80 % of the teacher's events, 71–86 % of the student's), and independent random events can't form lines (v6-wiggle's finding). The likelihood also rewards spreading events where the model is unsure. The model is v8.2's: the sampler draws slope events differently, and two losses are added.

| Change | v8.2 | v8.3 | Why |
|---|---|---|---|
| Slope events (sampler) | Poisson per pixel, polarity by coin flip | Each pixel adds up its expected slope events per polarity and fires each time the sum passes a whole number (uniform start phase, kept between gaps). The background stays Poisson | As a real pixel: every pixel an edge passes fires, no holes. Same expected count |
| `loss.emd` | – | Earth mover's distance between the expected and the real events, per polarity, along t (per pixel, ms), x (per row and bin, px) and y (per column and bin, px): shapes only, per real event. Teacher and student | The likelihood pays to cover every place an event might be (a cloud); this pays for how far events are off, so a sharp line beats a cloud. Chamfer is a distance too |
| `loss.pmse` | – | PMSE of a sampled stream as its expected value, `(E[N] − G)² + Var[N]`, on 2×2×2 and 4×4×4 blocks (16 time slices × px): crossings and slope events ~0 variance, background Poisson. Relative to the real blocks' mean square. Teacher and student | Trains PMSE directly; random events cost their variance |

- The likelihood still scores slope events as Poisson with the same rate; only the sampler fires them on schedule.
- Validation also prints `emd`, `pmse` and the student's `emd_student`, `pmse_student`.
- `loss.emd` 0.02: its value is a distance per event (about 25 on toy crops), so the term stays near the size of `nll`.
- Train teacher and student from scratch (v8.2 checkpoints also load: same model). The extra cost of the two losses was not measured on GPU.

```bash
torchrun --nproc_per_node 4 train.py train.stage=teacher train.steps=40000 train.out=runs/teacher_v8.3
torchrun --nproc_per_node 4 train.py train.stage=student train.init=runs/teacher_v8.3/last.pt model.pred_dim=384 train.out=runs/student_v8.3
```

Checks:
- Sampler alone, on the real v8.2 student (first 40 pairs, 50 sequences): Chamfer 4.844 → 4.839, PMSE-2 0.0312 → 0.0294, PMSE-4 0.0121 → 0.0118, IEI 0.233 → 0.253; Fano and ON % unchanged.
- Sampler (CPU): with constant rates every pixel's slope events are within 1 of the expected count (64×64 px, 4 gaps); background count and ON share as expected; a ramp gives crossings only.
- `emd` on one real event in a 256 px row: sharp in place 6.1, 3 px off 8.8, spread ±5 px 8.6, half the mass 11.2, 10× the mass 6.1, nothing 67. Predicting fewer events never lowers it.
- Toy (CPU), trained from scratch with the same budgets (teacher 800, student 400 steps), 3 val sequences × 10 frame pairs:

| | Chamfer | ON % (GT 45.6) | IEI | Fano | PMSE-2 | PMSE-4 | count |
|---|---|---|---|---|---|---|---|
| v8.2 | 2.29 | 41.7 | 0.313 | 0.650 | 0.0132 | 0.0046 | 0.23 |
| v8.3 | **1.91** | **46.5** | **0.295** | **0.577** | 0.0133 | 0.0049 | **0.15** |

- Tested and left out: two κ maps (ON only on rises, OFF only on falls). Alone: ON 32 % (GT 46 %), PMSE-2 0.019. With the two losses: Chamfer 1.82, but PMSE-2 0.0148 and PMSE-4 0.0061.

### `v8.2` (separate branch)

Built from `v8` (conv student head; v8.1's attention head is not in here). Two training changes; the model is v8's.

| Change | v8 | v8.2 | Why |
|---|---|---|---|
| Gap length | One `skip` per dataset | `skip: [lo, hi]`: one skip per batch, drawn evenly from lo..hi (hqevfi 0–4, eds 0–2, bsergb 0). The context frames follow the batch's stride | Generation and the paper's evaluation run on every frame pair (1-frame gaps); v8 saw only 3–5-frame gaps on hqevfi and eds and made far too few events on short gaps. The student's gap-length input now means something instead of naming the dataset |
| Masked slices (`loss.mask`) | – | A second teacher pass on a quarter of each batch with 2–3 of 8 time slices of its events hidden (set to −1; log counts are ≥ 0, so hidden is not empty). The decoder's path from that code must still make the hidden slices' events: the x-t loss with the visible slices set to the real counts, so only the hidden ones count | v5.3's trick, through the decoder instead of a separate head: the code must describe the whole gap from part of the events, closer to what the student does from frames alone |

- Validation: a skip range gives two lines per dataset, `val hqevfi skip4` and `val hqevfi skip0`. `mask` is the masked-slice score (same hidden slices at every validation; lower is better).
- Cost: the masked pass adds about a quarter of a teacher step; validation of hqevfi and eds takes twice as long (two skips).
- The measured frame noise and thresholds now also see short gaps (`nll_zero` moves, see the checks). Watch `c` per dataset.
- Train the teacher from scratch. v8 checkpoints load; a single `skip` number works as before.

```bash
torchrun --nproc_per_node 4 train.py train.stage=teacher train.steps=40000 train.out=runs/teacher_v8.2
torchrun --nproc_per_node 4 train.py train.stage=student train.init=runs/teacher_v8.2/last.pt model.pred_dim=384 train.out=runs/student_v8.2
```

Checks (CPU, toy):
- Skip range: the batch's skip sets the hidden frames, `dt` and the context stride; one skip per batch, drawn evenly (2,000 batches: 458 / 500 / 502 / 540 for skips 0–3).
- Hidden slices leak nothing: more events in a hidden slice change the masked code by 0.0, in a visible slice by 1.1. With no slice hidden the masked loss is 0 (1.6e-12). The masked loss reaches the teacher, the decoder and the thresholds; every stage and `model.tube` option gets gradients.
- Toy teachers, 800 steps each, toy gaps of 4 frames (skip 3) and of 1 frame (skip 0); events: expected per sampled draw against the real ones:

| Teacher | `nll` skip 3 | events skip 3 (real 56.1k) | `nll` skip 0 | events skip 0 (real 12.3k) |
|---|---|---|---|---|
| v8 (skip 3) | **0.452** | 56.5k | 0.432 | 4.5k (× 0.37) |
| skip 0–3 only | 0.478 | 58.3k | **0.382** | **7.4k (× 0.60)** |
| mask only (skip 3) | 0.483 | 56.2k | 0.407 | 5.4k (× 0.44) |
| v8.2 (both) | 0.567 | 59.9k | 0.387 | 7.0k (× 0.57) |

- Toy students (400 steps) on the last two teachers, skip 0–3: `nll_student` 0.944 / 0.485 (skip 3 / 0) on the skip-only teacher, **0.923 / 0.472** on the v8.2 teacher (gap 0.44 → 0.33 and 0.07 → 0.05). The masked teacher is worse on its own at skip 3, but its code is easier for the student, and the student is what generates.
- Random skip is what fixes short gaps (× 0.37 → × 0.60 of the real events); they are still short of the real count on toy. On real data compare against `loss.mask=0`.
- `generate.py`, `eval_interp.py`, `check.py` run.

### `v8` (separate branch)

Built from `v7`. Three parts removed and the student's head replaced; everything else is v7.

| Part | v7 | v8 | Why |
|---|---|---|---|
| RAFT | Frozen flows f01, f10: base motion (SloMo formula) and UNet input (with warp errors) | Gone. The decoder's flows are the whole motion: `F0 = a₁τ + a₂τ²`, `F1 = b₁(1−τ) + b₂(1−τ)²`, each term within ±`model.flow_scale` (64 px) | One fewer pretrained part; the teacher sees the motion in the events |
| UNet input | 12 channels: I0, I1, flows, warp errors | 6 channels: I0, I1 | No flows to give |
| `dL/dτ` | Exact slope from `render` (warp gradients) | Gone: `render` gives `L` and `Y`; slopes come from the grid | Training never used it; only the sampler did |
| Sampler | Quadratic pieces without a tube, linear pieces on the grid with it | Always linear pieces on a grid of at least 128 steps (finer where the path is fast) | One code path |
| Refractory time `R` | Learned per dataset (it kept rising to 0.2–0.4 ms) | Gone: after a crossing the pixel takes `ref = L` at once | ISI medians are 3–5 ms; the blind time touched few events and added a lot of code |
| Student head | Transformer: queries with an absolute position code, global attention over all context tokens | Conv head: the context time slots stacked on channels, `model.pred_depth` ConvNeXt blocks on the token grid, then 2 × 2 cells per token; no position code, no attention | Full frames failed: the position code took values never seen on 256 crops and attention spread over ~10× more tokens. A conv does the same at every spot, so any image size works |

What to expect:
- `z = 0` is now a cross-fade of the two frames with no motion, so `nll_zero` is much higher than in v7. Compare `nll` and `nll_student`, not `nll_zero`.
- Watch BS-ERGB (largest motions): the flows must now come from the decoder alone. Compare its `nll` with v7's teacher.
- v7 checkpoints don't load whole (the UNet's first layer changed and the refractory parameter is gone): train from scratch.
- The student runs on any image size: train on crops, run on full frames. `model.pred_depth` is now the number of conv blocks (8: 11.8M parameters at `pred_dim` 384, 20.2M at 512, without the backbone).
- `check.py`'s time-shift test uses the path at `z = 0`, which has no motion in v8, so it says little; use `check_align.py` for alignment.

Checks (CPU, toy):
- **Sampler = likelihood**, on a moving texture: 2,389 residual + background events sampled per draw against 2,400 expected; the nll is lowest at the true threshold (× 0.8 / 1 / 1.25), `κ` and `ν` (× 0.5 / 1 / 2).
- **Gradients** reach the teacher, decoder, student, `κ` and `ν` rows in every stage and with every `model.tube` option; the interp stage changes only the teacher.
- **Toy training** (800 teacher, 400 student, 300 interp steps, same settings as v7's toy run): teacher `nll` 0.48 (v7: 0.42), `nll_student` 1.00 (v7: 1.00), interp `photo` 0.0189 → 0.0160. On the training-style samples the student samples 47.5k events against v7's 48.8k (real 56k). `generate.py`, `check.py`, `eval_interp.py` run.
- **Student head**: shifting the frames by 16 px shifts the code by exactly 2 cells (max difference 0), so no absolute position is left. Toy student, same settings and about the same size (132k vs 142k parameters): gap `nll_student − nll` 0.43 for the conv head vs 0.52 for the transformer on training-size crops, 0.35 vs 0.47 on full toy frames. Both backbones (with LoRA) run at crop and larger sizes; gradients reach only the head and LoRA.
- On gaps of one frame, shorter than the toy training gaps (4 frames), both versions make too few events (v8 4.2k, v7 10.4k, real 12.3k); the crossings are the same, the learned residual rate differs. Train with `skip` close to the gaps you will generate.

```bash
torchrun --nproc_per_node 4 train.py train.stage=teacher train.steps=40000 train.out=runs/teacher_v8
torchrun --nproc_per_node 4 train.py train.stage=student train.init=runs/teacher_v8/last.pt train.out=runs/student_v8
```

### `v7` (separate branch)

Built from `v5.2`, trained from scratch. The aim: fewer hand-made parts, more capacity. The decoder switch-off test on v5.2 (all datasets) showed which hand-made parts carry weight.

| Change | v5.2 | v7 | Why |
|---|---|---|---|
| Tube width (`model.tube`) | Measured × a learned factor map (e^−3 to e^1) | Measured only (`measured`); `scalar` = × one learned factor per dataset; `false` = no tube | The factor map sat at both limits (p5 0.05, max 2.72; median 0.07 on HQ-EVFI): an on/off switch. The tube itself stays: tube off was the worst switch-off (chamfer +1.94, BS-ERGB +4.03) |
| Texture events | Rate `a·\|dL/dτ\|` of the **taut** path, a learned share `q` with the slope's polarity | **Residual events**: per pixel and polarity, `(κ_rise·rise + κ_fall·fall) / C_pol` of the **raw** path, plus a background `ν_pol` (6 decoder maps; they don't reset the pixel) | One learned rate replaces texture, `q`, the fixed `ν` and `nll_pol`. It reads the raw path, so a real change the tube flattens can still make events where `z` says so. On the taut path such a pixel would get no events at all |
| Background `ν` | Fixed 0.1 /px/s | Per pixel and polarity from the decoder; `model.nu` (0.1) at `z = 0` | Part of the residual rate |
| Arrival-time uncertainty `σ` (`sig_max`, `sig_patch`) | On | Removed | Switched off: nll +0.028, no change in generated events |
| Flow correction terms `K` | 3 | 2 | Without the 3rd term: nll +0.038 |
| Visibility | Cubic (tied to `K`) | Cubic (own constant `VIS` in `model.py`) | `V` constant in τ: nll +0.87 |
| Losses (teacher) | nll, nll_pol, xt, photo, smooth, sigreg | nll, xt, sigreg | `photo` moves to the interp stage; `smooth` and `cmax` dropped |
| Decoder (`model.depth`) | [4, 4, 8, 4], 15.3M | [4, 4, 12, 8], 26.0M | More capacity |
| Student predictor (`model.pred_dim`, `pred_depth`) | 384-d, 4 layers, 9.9M | 512-d, 8 layers, ~34M | More capacity |
| Interp stage (`train.stage=interp`) | – | Teacher only (decoder, sensor and student frozen), nll + photo; the noise table and thresholds are not updated | Frame interpolation as a separate fine-tune at the end |

Removed from the code (still on their branches): detail image, wiggle, painter, pattern loss, texture, `q`, `σ`, `nll_pol`, `smooth`, `cmax`. Checkpoints from older versions do not load whole (the decoder outputs changed).

The interp stage needs hidden frames. BS-ERGB is trained with `skip: 0`, so give it one hidden frame there (`data.sets.bsergb.skip=1`), or leave it out.

Checks (CPU, toy):
- **Sampler = likelihood.** 64×64 px with a big rise, a big fall, a small change below the tube, and frame noise. Over 20 draws the sampler makes 1,308 residual + background events per draw; the likelihood expects 1,314. The nll over `κ` × (0.5, 1, 2) and `ν` × (0.5, 1, 2) is lowest at the true values. Without the tube: 1,291 vs 1,314.
- **Raw slope.** On the small-change strip the taut path moves 0.006 (raw path 0.029): no crossings there, but residual events still come. Where the taut path is flat (78% of the pixels of a toy batch), the residual rate's gradient still reaches the raw path at every pixel.
- **Gradients.** In every stage and with every `model.tube` option, the `κ` and `ν` decoder rows get gradient. The interp stage changes only the teacher: decoder, sensor, student, noise table and thresholds stay bit for bit.
- **Toy training** (800 teacher, 400 student, 300 interp steps): validation `nll` 0.42 < `nll_student` 1.00 < `nll_shuffled` 1.06 < `nll_zero` 2.01; `res_share` 0.59. Interp: `photo` 0.0176 → 0.0156, `nll` 0.42 → 0.40. `generate.py` (learned and clean sensor), `check.py`, `eval_interp.py --student` run.

```bash
# 1. teacher + decoder
torchrun --nproc_per_node 4 train.py train.stage=teacher train.steps=80000 train.out=runs/teacher_v7
# 2. student (teacher and decoder frozen)
torchrun --nproc_per_node 4 train.py train.stage=student train.init=runs/teacher_v7/last.pt train.out=runs/student_v7
# 3. interp: teacher only, nll + photo (small lr)
torchrun --nproc_per_node 4 train.py train.stage=interp train.init=runs/student_v7/last.pt data.sets.bsergb.skip=1 \
    train.lr=5e-5 train.warmup=500 train.steps=20000 train.out=runs/interp_v7
python eval_interp.py --ckpt runs/interp_v7/last.pt --split test --n 200

# optional: LoRA fine-tune of the video backbone (V-JEPA 2.1 or LeVJEPA), from the trained student (head first, then LoRA)
torchrun --nproc_per_node 4 train.py train.stage=student train.init=runs/student_v7/last.pt model.lora_rank=16 model.lora_blocks=8 \
    train.lr=2e-5 train.lora_lr=5 train.warmup=500 train.steps=15000 train.out=runs/student_v7_lora

# tube ablation (teacher stage, everything else the same)
torchrun --nproc_per_node 4 train.py train.stage=teacher model.tube=scalar train.steps=80000 train.out=runs/teacher_v7_scalar
torchrun --nproc_per_node 4 train.py train.stage=teacher model.tube=false  train.steps=80000 train.out=runs/teacher_v7_notube
```

#### LoRA on the video backbone (optional)

`model.lora_rank > 0` adds LoRA (`W x + B A x`) to the attention (`qkv`, `proj`) of the last `model.lora_blocks` blocks of the video backbone (`vjepa2_1` or `levjepa`); everything else in the backbone stays frozen. It trains in the student (and joint) stage with its own learning rate (`train.lr × train.lora_lr`, no weight decay), and its weights are saved in the checkpoint (the frozen backbone is not).

- Start it from a trained student (`train.init`): `B` starts at 0, so step 0 predicts exactly what the frozen student predicts. Training a new head and LoRA together lets the head's early noise pull the backbone features around.
- Gradients reach only the LoRA blocks: nothing is stored for the frozen blocks before them, so memory grows with `lora_blocks`, not with the backbone's 24 blocks.
- Decide by `gap` on validation per dataset against the frozen student: smaller gap = the frozen features were the limit; a smaller train gap with a larger val gap = overfitting (lower `lora_rank` or `lora_blocks`).
- Checked here (official V-JEPA 2.1 code and LeVJEPA's v1 ViT from its GitHub repo, random weights): with `B = 0` the output equals the frozen backbone exactly; only the LoRA weights get gradients and only the LoRA blocks build a graph; teacher and decoder stay bit for bit; resume, `generate.py` and `eval_interp.py` run from a LoRA checkpoint.

### `v5.2` (separate branch)

v5.1's validation numbers flattened out (`nll_pol` ≈ 0.48 at every check, `q` stuck near 0.6–0.67): the path cannot put an edge at the right time at each pixel, so events fall back on texture. Two limits sit before the losses: one path code per 16×16 px, and the teacher sees events in 16 time bins (~2.3 ms each) while the losses check timing at ±1 ms. Trained from scratch:

| Change | v5.1 | v5.2 |
|---|---|---|
| Path code cell (`model.patch`) | 16 px | 8 px: one code per 8×8 px; the teacher has one level less, the student predicts on the 8×8 grid (2×2 queries per V-JEPA token); the decoder already resizes z |
| Teacher time bins (`data.bins`) | 16 | 32 (~1.1 ms each) |
| Teacher blocks per level (`model.enc_depth`) | 1 | 6 (teacher 8.0M parameters with the v5 width; trainable total ~33M) |
| `loss.photo` | 1 | 10: at 1 its value (~0.05) barely pulled the path toward the real frames |
| `loss.xt` | 1 | 2 |

Datasets trained with `skip: 0` (BS-ERGB) have no hidden frames, so no `photo` loss; see 4. Data.

Checks (CPU, toy): with `patch: 16`, `enc_depth: 1` the teacher and student losses and all gradients equal v5.1, bit for bit (and old v5 / v6 configs as before); at patch 8: teacher, student, `generate.py`, `eval_interp.py`.

```bash
torchrun --nproc_per_node 4 train.py train.stage=teacher train.steps=80000 train.out=runs/teacher_v52
```

### `v5.1` (separate branch)

v5 fine-tuned, to fix its main problem: **ON and OFF mixed on each curve** in x-t / y-t. The cause is in v5's loss. v5's texture events take a coin-flip polarity at a rate `a·|dL/dτ|` that is highest on the curves, so the likelihood can explain any wrong-polarity event on a curve as cheap texture, and no other loss checks polarity. Three changes to the likelihood, plus the x-t / y-t loss (which v5 never had):

| Change | What it does |
|---|---|
| **Texture follows the slope's polarity** (`model.texture_pol`) | A share `q` of the texture events takes the polarity of the path's slope, `1 − q` the other one: per unit τ, ON `a·(q·rise + (1−q)·fall)/C_on`, OFF `a·(q·fall + (1−q)·rise)/C_off`. `q` is learned per dataset (`log_q_odds`, own learning rate `train.q_lr` × `train.lr`) and starts at 0.5, which is exactly v5. GT has 9–21% opposite-polarity events near curves, so `q` should settle near 0.8–0.9. Likelihood, its cost term, the x-t loss and the sampler all use it |
| **Polarity part again** (`loss.nll_pol` 1.0, `losses.polarity_loss`) | Every event's score splits into "an event here, now" and "this polarity". This weighs the second part again: per real event, of the events the model expects at that pixel within ±`loss.pol_win` ms (1 ms; crossings, texture with its share, background), the share with the event's polarity; −log of it, averaged. It asks only which polarity |
| **Timing shift shared per patch** (`loss.sig_patch` 8) | The likelihood's timing uncertainty (a shift `d ~ N(0, sig²)` it averages over) is shared by each 8×8 block of pixels, not drawn per pixel. Wrong motion moves a whole region's events together; scattering single pixels in time now costs more |
| **x-t / y-t loss on** (`loss.xt` 1.0) | ON and OFF maps on blurred x-t / y-t slices, texture counted with its polarity share |

`config.yaml` holds v5's sizes (`model.depth [4, 4, 8, 4]`, `model.z_dim 64`, `data.min_events 1500`), so a v5 checkpoint loads whole.

Checks (CPU, toy: a soft edge moving 20 px per gap over a slowly varying background, v5 texture):
- **Old configs** (v5, tube only, v6 wiggle): same losses, gradients and sampled events as before, bit for bit.
- **`q = 0.5` is v5:** same nll, same expected texture, same events (10,773 = 10,773).
- **Sampler = likelihood:** 5,657–5,676 texture events where 5,601 are expected; share of texture events on the edge with its polarity 0.51 at `q = 0.5`, 0.86 at `q = 0.85`.
- **The likelihood finds `q`:** on events made with `q = 0.85`, the nll over `q ∈ {0.5, 0.7, 0.8, 0.85, 0.9, 0.97}` is lowest at 0.85.
- **Polarity term:** 0.27 on the model's own events, 0.37 with `q = 0.5`, 2.39 with every polarity flipped (coin flip: 0.69); its gradient reaches the path, the texture and `q`.
- **Shared timing shift:** moving a whole curve 0.6 ms late costs +0.10 (per pixel: +0.19); moving each pixel 0.6 ms early or late at random costs +0.36 (per pixel: +0.20).
- **From a v5-config checkpoint:** loads whole, `q` starts at 0.5; teacher (with resume), student, `generate.py`, the v5.1 notebook.

Train (empty `train.out`):

```bash
torchrun --nproc_per_node 4 train.py train.stage=teacher train.init=runs/teacher_v5/last.pt train.lr=1e-4 train.steps=60000 train.out=runs/teacher_v51
```

The validation line adds `xt`, `nll_pol` (lower = polarity closer to GT) and `q`. The nll is not comparable with v5's: the shared timing shift scores differently.

### `v6-wiggle` (separate branch)

Built on `v5-texture`, fine-tuned from a v5 checkpoint. Trained longer, v5 gets the number and place of events close to GT (0.8–1.35× GT events, against 0.15–0.77× for v4; lower nll on every checked sample), but:
- **ON and OFF are mixed on each curve** in x-t / y-t. v5's texture events take a coin-flip polarity, and their rate `a·|dL/dτ|` is highest exactly on the curves, so an OFF curve gets about as many texture ON events as OFF.
- **The dots form a cloud around the curves, not streaks**: every pixel draws its texture events alone.

Real texture is brightness, not events: small wiggles riding on the path. On a steep curve a wiggle only moves the curve's events in time (to make an event of the other polarity it would have to rise a full threshold against the fall); where the path is flat and the surface moves, it makes up-and-down events: a bright blob gives ON on its front and OFF on its back.

| Change | What it does |
|---|---|
| **Wiggle** (`model.wiggle`, replaces `model.texture`) | One decoder output per pixel: the wiggle size `A` (log brightness, 0.01 at `z = 0`, up to 1.5). **On curves:** the thresholds get the wider spread `1/k_A = 1/k + 2A² / c̄²` (`c̄ = (C_on + C_off)/2`, `model.wiggle_k`): the curve's events scatter in time, with the curve's polarity. **Pairs:** per unit τ and per polarity, `A · speed · exp(−|dL/dτ| / (A · speed)) / (C_on + C_off)` (`model.wiggle_rate`), with `speed = |dF0/dτ|` the speed (px per gap) of the image point the pixel sees: a wiggle of `A` per pixel of motion, and none where the path's own change outruns the wiggle. Pair events don't reset the pixel; the likelihood sums over which events were crossings with v5's forward pass (`loss.mix`). The slope and speed in the rate carry no gradient: the wiggle size is learned, the path and flow are not bent to make texture |
| **Random texture moved by the flow** (`physics.Simulator`; with `model.painter` this is the painter's starting point, below) | A smooth random brightness pattern on the frame-0 plane (white noise blurred over `sensor.grain` = 2 px, also outside frame 0), scaled per pixel so one pixel of motion changes it by `A` on average, seen by each pixel at `p + F0(τ)`. A pixel fires each time the pattern crosses a rung of a ladder with rungs `(C_on + C_off)/2` apart, at its own random height: ON going up, OFF going down, kept with probability `exp(−|dL/dτ| / (A · speed))`. The expected count is the likelihood's. Nearby pixels see the same blob, so texture events form soft patches along the motion with one polarity on each blob edge. A new texture every gap. (The first v6 sampler drew single-pixel specks instead: thin straight lines in x-y, regular dotted columns in x-t / y-t, and OFF and ON one pixel apart; replaced, same checkpoint) |
| **x-t / y-t loss counts texture** (`loss.xt` 1.0) | The texture events the model expects (pairs, or v5's texture) are added to the expected ON / OFF counts, so a 50/50 texture inside a one-polarity curve costs directly |
| From `v6-detail` (kept, off) | The x-t / y-t loss, decoder outputs matched by name on `train.init`, the detail image (`model.detail: false`) |

**Learned texture (latest).** Every texture rule above was hand-made, and the losses can't shape it: the nll scores each pixel alone, and gradients can't go through drawn events. So the texture's look is now learned:

| Change | What it does |
|---|---|
| **Texture painter** (`model.painter`, `model.Painter`) | A small conv net (4 dilated layers, reach 15 px, ~30k parameters): noise + frame features → the wiggle's texture on the frame-0 plane. It starts as the hand-made texture above (blurred noise times a learned scale; its last layer starts at 0). The texture is scaled so one pixel of motion along the flow changes it by 1 on average over ~16 px around, so the count of texture events stays the likelihood's and the painter only sets the look. It needs no events: teacher and student draw texture the same way. Own learning rate (`train.painter_lr` × `train.lr`): it starts from scratch even in a fine-tune |
| **Pattern loss** (`loss.stats` 1.0, `losses.stats_loss`) | Model events (the path's, plus one texture drawn by the painter) against real events, compared by statistics of x-y-t blocks (16 of the gap's 64 time bins × 32 × 32 px, 8 random blocks per sample) at the same place, not pixel by pixel. A block holds every x-t and y-t slice through it. Default `stats_kind: features`: the mean rectified response of 64 fixed random 3D filters (5×5×5 over ON/OFF, two scales), i.e. which small x-y-t patterns occur and how often (an OFF line with an ON line one pixel behind it, a slanted streak, a lone dot); `spectrum`: band powers of ON + OFF and ON − OFF. Squared difference of log. Both sides are draws of events: the path's with fixed thresholds from a random start, the texture's as the sampler makes them; the gradient comes through the texture's expected count (straight-through). Trains the painter and the wiggle size, not the path |
| **Path alone** (`sensor.texture_events: false`, notebook `TEXTURE = False`) | Generation without texture events (v5's or the wiggle's), to see the curves alone |

Checks for the learned texture (CPU, toy):
- **Old configs** (v5 texture, tube only): same losses, gradients and sampled events as before, bit for bit. Wiggle without painter: same losses and gradients; its sampler now scales the texture locally along the flow (counts 0.97× the likelihood's, painter or not).
- **Gradients:** the pattern loss reaches the painter and the wiggle row of the decoder; the path's rows get exactly 0.
- **Learning a hidden texture from events only:** flat frames, a hidden texture of thin slanted dark lines moving 15 px per gap. The true texture drawn again scores best on both statistics (the loss is calibrated; a first version compared the expected count plus a Poisson floor and was not: real sensor events are far more regular than Poisson). After 750 CPU steps with random filters: texture events in an OFF-ON pair 0.28 (truth 0.27, start 0.10); filter distance 0.072 → 0.053 (truth 0.040); spectrum distance 0.608 → 0.565 (truth 0.442). Spectrum only (2000 steps) gets its own distance to 0.53 but overshoots the pairs (0.38) and gets worse on the filters (0.075), so random filters are the default. Not there yet on this small test: the lines' direction shows in y-t, x-t is still scattered. Real training sees far more data.
- **From a v5-config checkpoint:** texture row dropped, wiggle row 0, painter new; teacher training with the pattern loss, resume, `generate.py`, and the v6 notebook (texture on and off).

Checks (CPU, simulated events; a soft edge moving 20 px per gap over a slowly varying background):
- **Old configs** (v5 texture, v4-like): same losses, gradients and sampled events as `v5-texture`, bit for bit.
- **Counts:** the sampler makes 33,081 ± 1,291 texture events (5 seeds) where the likelihood expects 33,023. With diagonal motion and a curved edge: 0.96–1.00× the expected count for `A` from 0.01 to 0.5 and grain 1–3 px (the speck sampler: 1.26×).
- **Likelihood vs sampler:** on events sampled with wiggle `A`, the likelihood over `A × (0.25, 0.5, 0.7, 1, 1.4, 2, 4)` is lowest at ×1.
- **Polarity on a curve:** on an OFF curve, 94% of the events near it are OFF with the wiggle, 74% with v5's texture.
- **Texture look:** 75% of the texture events have a same-polarity texture event one pixel further along the motion, one pixel of motion later (v5's texture: 34%). Texture events in an OFF-ON pair at the same pixel within 1.5 px of motion: 10% (specks: all). Neighbours across the motion firing at the same time agree in polarity 97% of the time (specks: 52%).
- **x-t / y-t loss:** its gradient reaches the wiggle size.
- **Init from a v5-config checkpoint:** path rows kept, texture row dropped, wiggle row 0. Teacher, student, `generate.py`, `check.py` on toy data.

Train: fine-tune from the v5 checkpoint (empty `train.out`; add your v5 run's size settings, e.g. `model.depth`, `model.z_dim`, if they differ from `config.yaml`). The optimizer and schedule start fresh, so use fewer steps and a lower lr than a run from scratch:

```bash
torchrun --nproc_per_node 4 train.py train.stage=teacher train.init=runs/teacher_v5/last.pt train.lr=1e-4 train.steps=60000 train.out=runs/teacher_v6p
```

The wiggle starts at 0.01 everywhere (v5's texture output means something else), so the event count first drops toward v4's and comes back as the wiggle grows. The validation line adds `wiggle` (mean size), `xt` and `stats` (pattern loss; lower = texture closer to real).

### `v6-detail` (detail image, separate branch)

Built on `v5-texture` (below). What the v4 / v5 checks showed, per dataset (HQ-EVFI / BS-ERGB / EDS):
- v4 makes 0.50× / 0.17× / 0.55× of GT's events. Most missing events are up-and-down pairs (HQ 98%), most likely small things passing that the frames don't show.
- v5's texture events have coin-flip polarity (50% polarity agreement, GT 79 / 84 / 91%) and break the one-polarity bursts (HQ 55 → 41%, EDS 57 → 38%): independent random events can't form curves.
- Blurred x-t / y-t agreement with GT (4 px, 4 ms): 0.83 / 0.69 / 0.82 for v4, against 0.97–1.00 for two samples of the same path. The tube is not the cause (without it: 0.83 / 0.64 / 0.79).

| Change | What it does |
|---|---|
| **Detail image** (`model.detail`) | A log-brightness image added to both keyframes before `render` moves them. Its mean per frame (`±2`, e.g. sharper edges) comes from the decoder like the rest of the path. Its random part `σ·n` lives at frame 0 and is moved to frame 1 by the path's own flow `F0(1)`, so a small dark thing makes OFF then ON at every pixel it crosses: connected curves in x-t / y-t, one polarity each. `σ` is a decoder output (0.01 at `z = 0`); `n` has the prior `N(0, 1)` per pixel. **Teacher:** a full-resolution branch (`detail_enc`, a small UNet on the event voxel + frames + flows) gives the posterior `N(m, s²)` of `n`; training uses a sample, validation the mean. **Generation / student:** `n ~ N(0, 1)`. The sampler, tube, likelihood and threshold fit are unchanged |
| **Detail price** (`loss.detail_kl`) | `KL(posterior ‖ prior)` per pixel, added to the likelihood (1 = the proper value, an ELBO). The teacher puts a detail only where the events pay for it, so the detail can't cheaply soak up path errors (what learned `k`, `ν` and v5's texture rate did). The KL also pins `σ` to the size of the details the teacher finds, which is what generation draws |
| **x-t / y-t loss** (`loss.xt`) | The ON / OFF events the path is expected to make (rise / `C_on`, fall / `C_off`, plus background) against the real events, both cut into x-t slices (one per row) and y-t slices (one per column), blurred by `loss.xt_blur` (2 px, 2 ms), compared slice by slice with the generalized KL of counts. It compares curves, not single events (which pixel fires when inside a curve is the sensor's randomness). Targets too few events along a curve, and curves a few ms or px off. 64 time bins (`losses.XT_BINS`) |
| **Texture off** (`model.texture: false`) | v5's random events are left out (the code stays) |
| **Init by name** (`train.py`) | Decoder outputs of the init checkpoint are matched by name (`model.head_rows`): from v4 or v5 the path outputs are kept, the texture row is dropped, the detail rows start at 0 |

Checks (CPU, toy data and simulated events):
- **Old configs** (no detail, `loss.xt` 0; v5 and v4-like): same losses and gradients as `v5-texture`, bit for bit, teacher and student stage.
- **Detail image off = no detail:** with `n = 0` and its outputs at 0, the path equals the path without it (difference 0); the random part at frame 1 is the frame-0 one moved by the path's flow.
- **Reads what the frames don't show:** flat gray frames, dark 2×2 specks present only in the events, moving 8 px per gap. On one batch, after 300 steps the teacher's detail correlates 0.98 with the true specks; `nll` −3.34 nats per pixel against 6.40 without the random part, for a KL of 1.07.
- **x-t / y-t loss:** 6e-6 (the background floor) when expected and real counts match; gradients agree with finite differences (median relative error 1e-7); an edge off by ±1 / 2 / 4 / 8 ms costs 0.15 / 0.58 / 2.3 / 8.5, against 0.01 when aligned.
- **Init from a v5-config checkpoint:** path rows kept, texture row dropped, detail rows 0.
- **Smoke:** teacher, student, `generate.py`, `check.py` on toy data.

Cost: the detail branch (UNet at full resolution, 32 / 64 / 128 channels); the x-t / y-t loss keeps two `2 × 64 × H × W` volumes per sample (recomputed in backward, a few GB at peak for batch 8 and 256² crops).

Train each run from scratch (`train.out` empty), one change at a time (on the `v6-detail` branch, whose config has `detail: true`, `xt: 0.1`):

```bash
torchrun --nproc_per_node 4 train.py train.stage=teacher model.detail=false loss.xt=0 train.out=runs/teacher_v6_base
torchrun --nproc_per_node 4 train.py train.stage=teacher model.detail=false train.out=runs/teacher_v6_xt
torchrun --nproc_per_node 4 train.py train.stage=teacher train.out=runs/teacher_v6
```

Before training, run `check_data.py` on BS-ERGB (time offset and pixel shift): a misaligned frame teaches the detail image the wrong edges.

### v5 (`v5-texture`)

Built on `v4-tube` (below). In v4 the background was clean, but dense moving regions had about half of the real events. On dense pixels v4 had the right net change (ON − OFF) and missed **up-and-down pairs**: HQ-EVFI 98%, BS-ERGB 79% and EDS 60% of the missing events. Those come from fine texture the frames don't show. The learned threshold was also off by 12–43% against the data, because it was soaking up path errors.

| Change | What it does |
|---|---|
| **Texture rate** (`model.texture`) | One more decoder output `a ≥ 0` per pixel (×1 = 0.01 at `z = 0`): hidden up-and-down change per unit of path change. It adds random events at `ρ_pol = a·|dL/dτ| / (2·C_pol)`, ON and OFF. They come only while the path moves (0 on a static background) and cancel in pairs, so the net change stays the path's job |
| **Pairs don't reset the pixel** | An up-and-back swing leaves the pixel's reference where it was, so texture and background events don't reset it; only real crossings do, and a pixel's events add up to its brightness change. The likelihood doesn't know which events were crossings: a forward pass over each pixel's events sums over it, looking back up to `loss.mix` events (16) for the last real crossing, or none since the gap started. Each pixel pays for the texture events it expects, `a · TV(path) · (1/C_on + 1/C_off)/2` |
| **Measured thresholds** (`model.measure_c`) | Over a gap a pixel's events add up to its change, `E[c_on·N_on − c_off·N_off | d] = d`, `d = L1 − L0`, whatever the path did (pairs cancel). The counts are the noisy side, so `(c_on, c_off)` is fitted as counts on change, with the rising and the falling part of `d` as the two equations. Pixels with `0.5 ≤ |d| ≤ 2` only. Kept as running sums per dataset (saved in the checkpoint); `log_c`, `log_r` are not trained |
| **`loss.smooth` 0.05** (was 0.01) | Also covers the tube factor and `log a`. Neighbors get the same path over the whole gap, so an edge reaches them at consistent times (continuous x-t / y-t), and texture forms solid areas |
| **Sampler warm start** (`Simulator.warm`) | Run the gap before and drop its events, so the pixels start the gap in a real state instead of the stationary guess (no burst in the first slice). The sampler's output also has `s` per event: 0 crossing, 1 background, 2 texture |

Checks (CPU, toy data and simulated events):
- **Forward pass:** equal to a brute-force sum over all 2^n crossing/other labelings of a pixel's events (to 1e-6 with a window of 16 for 7 events, and of 8 for 12 events).
- **Texture rate:** on events simulated with a known `a` (0.5, 1.0), the likelihood is lowest at the true `a`.
- **Events add up:** the simulated events add up to exactly 1.00× the frame change for any `a`. Pairs that reset the pixel would give 0.85× at `a` 0.5 and 0.74× at `a` 1.
- **Thresholds:** recovered from simulated events (0.250 / 0.354 vs 0.25 / 0.35, 60 samples). On the toy data, which has a real camera's dead zone (after rising, a pixel must fall a full threshold before an OFF event), the fit is about 10% high: a pixel that reverses direction loses up to one event.
- **Old configs** (no texture, no measured c): same loss and events as v4. Gradients agree to about 1e-8, because the backward pass adds terms in a different order.

Cost: the likelihood with texture took about 2× as long on CPU (toy batch). It grows with `loss.mix` and with the busiest pixel's event count.

### v4 (`v4-tube`)

Built on `v2-sensor-teacher` (not on `v3-vit-dpt`). Two changes, both for the salt-and-pepper noise in static regions, which was strongest at the end of the gap:

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
| Teacher | ConvNeXt encoder: time-binned real events + I0, I1 → `z` on the 8×8 patch grid. Training only |
| Student | Frozen video backbone on frames before I0 and after I1 (each side encoded alone) + conv head (time slots stacked on channels, ConvNeXt blocks, real frame times and gap length); v8.4: + I0, I1 at full resolution through a conv encoder, added on the code grid, 2 more blocks there → `ẑ`; no position code, any image size |
| Decoder | UNet on I0, I1, features **gated by `z`**: every output is multiplied by a function of `z`, so `z = 0` gives exactly the path with no motion (a cross-fade), `κ = 0.01` and `ν = model.nu`, and the decoder cannot ignore the code |
| Path | Warp both frames with learned time-polynomial flows (`K` terms per frame, no RAFT), blend with a time-varying visibility (sigmoid of a cubic in τ). `L(0)`, `L(1)` equal the frames by construction |
| Tube | `model.taut`: the least-event path within `ε` of `L` (free ends). `ε = 1.96/√2 · σ_L`: `σ_L` is the frame noise measured per dataset and brightness on pixels without events, so 95% of noise-only pixels fit a flat path. Crossings come from this path (`model.paths`) |
| Residual events | `model.residual_rate`: per pixel and polarity, `(κ_rise·rise + κ_fall·fall) / C_pol + ν_pol·dt` per unit τ, with rise / fall the slope of the **raw** path `L`. 6 decoder maps: `s["kappa"]` (ON / OFF × rise / fall) and `s["nu"]` (ON / OFF, events per second). They don't reset the pixel |
| Sensor | After each crossing the pixel sets `ref = L` and draws thresholds `X·C` with `X ~ IG(1, k)`. ON fires when the running max of `(L − ref)/C_on` reaches `X_on` (OFF likewise). Thresholds measured per dataset from frame change vs real event counts (`model.measure_c`); threshold spread per event fixed (`model.sigma_event`); no refractory time. Brightness per dataset: learned RGB weights, then `PNG^gamma`; under a Bayer filter (CED) each pixel uses its own channel |
| Events | `physics.Simulator` samples the same model: exact crossings of the taut path (grid of at least 128 steps, finer where the path is fast; linear pieces) with the random thresholds; slope events per polarity each time their running count passes a whole number; background as a Poisson process. Between gaps the pixel reference moves with the path's free start, so the jump fires nothing. Each event carries its source `s`: 0 crossing, 1 residual (slope), 2 background |

### Losses

| Loss | Stage | Meaning |
|---|---|---|
| `nll` | teacher, joint, interp | Likelihood of the real event **times**: crossing hazard of the taut path and residual rate at each event, survival of both thresholds between crossings, after the last event and in silent pixels; the first wait per pixel starts from the stationary state. Which events were crossings is summed over by a forward pass per pixel (`loss.mix`). Every pixel pays for the residual events it expects |
| `xt` | teacher, joint | Expected vs real ON / OFF events (crossings + residual) on blurred x-t and y-t slices (generalized KL of counts) |
| `mask` | teacher, joint | v8.2: second teacher pass on part of the batch with 2–3 of 8 event slices hidden; the x-t loss on the hidden slices only |
| `emd` | teacher, joint | v8.3: earth mover's distance between expected and real events along t, x, y, per polarity (shapes, per real event) |
| `pmse` | teacher, joint | v8.3: PMSE of a sampled stream as its expected value (block error + background variance), relative |
| `sigreg` | teacher, joint | LeJEPA SIGReg on `z`: keeps the code isotropic Gaussian (no collapse, easy to predict later) |
| `jepa` | student, joint | Student code vs teacher code (`stopgrad` in the student stage) |
| `nll_student` | student, joint | `nll` with the student's code |
| `emd_student`, `pmse_student` | student, joint | v8.3: `emd`, `pmse` with the student's code (same weights) |
| `xt_student` | student, joint | v8.4: `xt` with the student's code (same weight) |
| `photo` | interp | `L(τ)` must match the real hidden frames |

---

## 2. Files

| File | Content |
|---|---|
| `model.py` | Backbones, teacher, student, gated decoder, flows, `render(s, τ) → L, Y`, tube, residual rate |
| `losses.py` | Likelihood, x-t / y-t loss, photometric, SIGReg, stage logic, validation diagnostics |
| `physics.py` | Quadratic root solver and event `Simulator` |
| `data.py` | Sequence reader, training samples, collate, joint sampling over the datasets in `data.sets` |
| `train.py` | Stages `probe / teacher / student / joint / interp`, single GPU or `torchrun` DDP, bf16, resume |
| `generate.py` | RGB video → events `.npz` (student path) |
| `eval_interp.py` | Frame interpolation scores (PSNR, SSIM, LPIPS) of the teacher's path at the hidden frames |
| `check.py` | Per dataset and split: diagnostics, code / path size, event-frame time alignment |
| `HQ-EVFI/hqevfi.py`, `BS-ERGB/bsergb.py`, `EDS/eds.py`, `CED/ced.py` | One converter per dataset: the download (or its extracted folder) → the training layout |
| `check_data.py` | Checks every converted sequence, and the event / frame alignment per dataset (no model needed) |
| `toy_data.py` | Synthetic dataset for smoke tests |
| `config.yaml` | All settings |

### Code used as-is from other projects

| Part | Source | How |
|---|---|---|
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
- v8.2: `skip: [lo, hi]` draws one skip per batch from lo..hi (defaults: HQ-EVFI 0–4, EDS 0–2), so training also sees the 1-frame gaps of generation; validation runs at both ends.
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

# 2. student (teacher frozen; v8.4: the decoder trains with the student, so start from a student checkpoint if you have one)
python train.py train.stage=student train.init=runs/teacher/last.pt train.out=runs/student

# 3. optional joint fine-tune (small lr)
python train.py train.stage=joint train.init=runs/student/last.pt train.lr=2e-5 train.out=runs/joint

# 4. interp: teacher only, nll + photo; datasets need hidden frames (skip > 0)
python train.py train.stage=interp train.init=runs/student/last.pt data.sets.bsergb.skip=1 train.lr=5e-5 train.out=runs/interp
```

`train.init` loads every part whose size still fits and lists the parts it skipped.

### What to watch (validation line)

| Value | Meaning | Healthy |
|---|---|---|
| `nll` | Teacher code | Lowest |
| `nll_shuffled` | Codes swapped between samples | Clearly above `nll`, else `z` carries nothing |
| `nll_zero` | Path with no motion (`z = 0`, a cross-fade) | Far above `nll` |
| `nll_student` | Student code | Between `nll` and `nll_zero` |
| `gap` | `nll_student − nll`: in-between information the video cannot predict | Small; report it vs gap length |
| `xt` | The x-t / y-t loss on the teacher's path | Falls with training |
| `res_share` | Share of the expected events that are residual events (slope part + background) | Well below 1. Above ~0.8 the crossings matter little (v5.2's texture made 63–81% of the generated events) |
| `emd`, `emd_student` | v8.3: earth mover's distance per real event (ms along t + px along x and y), teacher's / student's code | Falls with training; the student's above the teacher's |
| `xt_student` | v8.4: x-t loss with the student's code | Falls in stage 2. The teacher's numbers rise there, since the decoder moves to the student |
| `pmse`, `pmse_student` | v8.3: expected PMSE of a sample relative to the real blocks' mean square | Falls with training; well below 1 |
| `photo` | The teacher's path vs the hidden frames (datasets with `skip` > 0) | Falls in the interp stage |
| `mask` | v8.2: masked-slice score, the hidden event slices guessed through the decoder | Falls with training |

The run resumes from `train.out/last.pt`. Checkpoints leave out the frozen backbone weights.

### Smoke test (CPU, a few minutes)

```bash
python toy_data.py data/toy 24
S="data.sets={\"toy\":{\"root\":\"data/toy\",\"skip\":3}} data.crop=64 data.min_events=50 data.workers=2 data.bins=8 \
   model.backbone=none model.width=16 model.depth=[1,1,1] model.enc_depth=1 model.pred_dim=64 model.pred_depth=2 \
   loss.grid=16 train.batch=4 train.lr=1e-3 train.warmup=20 train.log_every=100 train.val_batches=8"
python train.py $S train.stage=teacher train.steps=800 train.save_every=400 train.out=runs/teacher
python train.py $S train.stage=student train.init=runs/teacher/last.pt train.steps=400 train.save_every=200 train.out=runs/student
python train.py $S train.stage=interp train.init=runs/student/last.pt train.lr=3e-4 train.steps=300 train.save_every=150 train.out=runs/interp
python generate.py --ckpt runs/student/last.pt --seq data/toy/val/seq100 --out runs/student/events.npz
```

---

## 6. Generate

```bash
python generate.py --ckpt runs/student/last.pt --seq my_video/ --out events.npz --dataset bsergb   # whose sensor / brightness mapping
python generate.py --ckpt ... --seq ... --set shot_noise_rate_hz=0 pos_thres=x1.2                     # no background, larger ON threshold
```

`my_video/` needs only `frames/` and `frame_ts.npy`. Output: `t` (s, float64), `x`, `y` (int16), `p` (int8 ±1), sorted by time.
`shot_noise_rate_hz` scales the decoder's background map (the learned value is `model.nu`; 0 removes it). `sensor.residual_events: false` leaves out residual and background events.

### Event-guided frame interpolation

The teacher's path is also a reconstructor: from I0, I1 and the real events it gives the frames at any time in the gap. `eval_interp.py` renders it in RGB at the hidden frame times (warping and blending the RGB keyframes exactly as `render()` does, in linear light) and scores it against the real hidden frames (PSNR, SSIM, and LPIPS if `pip install lpips`), next to a plain blend, the frames-only path (`z = 0`: in v8 a cross-fade in linear light) and optionally the student (`--student`). It also saves one example with error maps. Run it on the interp-stage checkpoint.

```bash
python eval_interp.py --ckpt runs/interp/last.pt --split test --n 200
python eval_interp.py --ckpt ... --skip bsergb=3 --crop 4096      # large crop = full frames
```

---

## 7. Main knobs

| Key | Effect |
|---|---|
| `data.sets` | Datasets trained together: `root`, `skip` (hidden frames in the gap; `(skip+1)/fps` close to the frame interval of the videos you will convert; v8.2: `[lo, hi]` = one per batch), `weight` (share of batches), `gamma` (PNG → linear light, from `check_data.py`) |
| `data.context` | Frames per side for the student, taken at the gap's own stride, so training sees the same frame spacing as generation (even for V-JEPA 2.1) |
| `data.bins` | Time bins of the teacher's event input (32, ~1.1 ms each) |
| `model.backbone` | `vjepa2_1`, `levjepa`, or `none` (no world knowledge) |
| `model.patch`, `model.z_dim`, `model.enc_depth` | Path code cell (8 px), its size (64), teacher blocks per level (6) |
| `model.depth`, `model.width` | Decoder UNet blocks per level ([4, 4, 12, 8]) and first-level width (64) |
| `model.pred_dim`, `model.pred_depth` | Student head width (512) and conv blocks (8) |
| `model.lora_rank`, `model.lora_blocks`, `train.lora_lr` | LoRA on the video backbone's attention (V-JEPA 2.1 or LeVJEPA; 0 = off; e.g. 16), on the last N blocks (8 of 24), its learning rate × `train.lr` (5) |
| `model.K`, `model.flow_scale` | Flow terms per frame (2) and the bound of each term (64 px) |
| `model.tube` | `measured` (width from the frame noise), `scalar` (× one learned factor per dataset), `false` (no tube) |
| `model.nu` | Background events / px / s at `z = 0`; the decoder scales it per pixel and polarity |
| `model.sigma_event` | Threshold spread per event, fixed (fraction of c); `null` = learned |
| `model.measure_c` | Thresholds measured from frame change vs real ON/OFF counts per dataset (`c_init` until the first batch of a dataset) |
| `loss.xt`, `loss.xt_blur` | Weight of the x-t / y-t loss (2.0; 0 = off) and its blur (2 px, 2 ms) |
| `loss.mask`, `loss.mask_share`, `loss.mask_slices` | v8.2 masked slices: weight (1.0; 0 = off), share of each batch (0.25), hidden slices per sample (2–3 of 8) |
| `loss.photo` | Weight of the photometric loss in the interp stage (10) |
| `loss.mix` | How far back, in events, the likelihood looks for a pixel's last crossing (16; cost grows with it) |
| `loss.grid` | τ steps of the likelihood path (128) |
| `sensor.*` | Generation only: profile and overrides (`set`), `max_steps`, `residual_events: false` for crossings only |

---

## 8. Status

Verified here (CPU, toy data), `v8`:
- Sampler and likelihood are the same model on a moving path; the nll is lowest at the true threshold, `κ` and `ν`.
- All stages, all `model.tube` options; interp changes only the teacher; toy training close to v7.

Verified here (CPU, toy data), `v7`:
- Sampler and likelihood are the same model: residual event counts match (1,308 vs 1,314 per draw), and the nll is lowest at the true `κ` and `ν`.
- The residual rate reads the raw path: its events and gradient reach pixels whose taut path is flat.
- Teacher, student, joint and interp stages with every `model.tube` option; the interp stage changes only the teacher.
- Toy training: `nll` < `nll_student` < `nll_shuffled` < `nll_zero`; `generate.py`, `check.py`, `eval_interp.py`.

Earlier versions (still true for the parts v7 kept):
- Inverse-Gaussian terms match numerical integration.
- Taut path: values equal, and gradients agree to 1e-14, with a plain autograd version; `gradcheck` passes; the path stays in the tube.
- V-JEPA 2.1 wrapper (official code, random weights), 2-process DDP.

Not tested here: real data on GPU, pretrained weight downloads, the LeVJEPA wrapper (Hugging Face is blocked in this sandbox).

Known limits:
- The sensor model ignores low-light bandwidth; the likelihood reads `L` on a τ grid (`loss.grid`) with linear interpolation.
- The noise table uses pixels without events; tiny real changes below the threshold are counted as noise, so `σ_L` is slightly too large. In v7 nothing learned narrows the tube (unless `model.tube: scalar`); the residual rate makes the events of changes the tube flattens.
- The likelihood scores slope events as a Poisson process, the sampler fires them on schedule (v8.3): same expected count, more regular timing than the likelihood assumes. The background is Poisson in both.
