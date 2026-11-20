# Event-generation baselines for evtween

Each baseline turns the same RGB video into one raw event stream, at the video's full resolution, over the full sequence. Every method runs the pipeline its authors ship, with their default settings: nothing added, nothing removed. The runner reads `frames/*.png` in filename order and `frame_ts.npy` in seconds, as in evtween (zero-padded names, 8-bit PNGs, one strictly increasing clock).

## Methods

| Method | Authors' pipeline we run | Input | Defaults | Code |
| --- | --- | --- | --- | --- |
| ESIM (vid2e) | FILM upsampling (bisect until the largest flow is ≤ 1 px), then `esim_torch` | Video | C+ = C− = 0.2, no refractory, `log(gray/255 + 1e-5)`, no noise | [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e/tree/ecbb11a9345bb9d31b4b691e7d82965da4401345) |
| v2e | SuperSloMo, automatic upsampling (≤ 1 px per step), batch 8, then `EventEmulator` | Video | C = 0.2/0.2, σ 0.03, cutoff 300 Hz, leak 0.01 Hz, shot noise 0.001 Hz, refractory 0.5 ms | [v2e](https://github.com/SensorsINI/v2e/tree/23f6d5b18aedee5763a1cae7adf43112cf686e20) |
| DVS-Voltmeter | `EventSim`, DVS346 parameters | Video (the authors ask for high-frame-rate video) | Released config, noise on | [DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter/tree/9d86353ecfaaa1af82b9eb66b162f0bf80cd4861) |
| IEBCS | `DvsSensor` as in the authors' example | Video | C = 0.4, 100 µs latency, 161-lux measured noise, L* mapped to 0–10,000 | [IEBCS](https://github.com/neuromorphicsystems/IEBCS/tree/8fdab5c3d078435d091fa8a94ad657e1fedf0c59) |
| FracEvent | Stateful frame-pair solver | Video | `davis346.yaml`: C = 0.45/0.50, α 0.72, noise off | [FracEvent](https://github.com/bonaparte233/FracEvent/tree/1fe6ee7e8e7b4cbc3d2526aa80c033570bb2637f) |
| V2CE | V2CE-3D, LDATI slope sampler | Video | Released MVSEC weights | [V2CE](https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox/tree/67f1624f6539eec7ee7df2a73f95198b3d546969) |

## Where we differ from the released scripts

These changes fix the scripts' handling of time, color or size. None of them changes a method's event model.

| Method | Change | Why |
| --- | --- | --- |
| all | Real frame times from `frame_ts.npy` | ESIM's and v2e's scripts assume a constant fps. The datasets' frame clocks are not regular. |
| ESIM | A largest FILM flow below 1 px gives one midpoint | The released recursion never ends there. Above 1 px it is unchanged. |
| ESIM | Gray from RGB with `RGB2GRAY` | The released upsampler applies `BGR2GRAY` to RGB frames, which swaps the red and blue weights. |
| v2e | A synthesized frame gets the time it was made for, `(k + 0.5)/n` of its frame interval | The released script labels it `k/n`, which shifts every event by half a step. |
| V2CE | Runs at the input resolution | The released script resizes to height 260. The network is fully convolutional, so it runs on any size. |
| V2CE | A video with fewer than 16 frame pairs repeats its last frame to fill the window, and the extra pairs are dropped | The network takes windows of exactly 16 pairs. |

## Setup

Two environments share this folder. The main one (PyTorch + CUDA) runs every method. ESIM's FILM step runs in a TensorFlow 2 environment, as in vid2e: use Python 3.9, `tensorflow==2.6.2`, `opencv-python==4.5.5.64` and `numpy==1.19.5`, the versions vid2e pins.

```bash
# main environment
python -m pip install torch torchvision          # CUDA build matching your driver
python -m pip install -r requirements.txt
python fetch.py                                 # pinned repos, weights, builds esim_torch (needs nvcc)
```

`fetch.py` checks the SHA-256 of the SuperSloMo and V2CE weights. The FILM checkpoint (`pretrained_models.zip` from the vid2e authors) is not pinned yet. `fetch.py` prints its hash after download: put it in `WEIGHTS`.

## Run

```bash
# 1. TensorFlow environment: ESIM's FILM frames
python run.py film /data/hqevfi/test/scene01 --out predictions/hqevfi
# 2. main environment: all six methods
python run.py all /data/hqevfi/test/scene01 --out predictions/hqevfi
# one method
python run.py fracevent /data/hqevfi/test/scene01 --out predictions/hqevfi
```

`--stride` (default 1, the full video) selects every n-th frame as the input video. It must be the same for every method, ESIM's `film` step and evtween. evtween reads the sequence folder directly at stride 1. For a larger stride, `python run.py prepare <seq> --stride n --out <out>` writes the selected frames to `<out>/<seq>/s<n>/rgb` for evtween's `generate.py --seq`.

`--seed` (default 0) seeds Python, NumPy and PyTorch before each method. With seed 0, v2e uses this global seed rather than its own. ESIM and FracEvent (noise off) are deterministic. IEBCS and DVS-Voltmeter run on CPU, as released. The other methods use the GPU when `--device auto` finds one. ESIM needs CUDA. Keep one device for all runs, because v2e's SuperSloMo normalizes its input differently on CPU and GPU.

## Output

`<out>/<sequence>/s<stride>-seed<seed>/<method>.npz`, opened with `np.load(path, allow_pickle=False)`. Rerunning the same configuration replaces it. ESIM's FILM frames are kept in `<out>/<sequence>/s<stride>/film/` and are large: delete them after the ESIM run.

| Fields | Contents |
| --- | --- |
| `t, x, y, p` | Time-sorted events. Float64 seconds on the input clock, int16 column and row, int8 polarity (+1 ON, −1 OFF) |
| `frame_indices, frame_ts` | Indices and timestamps of the input frames |
| `width, height` | Pixels (the input resolution) |
| `metadata` | JSON: method, source commit, weights hash, device, seed, and the number of events outside `[first, last]` frame time (excluded) |

## Evaluation: Chamfer

`chamfer.py` follows `chamfer_full` in `benchmarks/METRICS.md` (branch `event-evaluation-v52-v53`). It scores every event of every method in `<dataset>/<sequence>/<run>/` against the recorded events (`ev_t/x/y/p.npy` of the sequence named in the baselines' metadata), over `[first, last]` input frame time:

- points `(x, y, v·t_ms)`, with `v` in pixels/ms (default 1)
- per polarity: `C_s = ½ [mean NN distance P→G + mean NN distance G→P]`, exact (SciPy KD-tree, no sampling)
- `chamfer` = mean over the polarities present in both streams. With no shared polarity, it is undefined.

Each dataset is scored and summarized on its own. The summary averages over the sequences that have a score for every method in that dataset, so all methods share one sequence set.

```bash
# one sequence (e.g. one Slurm array task); scores go to <seq>/<run>/chamfer.json and are not recomputed
python chamfer.py $PRED/hqevfi --seq 2024-01-05-15-04_1
# after all tasks: per-dataset tables
python chamfer.py $PRED/hqevfi $PRED/bsergb $PRED/eds --summary
# or both steps for a whole dataset at once
python chamfer.py $PRED/eds
```

The outputs per dataset are `chamfer_<run>.csv` (one row per sequence and method) and `chamfer_<run>_summary.csv` (mean and std over sequences). `--v 0.1 1 10` scores several time scales in one pass. To score evtween next to the baselines, put its `generate.py` output into the same folder as `evtween.npz`.
