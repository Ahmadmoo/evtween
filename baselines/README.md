# Event-generation baselines for evtween v7

Generate one raw event stream per method from the same observed RGB frames. The runner reads `frames/*.png` in filename order and `frame_ts.npy` in seconds, as in [evtween v7](https://github.com/Ahmadmoo/evtween/tree/v7). Use one directory per recording, zero-padded frame names, 8-bit PNGs and a single, strictly increasing frame clock.

## Run

Use a Python environment with Git, Tk and a matching PyTorch/torchvision pair. The CPU run was tested with Python 3.12, torch 2.6.0, torchvision 0.21.0 and NumPy 1.26.4. In a new CPU environment the first command installs that pair. On CUDA, use the torch and torchvision builds for your GPU.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python fetch.py
python run.py all /data/hqevfi/test/scene01 --view center346 --stride 5 --out predictions/hqevfi
```

`fetch.py` downloads the six pinned repositories plus the V2CE and SuperSloMo weights (354 MiB combined) and checks their SHA-256 hashes. You can fetch or run one method by name, for example `python fetch.py fracevent` and `python run.py fracevent /data/seq --view native`. Device selection is automatic. IEBCS and DVS-Voltmeter run on CPU.

The six-way run needs **17 observed frames after striding** and a 346 × 260 input. If the recording already has that resolution, use `--view native`. For a larger image, `--view center346` takes a single centered 346 × 260 crop in source pixels and supplies it to every method. A smaller source image needs a separate comparison. The other five methods can run individually on the native grid.

To run the evtween **student** on exactly the same inputs, export the selection and point v7's generator to it. Use the trained checkpoint for the named dataset. The example paths follow the first command above.

```bash
python run.py prepare /data/hqevfi/test/scene01 --view center346 --stride 5 --out predictions/hqevfi
python /path/to/evtween/generate.py --ckpt /path/to/student/last.pt --dataset hqevfi \
  --seq predictions/hqevfi/scene01/center346-s5/rgb \
  --out predictions/hqevfi/scene01/center346-s5-seed1/evtween.npz
```

## Inputs that change the comparison

| Dataset | v7 observed-frame stride | EVIS-core PNG exponent |
| --- | ---: | ---: |
| HQ-EVFI | 5 | 1.0 |
| BS-ERGB | 1 | 2.2 |
| EDS | 3 | 1.0 |

The v7 setting is `skip + 1`. The runner defaults to stride 1 and exponent 1. Pass `--stride 3` for EDS and `--evis-gamma 2.2` for BS-ERGB. Frame times come from the converted sequence. Each method receives indices `0, stride, 2*stride, ...`. The final partial stride is excluded. Keep any hidden high-speed frames and recorded events out of the inputs. The student and V2CE can use future observed frames, so this is an offline comparison. The evtween teacher uses recorded events.

**Spatial support.** Crop the recorded events with the saved `roi_xywh = [x0, y0, w, h]`. Keep `x0 <= x < x0+w` and `y0 <= y < y0+h`, then subtract `x0,y0`. The predictions already use these local coordinates. A center crop removes peripheral motion. Report its size and keep native-resolution results in a separate table. Use the same preconverted recording for every method. In particular, evtween's EDS converter can resample RGB onto the event grid, crop the overlap and adjust the frame timestamps. Record the conversion settings once for the dataset.

**Temporal support.** Score from the first to the last selected timestamp. Use one bin-boundary convention for both predictions and ground truth. EVIS-core timestamps land on observed frames. With v7's `[t_i, t_i+1)` bins, spikes emitted at the right frame enter the following bin and those on the final boundary fall outside the last bin. Moving them for only one model changes the timing score. If a dataset has recording gaps, apply the same valid-time mask to every method.

The v7 pair loader defaults to a 256 × 256 crop and skips intervals with zero recorded events. Comparing its numbers directly with full recordings or the 346 × 260 selection mixes spatial and temporal supports. Preserve genuinely quiet intervals when scoring false events.

**Sensor settings.** The runs below use released presets with different thresholds, noise and brightness mappings. These affect counts, polarity balance and timing. A dataset-calibrated evtween student and baselines on factory presets have different calibration budgets. For calibrated results, choose the sensor settings on independent calibration or validation recordings, freeze them and report the seed(s). Keep test recordings out of profile and checkpoint selection. v7 can fall back to the test folder when a dataset has no validation folder.

## What each method runs

| Method | Implementation and evaluation detail | Authors' code |
| --- | --- | --- |
| EVIS-core (RGB only) | Clean `BatchedMultiCamProcessor`, threshold 0.15. Each observed frame can emit at most one event per pixel. Full EVIS uses renderer motion vectors for high-rate warping. The RGB variant has no access to those buffers. | [EVIS](https://github.com/spikelab-jhu/isaac-sim-event-camera-plugin/tree/d6fe47923fad09c4bfa2fd2cb13f82520114cfae) |
| FracEvent | Stateful frame-pair solver using `davis346.yaml`. ON/OFF thresholds 0.45/0.50, fractional order 0.72, noise disabled. | [FracEvent](https://github.com/bonaparte233/FracEvent/tree/1fe6ee7e8e7b4cbc3d2526aa80c033570bb2637f) |
| IEBCS | Python `DvsSensor` with the example's 0.4 thresholds, measured 161-lux noise histograms, and L* mapped to 0–10,000. Uses actual elapsed microseconds from the first frame. No arbiter or discarded warm-up. An entirely black update after initialization raises an error in the adapter because upstream stops its clock. Portrait inputs are transposed internally for the upstream noise-table index and returned to the source axes. | [IEBCS](https://github.com/neuromorphicsystems/IEBCS/tree/8fdab5c3d078435d091fa8a94ad657e1fedf0c59) |
| v2e | SuperSloMo plus `EventEmulator`. Automatic motion-based interpolation, thresholds 0.2/0.2, spread 0.03, cutoff 300 Hz, refractory period 0.5 ms. The runner edge-pads right and bottom to a multiple of 32 and removes the padding after interpolation. It keeps observed endpoints and assigns each synthesized frame its actual midpoint time in that gap. Both choices differ from the stock video CLI and can change events at borders and in time. | [v2e](https://github.com/SensorsINI/v2e/tree/23f6d5b18aedee5763a1cae7adf43112cf686e20) |
| V2CE | Released V2CE-3D weights and LDATI slope sampler. A window covers 17 observed frames, and each gap is exported once. LDATI uses the duration of each actual gap. The authors trained on MVSEC at 346 × 260. Check sequence overlap if evaluating on MVSEC. The toolbox typically resizes to height 260 before cropping or tiling the width. | [V2CE](https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox/tree/67f1624f6539eec7ee7df2a73f95198b3d546969) |
| DVS-Voltmeter | Released `EventSim`, DVS346 parameters, raw observed frames and integer-microsecond clock. The authors also use high-rate or interpolated video. A high-rate run has a different input condition and should be labeled separately. The upstream signed 32-bit clock limits a continuous run to 35.8 minutes. | [DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter/tree/9d86353ecfaaa1af82b9eb66b162f0bf80cd4861) |

For v2e, `--v2e-steps 0` uses automatic interpolation. `--v2e-steps N` with `N >= 2` generates a fixed number of intermediate samples per gap. `--v2e-steps 1` runs the sensor alone and labels the output accordingly. Motion-based interpolation can undersample the 300 Hz photoreceptor on slow or static content. If upstream prints an IIR accuracy warning, record it and test a fixed subdivision using validation data. Keep the device fixed across runs because upstream SuperSloMo uses different input normalization on CPU and GPU.

## Output

Each prediction is saved under `<out>/<sequence>/<view>-s<stride>-seed<seed>/`. Use a separate `--out` per dataset when sequence names overlap. Rerunning the same configuration replaces that prediction. The NPZ opens with `np.load(path, allow_pickle=False)`.

| Fields | Contents |
| --- | --- |
| `t, x, y, p` | Time-sorted events. Float64 seconds on the input clock, int16 zero-based column and row, int8 polarity (+1 ON, −1 OFF). |
| `frame_indices, frame_ts` | Original indices and timestamps of the observed frames. |
| `source_wh, roi_xywh, width, height` | Original dimensions, source crop and output dimensions in pixels. |
| `metadata` | JSON with the method, source commit, checkpoint hash where applicable, device, seed and number of events outside the selected time window. Those out-of-window events are excluded. |

All six ran on CPU on the same moving test clip with irregular frame intervals. The common crop was verified pixel for pixel. On the DVS-Voltmeter authors' `office_zigzag` sample, all 419 events matched a direct seeded `EventSim` call using the same relative clock.
