# Reconstruction training (FracEvent-style: train on simulated events, test on real)

One E2VID-style network is trained per simulator on events that simulator generated from the **same GoPro
frames**; every network is then tested on **real** events (DAVIS240C, HQF). Only the event source changes.

## Phase 1 decisions
| Item | Choice |
|---|---|
| Training frames | GoPro `GOPRO_Large_all` train split: 22 sequences, 22,653 sharp frames, 1280x720, 240 fps |
| Frames given to simulators | 240 fps, centre crop to 4:3 (960x720), 4x4 pixel average to 240x180, grayscale (same for every simulator) |
| Simulators | chosen per run (`--methods`); first: v2e, dvs_voltmeter, evtween_v8.2_student; later: ESIM, evis_core, iebcs, fracevent |
| Loss | LPIPS + temporal consistency (flow between target frames from RAFT on the GoPro frames) |
| Training windows | option: fixed spacing (N GoPro frames between targets, N = 1 is every 240 fps frame) or random spacing in a range (suggested 6-14 frames = 25-58 ms, centred on DAVIS240C's 24 fps) |
| Seeds | 3 per simulator |
| Evtween | used as is (no calibration to DAVIS240C), stated in the paper |

## Phase 1a: data and events
```bash
# 1. GoPro frames (done once; 35.3 GB download, train split extracted)
python benchmarks/recon_train/download_gopro.py --dest /u/marmary/airlabv2e/evtween/data

# 2. Generator layout: 240x180 gray, 240 fps -> $EVTWEEN_DATA/gopro/train/<sequence>/
python benchmarks/recon_train/prepare_gopro.py --src /u/marmary/airlabv2e/evtween/data

# 3. Events for the chosen simulators -> data_test_evt/gopro/train/<method>/<sequence>/events.npz
python benchmarks/event_generation/generate_event_streams_v3_fast.py gopro train \
    --methods v2e dvs_voltmeter evtween_v8.2_student --gpu-jobs 4

# 4. Gate check: event rates, events per training window, sample images
python benchmarks/recon_train/check_training_events.py gopro --methods v2e dvs_voltmeter evtween_v8.2_student
```
All steps can be rerun after an interruption; finished sequences are skipped. More simulators can be added
later by rerunning steps 3 and 4 with other `--methods`.

## Phase 1b: training (next)
Pilot on one simulator, then all chosen simulators x 3 seeds; resumable from the last checkpoint.
