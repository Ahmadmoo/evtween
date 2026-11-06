"""Deterministic benchmark preparation using the unchanged Evtween Sequence reader."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def discover(root):
    return sorted(p.parent for p in Path(root).expanduser().rglob('frame_ts.npy'))

def prepare(sequence, run_dir, start=0, count=21, stride=1, crop_size=(260, 346)):
    spec = importlib.util.spec_from_file_location('evtween_benchmark_data', ROOT / 'data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seq = module.Sequence(str(Path(sequence).expanduser().resolve()))
    if seq.cfa:
        raise ValueError('Bayer/CED sequences need a sensor-specific protocol; do not compare silently as grayscale')
    if len(seq.files) != len(seq.ts) or not np.all(np.isfinite(seq.ts)) or np.any(np.diff(seq.ts) <= 0):
        raise ValueError('Frame count/timestamps are invalid')
    if start < 0 or count < 2 or stride < 1:
        raise ValueError('Require start >= 0, count >= 2, stride >= 1')
    indices = np.arange(start, start + count * stride, stride)
    if indices[-1] >= len(seq):
        raise ValueError('Requested frame range exceeds sequence')
    h, w = crop_size
    _, H, W = seq.frame(int(indices[0])).shape
    if H < h or W < w:
        raise ValueError('Crop exceeds source resolution; select a smaller crop or another sequence')
    y0, x0 = (H-h)//2, (W-w)//2
    run = Path(run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    frames = run / 'input' / 'frames'
    frames.mkdir(parents=True)
    for n, i in enumerate(indices):
        image = seq.frame(int(i))
        if tuple(image.shape[1:]) != (H, W):
            raise ValueError('Frame resolution changed inside sequence')
        arr = image[:, y0:y0+h, x0:x0+w].permute(1, 2, 0).numpy()
        Image.fromarray(np.rint(arr*255).astype('uint8')).save(frames / f'{n:06d}.png')
    ts = seq.ts[indices] - seq.ts[indices[0]]
    np.save(run / 'input' / 'frame_ts.npy', ts)
    if seq.ev is not None:
        ev = seq.events(seq.ts[indices[0]], seq.ts[indices[-1]])
        m = (ev['x'] >= x0) & (ev['x'] < x0+w) & (ev['y'] >= y0) & (ev['y'] < y0+h)
        save_events(run / 'ground_truth.npz', ev['t'][m]-seq.ts[indices[0]], ev['x'][m]-x0, ev['y'][m]-y0, ev['p'][m], ts[-1], w, h)
    meta = dict(source=str(sequence), indices=indices.tolist(), absolute_t0=float(seq.ts[indices[0]]),
                duration=float(ts[-1]), height=h, width=w, crop=[x0,y0,w,h],
                interpolation='none', input_protocol='shared selected frames; no hidden intermediate frames',
                uniform_timestamps=bool(np.allclose(np.diff(ts), np.diff(ts).mean(), rtol=1e-3, atol=1e-7)))
    (run / 'input.json').write_text(json.dumps(meta, indent=2))
    return run

def save_events(path, t, x, y, p, duration, width, height):
    t, x, y, p = map(np.asarray, (t,x,y,p))
    if not (len(t)==len(x)==len(y)==len(p)) or not np.all(np.isfinite(t)):
        raise ValueError('Malformed events')
    if not np.all(np.isin(p, [-1,0,1])):
        raise ValueError('Unknown polarity encoding')
    m = (t>=0) & (t<duration) & (x>=0) & (x<width) & (y>=0) & (y<height)
    order = np.argsort(t[m], kind='stable')
    np.savez_compressed(path, t=t[m][order].astype('float64'), x=x[m][order].astype('int32'),
                        y=y[m][order].astype('int32'), p=np.where(p[m][order]>0,1,-1).astype('int8'))

