"""V2CE inference with explicit per-pair physical-time decoding.

Uses the installed upstream network and LDATI, without editing that checkout.
The network is not conditioned on dt; this is an adapter variant, not a claim
that the pretrained model was trained for arbitrary frame cadence.
"""
import importlib.util
import json
import logging
from pathlib import Path
import random
import sys

import numpy as np

from benchmark import save_events
from hashlib import sha256 as _sha256


def sha256(path):
    digest = _sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_times(timestamp_us, start, end, sampler_fps=30.):
    """Map LDATI's within-pair nominal clock to the actual pair interval."""
    relative = np.asarray(timestamp_us, dtype=np.float64) * 1e-6 * sampler_fps
    if end <= start or not np.isfinite(relative).all():
        raise ValueError('Invalid interval or sampler timestamps')
    # LDATI emits integer microseconds, so permit at most one nominal tick.
    tolerance = sampler_fps * 1e-6
    if np.any(relative < -tolerance) or np.any(relative > 1+tolerance):
        raise ValueError('LDATI output lies outside its input frame pair')
    # Endpoint events belong to the next half-open interval; never extrapolate.
    keep = (relative >= 0) & (relative < 1)
    return start + relative[keep] * (end-start), keep


def load_generator(assets):
    import torch
    from rng_snapshot import rng_reset
    source = assets / 'third_party/v2ce'
    weights = assets / 'weights/v2ce_3d.pt'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_v2ce_upstream', source / 'v2ce.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.logger = logging.getLogger('V2CE')
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    model = module.get_trained_mode(str(weights))
    reset = rng_reset(torch)
    hashes = dict(upstream_sha256=sha256(source / 'v2ce.py'),
                  sampler_sha256=sha256(source / 'scripts/LDATI.py'),
                  weights_sha256=sha256(weights))

    def generate(run):
        reset()
        out = run / 'v2ce'
        meta = json.loads((run / 'input.json').read_text())
        ts = np.load(run / 'input/frame_ts.npy')
        paths = sorted((run / 'input/frames').glob('*.png'))
        if len(paths) != len(ts) or len(ts) < 17 or np.any(np.diff(ts) <= 0):
            raise ValueError('V2CE requires at least 17 frames and increasing timestamps')
        if (meta['height'], meta['width']) != (260, 346):
            raise ValueError('V2CE requires the shared 260x346 crop')
        if not torch.cuda.is_available():
            raise RuntimeError('The installed V2CE network requires CUDA')
        voxels = module.video_to_voxels(model, image_paths=list(map(str, paths)), batch_size=1)
        expected = (len(ts)-1, 2, 10, 260, 346)
        if voxels.shape != expected or not np.isfinite(voxels).all():
            raise ValueError(f'Unexpected V2CE output: {voxels.shape}; expected {expected}')
        chunks, discarded = [], 0
        for i in range(len(ts)-1):
            # Decode separately to retain pair identity, including boundary timestamps.
            ev = module.sample_voxel_statistical(
                torch.as_tensor(voxels[i:i+1], device='cuda'), fps=30.,
                bidirectional=False, additional_events_strategy='slope')[0]
            t, keep = decode_times(ev['timestamp'], ts[i], ts[i+1])
            discarded += int((~keep).sum())
            chunks.append(np.column_stack((t, ev['x'][keep], ev['y'][keep], ev['polarity'][keep])))
        events = np.concatenate(chunks) if chunks else np.empty((0, 4))
        save_events(out / 'events.npz', *events.T, meta['duration'], meta['width'], meta['height'])
        record = dict(variant='V2CE with per-pair physical-time LDATI adapter',
                      input_frames='Unmodified shared crop; no temporal frame interpolation',
                      clock='t = frame_ts[i] + timestamp_us * 1e-6 * 30 * diff(frame_ts)[i]',
                      sampler_fps=30., seed=123, dropped_endpoint_events=discarded,
                      frame_intervals_s=np.diff(ts).tolist(),
                      **hashes,
                      limitations=['Network has no dt input; variable-cadence accuracy is not established.',
                                   'Nominal sampler timestamps are quantized to microseconds before mapping.',
                                   'This adapter differs from the upstream fixed-FPS CLI protocol.'])
        (out / 'representation.json').write_text(json.dumps(record, indent=2))

    return generate


def main():
    assets, run = (Path(p).resolve() for p in sys.argv[1:3])
    load_generator(assets)(run)


if __name__ == '__main__':
    main()
