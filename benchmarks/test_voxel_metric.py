import unittest
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
from benchmarks import voxel_metric as v
from benchmarks import run_metric as r


def events(t=(), x=None, y=None, p=None):
    n = len(t)
    return dict(t=np.asarray(t, dtype=float),
                x=np.zeros(n, dtype=int) if x is None else np.asarray(x),
                y=np.zeros(n, dtype=int) if y is None else np.asarray(y),
                p=np.ones(n) if p is None else np.asarray(p))


class VoxelMetrics(unittest.TestCase):
    domain = dict(start=10., stop=11., temporal_bins=4, height=4, width=4)

    def test_temporal_weights_and_endpoints(self):
        grid = v._event_voxels(events([9., 10., 10.125, 10.875, 11.]), **self.domain)
        np.testing.assert_allclose(grid[0, :, 0, 0], [1.5, .5, 0., .5])
        self.assertEqual(grid[1].sum(), 0)

    def test_pmse_analytical_and_normalization(self):
        empty = events()
        one = events([10.])
        # One event / 8 cells, squared, averaged over 2 polarities * 8 blocks.
        self.assertEqual(v.pmse_2_v2ce(one, empty, **self.domain), 1/1024)
        self.assertEqual(v.pmse_4_v2ce(one, empty, **self.domain), 1/8192)
        self.assertEqual(v.pmse_2_v2ce(events([10., 10.]), empty, **self.domain), 4/1024)
        self.assertEqual(v.pmse_2_v2ce(one, one, **self.domain), 0)
        self.assertEqual(v.pmse_4_v2ce(empty, empty, **self.domain), 0)

    def test_polarity_does_not_cancel(self):
        on, off = events([10.]), events([10.], p=[0])
        self.assertEqual(v.pmse_2_v2ce(on, off, **self.domain), 2/1024)
        np.testing.assert_equal(v._event_voxels(off, **self.domain),
                                v._event_voxels(events([10.], p=[-1]), **self.domain))

    def test_temporal_pooling(self):
        self.assertEqual(v.pmse_2_v2ce(events([10.]), events([10.25]), **self.domain), 0)
        self.assertGreater(v.pmse_2_v2ce(events([10.]), events([10.5]), **self.domain), 0)

    def test_valid_pooling_remainder(self):
        domain = dict(self.domain, height=5, width=5, temporal_bins=5)
        self.assertEqual(v.pmse_2_v2ce(events([10.], x=[4]), events(), **domain), 0)
        with self.assertRaises(ValueError):
            v.pmse_4_v2ce(events(), events(), **dict(self.domain, temporal_bins=3))

    def test_validation(self):
        for bad in (events([10.], x=[.5]), events([np.nan]), events([10.], p=[2])):
            with self.assertRaises(ValueError):
                v.pmse_2_v2ce(bad, events(), **self.domain)

    def test_runner_shared_alignment_and_frame_segmentation(self):
        gt, pred = events([.0]), events([.5])
        values = []
        for edges in ([0., .2, 1.], [0., .8, 1.]):
            seq = SimpleNamespace(geometry=(np.asarray(edges), dict(width=4, height=4)), gt=gt)
            values.append(r.score('pmse_2_v2ce', pred, seq, dict(temporal_bins=4))[0])
            self.assertEqual(r.score('voxel_mae', pred, seq, dict(temporal_bins=4))[0], 2/64)
        self.assertEqual(values[0], values[1])
        self.assertGreater(values[0], 0)

    def test_legacy_api(self):
        a = np.zeros((2, 2, 2))
        b = a.copy(); b[0, 0, 0] = 2
        self.assertEqual(v.voxel_mae(a, b), .25)
        self.assertEqual(v.voxel_rmse(a, b), np.sqrt(.5))
        self.assertEqual(v.voxel_relative_l1(a, b), 1)
        self.assertTrue(np.isnan(v.voxel_mae(a, b, np.zeros((2, 2), dtype=bool))))

    def test_cli_reports_resume_and_native_rejection(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'data'/'toy'/'test'/'seq'
            (source/'frames').mkdir(parents=True)
            for i in range(3):
                Image.new('L', (4, 4)).save(source/'frames'/f'{i}.png')
            np.save(source/'frame_ts.npy', [10., 10.2, 11.])
            gt = events([10.])
            for key, a in gt.items():
                np.save(source/f'ev_{key}.npy', a)
            _, _, edges, meta, _ = r.v2.sequence_info(source)
            for method in ('v2ce', 'v2v'):
                dest = root/'pred'/'toy'/'test'/method/'seq'
                dest.mkdir(parents=True)
                (dest/'input.json').write_text(json.dumps(meta))
                if method == 'v2ce':
                    np.savez(dest/'events.npz', **events([.5]))
                else:
                    np.savez(dest/'voxels.npz', edges_s=edges, voxels=np.zeros((2, 4, 4)))
            command = [sys.executable, str(Path(r.__file__)), 'toy', 'test',
                       '--methods', 'v2ce', 'v2v', '--data-root', str(root/'data'),
                       '--pred-root', str(root/'pred'), '--report-root', str(root/'reports'),
                       '--voxel-temporal-bins', '4']
            for metric in ('pmse_2_v2ce', 'pmse_4_v2ce', 'voxel_mae'):
                result = subprocess.run(command + ['--metric', metric], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = next((root/'reports'/'toy'/metric).glob('*/results.csv'))
                with report.open() as stream:
                    rows = {a['method']: a for a in csv.DictReader(stream)}
                self.assertEqual(rows['v2ce']['status'], 'ok')
                self.assertEqual(rows['v2v']['status'], 'not_applicable')
                repeated = subprocess.run(command + ['--metric', metric], capture_output=True, text=True)
                self.assertEqual(repeated.returncode, 0, repeated.stderr)
                self.assertIn('SKIP seq/v2ce', repeated.stdout)
            legacy = subprocess.run(command + ['--metric', 'voxel_mae', '--legacy-frame-voxels'],
                                    capture_output=True, text=True)
            self.assertEqual(legacy.returncode, 0, legacy.stdout + legacy.stderr)
            self.assertIn('ok seq/v2v', legacy.stdout)


if __name__ == '__main__':
    unittest.main()
