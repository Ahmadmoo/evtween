"""CLI regression checks for parallel scoring, resume, and input changes."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from benchmarks import run_metric as old
from benchmarks.run_voxel_metric_v2 import METRICS


def events(times):
    count = len(times)
    return dict(t=np.array(times, dtype=float), x=np.zeros(count, dtype=int),
                y=np.zeros(count, dtype=int), p=np.ones(count, dtype=int))


class ParallelVoxelRunner(unittest.TestCase):
    def test_equivalence_resume_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, times in [('a', [10.01, 10.02]), ('b', [])]:
                source = root / 'data/toy/test' / name
                (source / 'frames').mkdir(parents=True)
                for index in range(2):
                    Image.new('L', (7, 5)).save(source / 'frames' / f'{index}.png')
                np.save(source / 'frame_ts.npy', [10., 10.1])
                for key, array in events(times).items():
                    np.save(source / f'ev_{key}.npy', array)
                meta = old.v2.sequence_info(source)[3]
                dest = root / 'pred/toy/test/evtween_v7_student' / name
                dest.mkdir(parents=True)
                np.savez(dest / 'events.npz', **events([.01, .099]))
                (dest / 'input.json').write_text(json.dumps(meta))

            common = ['toy', 'test', '--metric', 'all', '--voxel-bin-ms', '30',
                      '--methods', 'evtween_v7_student', 'v2v', 'v2ce',
                      '--data-root', str(root / 'data'), '--pred-root', str(root / 'pred')]

            def run(fast, workers=2):
                script = Path(__file__).with_name(
                    'run_voxel_metric_v2_fast.py' if fast else 'run_voxel_metric_v2.py')
                cmd = [sys.executable, str(script), *common, '--report-root',
                       str(root / ('fast' if fast else 'serial'))]
                if fast:
                    cmd += ['--workers', str(workers)]
                result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            def rows(folder, metric):
                reports = list((root / folder / 'toy' / metric).glob('*/results.csv'))
                self.assertEqual(len(reports), 1)
                with reports[0].open() as stream:
                    return sorted(csv.DictReader(stream), key=lambda row: (row['sequence'], row['method']))

            run(False)
            run(True)
            for metric in METRICS:
                self.assertEqual(rows('serial', metric), rows('fast', metric))
            # Concurrency does not change report identity; missing files are retried.
            output = run(True, workers=1)
            self.assertEqual(output.count('SKIP '), 20)
            self.assertEqual(output.count('missing_prediction '), 10)
            # Changing one input invalidates only that sequence/model's records.
            file = root / 'pred/toy/test/evtween_v7_student/a/events.npz'
            np.savez(file, **events([.01]))
            output = run(True)
            self.assertEqual(output.count('SKIP '), 15)
            run(False)
            for metric in METRICS:
                self.assertEqual(rows('serial', metric), rows('fast', metric))


if __name__ == '__main__':
    unittest.main()
