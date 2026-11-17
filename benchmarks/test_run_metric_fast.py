"""Regression checks for parallel reports, arrays, caching, and resume."""
import copy
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from benchmarks import run_metric as base
from benchmarks import run_metric_fast as fast


def events(times):
    count = len(times)
    return dict(t=np.array(times, dtype=float), x=np.zeros(count, dtype=int),
                y=np.zeros(count, dtype=int), p=np.where(np.arange(count) % 2, -1, 1))


class FastMetrics(unittest.TestCase):
    def test_cli_equivalence_arrays_resume_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, times in [('a', [10.001, 10.005, 10.01, 10.02]), ('b', [])]:
                source = root / 'data/toy/test' / name
                (source / 'frames').mkdir(parents=True)
                for index in range(2):
                    Image.new('L', (16, 16)).save(source / 'frames' / f'{index}.png')
                np.save(source / 'frame_ts.npy', [10., 10.1])
                for key, array in events(times).items():
                    np.save(source / f'ev_{key}.npy', array)
                meta = base.v2.sequence_info(source)[3]
                dest = root / 'pred/toy/test/evtween_v7_student' / name
                dest.mkdir(parents=True)
                np.savez(dest / 'events.npz', **events([.002, .004, .01, .099]))
                (dest / 'input.json').write_text(json.dumps(meta))
                native = root / 'pred/toy/test/v2v' / name
                native.mkdir(parents=True)
                np.savez(native / 'voxels.npz', voxels=np.zeros((1, 16, 16)))
            common = ['toy', 'test', '--methods', 'evtween_v7_student', 'v2v', 'v2ce',
                      '--data-root', str(root / 'data'), '--pred-root', str(root / 'pred'),
                      '--device', 'cpu', '--workers', '1']

            def run(parallel, metrics, jobs=2):
                script = Path(__file__).with_name('run_metric_fast.py' if parallel else 'run_metric.py')
                cmd = [sys.executable, str(script), *common, '--report-root',
                       str(root / ('fast' if parallel else 'serial')), '--metric', *metrics]
                if parallel:
                    cmd += ['--jobs', str(jobs)]
                result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            def read(folder, metric):
                reports = list((root / folder / 'toy' / metric).glob('*/results.csv'))
                self.assertEqual(len(reports), 1)
                with reports[0].open() as stream:
                    return reports[0].parent, list(csv.DictReader(stream))

            for metric in fast.ALL_METRICS:
                run(False, [metric])
            run(True, ['all'])
            for metric in fast.ALL_METRICS:
                serial_folder, serial = read('serial', metric)
                fast_folder, parallel = read('fast', metric)
                self.assertEqual(serial, parallel)
                for row in serial:
                    if row.get('array_path'):
                        with np.load(serial_folder / row['array_path']) as a, np.load(fast_folder / row['array_path']) as b:
                            self.assertEqual(a.files, b.files)
                            for key in a.files:
                                np.testing.assert_array_equal(a[key], b[key])
            output = run(True, ['all'], jobs=1)
            self.assertEqual(output.count('SKIP '), 58)
            self.assertEqual(output.count('missing_prediction '), 22)
            # Losing one array triggers recomputation of just that score.
            folder, rows = read('fast', 'event_rate')
            victim = next(row for row in rows if row['method'] == 'evtween_v7_student')
            (folder / victim['array_path']).unlink()
            output = run(True, ['all'])
            self.assertEqual(output.count('SKIP '), 57)
            # A single changed prediction invalidates all its metric records.
            np.savez(root / 'pred/toy/test/evtween_v7_student/a/events.npz', **events([.01]))
            output = run(True, ['all'])
            self.assertEqual(output.count('SKIP '), 47)
            for metric in fast.ALL_METRICS:
                run(False, [metric])
                self.assertEqual(read('serial', metric)[1], read('fast', metric)[1])

            # The direct worker entry point reads a prediction just once.
            args, _ = fast.arguments(['toy', 'test', '--metric', 'event_count', '--workers', '1', '--jobs', '1'])
            source = root / 'data/toy/test/a'
            file = root / 'pred/toy/test/evtween_v7_student/a/events.npz'
            specs = []
            for metric in ('event_count', 'on_percentage', 'event_rate'):
                single = copy.copy(args)
                single.metric = metric
                folder = root / 'cache_test' / metric
                folder.mkdir(parents=True)
                with sqlite3.connect(folder / '.resume.sqlite') as db:
                    db.execute(fast.SCHEMA)
                specs.append((single, base.settings(single), folder))
            with patch.object(base, 'read_events', wraps=base.read_events) as reader:
                self.assertFalse(fast.evaluate_job('a', 'evtween_v7_student', source, file, specs))
                self.assertEqual(reader.call_count, 1)


if __name__ == '__main__':
    unittest.main()
