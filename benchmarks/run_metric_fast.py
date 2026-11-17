"""Parallel full-sequence metrics with input reuse and resumable CSV reports.

--metric all selects event_count, event_count_ratio, on_percentage, event_rate,
isi_distribution, isi_median_ms, isi_cv, iei_distance, same_ts, fano_factor and
chamfer_full. Use run_voxel_metric_v2_fast.py for fixed-duration v2 voxel scores.
--jobs controls concurrent sequence/model processes; --workers controls CPU
nearest-neighbour query threads per Chamfer job. Defaults share allocated CPUs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, closing
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys

# Chamfer's explicit query threads are separate from numerical-library pools.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import run_metric as base

ALL_METRICS = ('event_count', 'event_count_ratio', 'on_percentage', 'event_rate',
               'isi_distribution', 'isi_median_ms', 'isi_cv', 'iei_distance',
               'same_ts', 'fano_factor', 'chamfer_full')
RETRY = {'error', 'missing_source', 'missing_gt', 'missing_prediction'}
SCHEMA = ('CREATE TABLE IF NOT EXISTS scores (sequence TEXT, method TEXT, '
          'signature TEXT, record TEXT, PRIMARY KEY(sequence, method))')


def signature(paths):
    return base.digest([base.stamp(path) for path in paths])


def evaluate_job(name, method, source, file, specs):
    """Reuse one aligned GT and prediction across the pending metrics."""
    seq = base.Sequence(source)
    paths = list(dict.fromkeys(path for args, _, _ in specs
                              for path in base.score_paths(source, method, file, args.metric)))
    initial = signature(paths)
    events = None

    def read_prediction(path, meta):
        nonlocal events
        if events is None:
            events = base.read_events(path, meta)
        return events

    print(f'RUN {name}/{method} ({len(specs)} metrics)', flush=True)
    failed = False
    with closing(sqlite3.connect(':memory:')) as db:
        db.execute(SCHEMA)
        for args, config, folder in specs:
            # The serial runner has one database per metric.
            db.execute('DELETE FROM scores')
            base.evaluate_one(args, seq, name, method, file, folder, db, config,
                              read_prediction=read_prediction)
            saved_signature, record = db.execute(
                'SELECT signature,record FROM scores WHERE sequence=? AND method=?',
                (name, method)).fetchone()
            row = json.loads(record)
            # A cached input must not be attributed to a newer file version.
            if initial != signature(paths):
                row.update(value=None, status='error', array_path='',
                           error='Inputs changed during scoring; rerun')
            publish_result(folder, args.metric, saved_signature, row)
            failed |= row['status'] == 'error'
            print(f'{row["status"]} {name}/{method}/{args.metric}', flush=True)
    return failed


def publish_result(folder, metric, saved_signature, row):
    """Commit each completed metric; serialize only the short publication step."""
    db = sqlite3.connect(folder / '.resume.sqlite', timeout=3600)
    try:
        with db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?,?)',
                       (row['sequence'], row['method'], saved_signature,
                        json.dumps(row, allow_nan=False)))
            base.export_csv(db, folder, metric)
    finally:
        db.close()


def reusable(db, source, name, method, file, metric, folder):
    saved = db.execute('SELECT signature,record FROM scores WHERE sequence=? AND method=?',
                       (name, method)).fetchone()
    if saved is None or saved[0] != signature(base.score_paths(source, method, file, metric)):
        return False
    row = json.loads(saved[1])
    return row['status'] not in RETRY and (
        not row.get('array_path') or (folder / row['array_path']).is_file())


def arguments(argv):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset')
    parser.add_argument('split', choices=('train', 'test'))
    parser.add_argument('--metric', nargs='+', choices=(*base.UNITS, 'all'), default=['all'])
    parser.add_argument('--methods', nargs='+', choices=base.v2.old.METHODS)
    parser.add_argument('--sequence')
    parser.add_argument('--jobs', type=int, default=min(8, cpus),
                        help='Concurrent sequence/model processes (default: up to 8)')
    parser.add_argument('--workers', type=int,
                        help='CPU Chamfer threads per job (default: allocated CPUs / jobs)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-points', type=int)
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--rate-bin-s', type=float, default=0.01)
    parser.add_argument('--count-weight', type=float, default=1.0)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--pred-root', type=Path, default=base.ROOT / 'data_test_evt')
    parser.add_argument('--report-root', type=Path, default=base.ROOT / 'reports')
    parser.add_argument('--voxel-block-size', type=int, default=1)
    parser.add_argument('--voxel-temporal-bins', type=int, default=160)
    parser.add_argument('--legacy-frame-voxels', action='store_true')
    args = parser.parse_args(argv)
    if Path(args.dataset).name != args.dataset or args.dataset in ('', '.', '..'):
        parser.error('Expected a single dataset directory name')
    metrics = list(dict.fromkeys(metric for value in args.metric
                                for metric in (ALL_METRICS if value == 'all' else (value,))))
    if args.jobs < 1 or (args.workers is not None and args.workers < 1):
        parser.error('--jobs and --workers must be positive')
    args.jobs = min(args.jobs, cpus)
    args.workers = args.workers or max(1, cpus // args.jobs)
    if args.device == 'cpu' and any(m in base.FULL_CHAMFERS for m in metrics):
        if args.jobs * args.workers > cpus:
            parser.error(f'--jobs * --workers must be <= {cpus} allocated CPUs')
    if args.device == 'cuda' and args.jobs > 1:
        print('CUDA jobs share GPU memory; reduce --jobs if memory is insufficient.', flush=True)
    if args.max_points is not None and metrics != ['chamfer']:
        parser.error('--max-points applies only to an explicit sampled chamfer run')
    if args.max_points is not None and args.max_points < 1:
        parser.error('--max-points must be positive')
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    if not np.isfinite(args.rate_bin_s) or args.rate_bin_s <= 0:
        parser.error('--rate-bin-s must be finite and positive')
    if not np.isfinite(args.count_weight) or args.count_weight < 0:
        parser.error('--count-weight must be finite and nonnegative')
    if args.voxel_temporal_bins < 1 or args.voxel_block_size < 1:
        parser.error('Voxel bin and block counts must be positive')
    if args.legacy_frame_voxels and any(not m.startswith('voxel_') for m in metrics):
        parser.error('--legacy-frame-voxels requires only voxel_* metrics')
    if args.voxel_block_size != 1 and (not args.legacy_frame_voxels or
            any(m not in ('voxel_mae', 'voxel_rmse') for m in metrics)):
        parser.error('Block averaging requires legacy voxel_mae/voxel_rmse')
    for metric in metrics:
        if metric in base.PMSE and args.voxel_temporal_bins < int(metric.split('_')[1]):
            parser.error('PMSE requires at least kernel-size temporal bins')
    return args, metrics


def main(argv=None):
    args, metrics = arguments(argv)
    data = args.data_root or base.v2.old.setting('EVTWEEN_DATA', base.v2.old.LAB / 'evtween/data')
    source_split = data / args.dataset / args.split
    pred = args.pred_root / args.dataset / args.split
    dirs = [pred / method for method in base.v2.old.METHODS if (pred / method).is_dir()]
    names = {path.name for folder in dirs for path in folder.iterdir() if path.is_dir()}
    if source_split.is_dir():
        names.update(path.name for path in source_split.iterdir() if path.is_dir())
    if args.sequence:
        if args.sequence not in names:
            raise ValueError(f'Unknown sequence: {args.sequence}')
        names = {args.sequence}
    if not names:
        raise ValueError('No sequences found; check input paths')
    methods = list(dict.fromkeys(args.methods or base.v2.old.METHODS))
    specs, databases = {}, {}
    with ExitStack() as stack:
        for metric in sorted(metrics):
            single = copy.copy(args)
            single.metric = metric
            single.max_points = args.max_points or 20000
            single.chunk_size = args.chunk_size or (131072 if metric in base.FULL_CHAMFERS else 4096)
            config = base.settings(single)
            config['code_sha256'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            folder = args.report_root / args.dataset / metric / f'{args.split}-full-{base.digest(config)[:12]}'
            folder.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((folder / '.fast.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Another fast runner is writing {folder}') from exc
            with base.atomic_file(folder / 'config.json') as stream:
                json.dump(config, stream, indent=2)
            db = sqlite3.connect(folder / '.resume.sqlite', timeout=3600)
            stack.callback(db.close)
            db.execute(SCHEMA)
            specs[metric] = (single, config, folder)
            databases[metric] = db
            print(f'Results: {folder / "results.csv"}', flush=True)

        tasks = []
        for name in sorted(names):
            source = base.source_for(name, source_split, dirs, args)
            selected_methods = methods + (['ground_truth'] if any(m in base.DESCRIPTIVE for m in metrics) else [])
            for method in selected_methods:
                file = pred / method / name / ('voxels.npz' if method == 'v2v' else 'events.npz')
                pending = []
                for metric in metrics:
                    if method == 'ground_truth' and metric not in base.DESCRIPTIVE:
                        continue
                    folder = specs[metric][2]
                    if reusable(databases[metric], source, name, method, file, metric, folder):
                        print(f'SKIP {name}/{method}/{metric}', flush=True)
                    else:
                        pending.append(specs[metric])
                if pending:
                    tasks.append((name, method, source, file, pending))
        for metric, db in databases.items():
            with db:
                base.export_csv(db, specs[metric][2], metric)
        if not tasks:
            return 0
        jobs = min(args.jobs, len(tasks))
        print(f'Running {len(tasks)} sequence/model jobs, {jobs} concurrent; '
              f'{args.workers} CPU Chamfer threads per job', flush=True)
        failed = False
        pool = ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn'))
        try:
            futures = [pool.submit(evaluate_job, *task) for task in tasks]
            for future in as_completed(futures):
                failed |= future.result()
        except BaseException:
            # Python 3.10 lacks a public executor API to terminate workers.
            for process in list((pool._processes or {}).values()):
                if process.is_alive():
                    process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
