"""Parallel fixed-duration voxel v2 scoring with input reuse and per-result resume.

Each worker loads one sequence/model pair and calls the existing v2 metric
functions. --workers changes concurrency only. PMSE still builds dense grids,
so choose worker count according to host RAM and sequence duration.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

# Each process uses one numerical-library thread, avoiding nested CPU pools.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import run_metric as old
from benchmarks import voxel_metric_v2 as metric_v2
from benchmarks.run_voxel_metric_v2 import METRICS

FIELDS = ['sequence', 'method', 'value', 'value_on', 'value_off', 'value_aware',
          'value_total', 'status', 'start_s', 'stop_s', 'bin_ms', 'temporal_bins',
          'last_bin_ms', 'height', 'width', 'error']
REUSABLE = {'ok', 'undefined', 'not_applicable'}


def input_signature(paths):
    return old.digest([old.stamp(path) for path in paths])


def evaluate(args, name, method, source, file, metrics):
    """Load prediction and GT once; preserve each v2 metric's numeric path."""
    rows = {metric: dict(sequence=name, method=method, status='ok',
                         bin_ms=args.voxel_bin_ms) for metric in metrics}
    paths = [file, file.parent / 'input.json', source / 'frame_ts.npy',
             *(source / f'ev_{key}.npy' for key in 'txyp')]
    before = input_signature(paths)
    print(f'RUN {name}/{method}', flush=True)
    try:
        if method == 'v2v':
            for row in rows.values():
                row['status'] = 'not_applicable'
        elif not file.is_file():
            for row in rows.values():
                row['status'] = 'missing_prediction'
        else:
            seq = old.Sequence(source)
            source_edges, meta = seq.geometry
            start, stop = metric_v2.scoring_window(
                source_edges, meta['absolute_t0'], args.voxel_start_s, args.voxel_stop_s)
            edges = metric_v2.time_edges(start, stop, args.voxel_bin_ms)
            old.verify_metadata(file.parent / 'input.json', meta, args, name)
            events, reference = old.read_events(file, meta), seq.gt
            for metric, row in rows.items():
                row.update(start_s=start, stop_s=stop, temporal_bins=len(edges)-1,
                           last_bin_ms=float(np.diff(edges)[-1]*1000),
                           height=meta['height'], width=meta['width'])
                try:
                    scores = metric_v2.score(metric, events, reference, edges=edges,
                                            height=meta['height'], width=meta['width'],
                                            bin_ms=args.voxel_bin_ms)
                    row.update({key: float(value) if np.isfinite(value) else None
                                for key, value in scores.items()})
                    if row['value'] is None:
                        row['status'] = 'undefined'
                except Exception as exc:
                    row.update(status='error', value=None, error=str(exc))
        if before != input_signature(paths):
            raise RuntimeError('Input changed during evaluation; rerun')
    except Exception as exc:
        for row in rows.values():
            row.update(status='error', value=None, error=str(exc))
    return before, rows


def configuration(args, metric):
    options=dict(protocol=metric_v2.PROTOCOL,bin_ms=args.voxel_bin_ms,start_s=args.voxel_start_s,
                 stop_s=args.voxel_stop_s,final_bin='keep shorter',normalization='none',
                 pmse_boundary='fold overflow into last bin',pooling='partial means; original-cell weighted MSE',
                 cell_weight='equal per original voxel; short final bin is not duration-normalized',
                 provenance='https://arxiv.org/html/2309.08891v2; benchmark adaptation, not exact reproduction')
    files=[Path(__file__),Path(metric_v2.__file__),Path(old.__file__),Path(old.voxel_metric.__file__),Path(old.m.__file__),Path(old.v2.__file__)]
    config=dict(dataset=args.dataset,split=args.split,metric=metric,options=options,
                unit='squared events/voxel' if metric.startswith('pmse_') else old.UNITS[metric],
                selection=dict(methods=sorted(set(args.methods or old.v2.old.METHODS)),sequence=args.sequence),
                numpy_version=np.__version__,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return config


def export(folder, records):
    """Only the parent publishes sorted CSVs, using atomic replacement."""
    with old.atomic_file(folder / 'results.csv') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('split',choices=('train','test'))
    parser.add_argument('--metric',choices=(*METRICS,'all'),required=True)
    parser.add_argument('--voxel-bin-ms',type=float,required=True)
    parser.add_argument('--voxel-start-s',type=float,default=0.,help='Seconds relative to first source frame')
    parser.add_argument('--voxel-stop-s',type=float,help='Default: last source frame, in relative seconds')
    parser.add_argument('--methods',nargs='+',choices=old.v2.old.METHODS)
    parser.add_argument('--sequence')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--pred-root',type=Path,default=old.ROOT/'data_test_evt')
    parser.add_argument('--report-root',type=Path,default=old.ROOT/'reports_voxel_v2')
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent sequence/model processes (default: 4)')
    args=parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if Path(args.dataset).name != args.dataset or args.dataset in ('','.','..'):
        parser.error('Expected one dataset directory name')
    if not np.isfinite([args.voxel_bin_ms,args.voxel_start_s]).all() or args.voxel_bin_ms<=0 or args.voxel_start_s<0:
        parser.error('Require finite positive bin width and nonnegative start')
    if args.voxel_stop_s is not None and (not np.isfinite(args.voxel_stop_s) or args.voxel_stop_s<=args.voxel_start_s):
        parser.error('Require finite stop > start')
    metrics = METRICS if args.metric == 'all' else (args.metric,)
    methods = list(dict.fromkeys(args.methods or old.v2.old.METHODS))
    data = args.data_root or old.v2.old.setting('EVTWEEN_DATA', old.v2.old.LAB / 'evtween/data')
    split_dir = data / args.dataset / args.split
    pred = args.pred_root / args.dataset / args.split
    dirs = [pred / method for method in old.v2.old.METHODS if (pred / method).is_dir()]
    names = {path.name for folder in dirs for path in folder.iterdir() if path.is_dir()}
    if split_dir.is_dir():
        names.update(path.name for path in split_dir.iterdir() if path.is_dir())
    if args.sequence:
        if args.sequence not in names:
            parser.error('Unknown sequence')
        names = {args.sequence}
    if not names:
        parser.error('No sequences found; check input paths')
    folders, records = {}, {metric: {} for metric in metrics}
    with ExitStack() as stack:
        # Stable lock order prevents concurrent writers to any selected report.
        for metric in sorted(metrics):
            config = configuration(args, metric)
            folder = args.report_root / args.dataset / metric / f'{args.split}-v2-{old.digest(config)[:12]}'
            folder.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((folder / '.fast.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error(f'Another fast runner is writing {folder}')
            folders[metric] = folder
            (folder / '.rows').mkdir(exist_ok=True)
            with old.atomic_file(folder / 'config.json') as stream:
                json.dump(config, stream, indent=2)
            print(f'Results: {folder / "results.csv"}', flush=True)

        tasks = []
        for name in sorted(names):
            source = old.source_for(name, split_dir, dirs, args)
            for method in methods:
                file = pred / method / name / 'events.npz'
                paths = [file, file.parent / 'input.json', source / 'frame_ts.npy',
                         *(source / f'ev_{key}.npy' for key in 'txyp')]
                signature = input_signature(paths)
                key = (name, method)
                cache_name = old.digest(key) + '.json'
                pending = []
                for metric in metrics:
                    cache = folders[metric] / '.rows' / cache_name
                    try:
                        saved = json.loads(cache.read_text())
                        row = saved['row']
                        valid = (saved['signature'] == signature and
                                 (row['sequence'], row['method']) == key and
                                 row['status'] in REUSABLE)
                    except (OSError, ValueError, KeyError, TypeError):
                        valid = False
                    if valid:
                        records[metric][key] = row
                        print(f'SKIP {name}/{method}/{metric}', flush=True)
                    else:
                        pending.append(metric)
                if pending:
                    tasks.append((args, name, method, source, file, pending))
        for metric in metrics:
            export(folders[metric], records[metric])
        if not tasks:
            return 0
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
        workers = min(args.workers, cpus, len(tasks))
        print(f'Running {len(tasks)} sequence/model jobs with {workers} workers', flush=True)
        failed = False
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'))
        futures = {pool.submit(evaluate, *task): task for task in tasks}
        try:
            for future in as_completed(futures):
                task = futures[future]
                _, name, method, _, _, _ = task
                signature, rows = future.result()
                key = (name, method)
                for metric, row in rows.items():
                    folder = folders[metric]
                    with old.atomic_file(folder / '.rows' / (old.digest(key) + '.json')) as stream:
                        json.dump(dict(signature=signature, row=row), stream, allow_nan=False)
                    records[metric][key] = row
                    export(folder, records[metric])
                    failed |= row['status'] == 'error'
                    print(f'{row["status"]} {name}/{method}/{metric}', flush=True)
                    if row.get('error'):
                        print(f'  {row["error"]}', flush=True)
        except BaseException:
            # Python 3.10 has no public ProcessPoolExecutor termination API.
            for process in list(pool._processes.values()):
                if process.is_alive():
                    process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
