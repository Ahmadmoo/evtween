"""Print equal-sequence means from saved reports without recomputing metrics.

Usage: python benchmarks/summarize_metrics.py bsergb --voxel-bin-ms 10
Different scoring protocols are printed separately, never averaged together.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def protocol_key(config):
    """Ignore execution and selection settings, retaining scoring parameters."""
    options = dict(config.get('options', {}))
    for key in ('device', 'workers', 'chunk_size', 'backend_version', 'torch_version', 'provenance'):
        options.pop(key, None)
    if options.get('block_size') == 1:
        options.pop('block_size')
    return json.dumps({k: config.get(k) for k in ('unit', 'interval', 'crop')} | {'options': options}, sort_keys=True)


def summarize_dataset(dataset, split='test', report_roots=None, metrics=None,
                      methods=None, voxel_bin_ms=None, include_partial=False, shared=False):
    """Return per-method means, coverage and source runs, plus read warnings.

    Each sequence contributes once. Duplicate scores prefer valid rows and then
    the newest report. ISI distributions are summarized by per-sequence medians.
    Source code hashes are not treated as protocols; scoring options are.
    """
    roots = report_roots or [ROOT / 'reports', ROOT / 'reports_voxel_v2', ROOT / 'reports_voxel_v2_ab']
    groups, warnings, seen = {}, [], set()
    for root in roots:
        for cfg_path in sorted((Path(root) / dataset).glob('*/*/config.json')):
            if cfg_path.resolve() in seen:
                continue
            seen.add(cfg_path.resolve())
            try:
                cfg = json.loads(cfg_path.read_text())
                metric = cfg.get('metric')
                if cfg.get('dataset') != dataset or cfg.get('split') != split or not metric:
                    continue
                if metrics and metric not in metrics:
                    continue
                bin_ms = cfg.get('options', {}).get('bin_ms')
                if voxel_bin_ms is not None and (metric.startswith(('voxel_', 'pmse_'))):
                    if bin_ms is None or not np.isclose(float(bin_ms), voxel_bin_ms, rtol=0, atol=1e-9):
                        continue
                path = cfg_path.with_name('results.csv')
                rows = pd.read_csv(path)
                required = {'sequence', 'method', 'value', 'status'}
                if not required.issubset(rows.columns):
                    raise ValueError(f'Missing columns: {sorted(required - set(rows.columns))}')
                if rows.duplicated(['sequence', 'method']).any():
                    raise ValueError('Duplicate sequence/method rows in one CSV')
                if methods:
                    rows = rows[rows.method.isin(methods)].copy()
                rows['_value'] = pd.to_numeric(rows.value, errors='coerce')
                if metric == 'isi_distribution':
                    rows['_value'] = np.nan
                    for index, row in rows.iterrows():
                        if not isinstance(row.get('array_path'), str):
                            continue
                        try:
                            with np.load(path.parent / row.array_path, allow_pickle=False) as data:
                                isi = data['isi_ms']
                                isi = isi[np.isfinite(isi)]
                                if isi.size:
                                    rows.loc[index, '_value'] = np.median(isi)
                        except (OSError, ValueError, KeyError) as exc:
                            warnings.append(f'{path}: {row.sequence}/{row.method}: {exc}')
                rows['_valid'] = rows.status.eq('ok')
                if include_partial and metric == 'fano_factor':
                    rows['_valid'] |= rows.status.eq('partial_support')
                rows['_valid'] &= np.isfinite(rows['_value'])
                rows['_mtime'] = path.stat().st_mtime_ns
                rows['_source'] = str(path)
                key = (metric, protocol_key(cfg))
                groups.setdefault(key, []).append(rows)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f'{cfg_path}: {exc}')
    output = []
    for (metric, protocol), frames in sorted(groups.items()):
        rows = pd.concat(frames, ignore_index=True).sort_values(['_valid', '_mtime', '_source'], kind='stable')
        rows = rows.drop_duplicates(['sequence', 'method'], keep='last')
        good = rows[rows['_valid']]
        if shared:
            names = methods or sorted(set(rows.method) - {'ground_truth'})
            sets = [set(good.loc[good.method.eq(m), 'sequence']) for m in names]
            common = set.intersection(*sets) if sets else set()
            good = good[good.sequence.isin(common)]
        settings = json.loads(protocol)
        for method, all_rows in rows.groupby('method', sort=True):
            valid = good[good.method.eq(method)]
            output.append(dict(metric=metric, protocol=hashlib.sha256(protocol.encode()).hexdigest()[:10],
                               method=method, mean=valid['_value'].mean(), n=len(valid),
                               available=len(all_rows), excluded=len(all_rows)-len(valid),
                               partial=int(valid.status.eq('partial_support').sum()),
                               unit='ms (mean sequence median)' if metric == 'isi_distribution' else settings.get('unit'),
                               bin_ms=settings['options'].get('bin_ms'),
                               runs=';'.join(sorted(set(Path(p).parent.name for p in valid['_source']))),
                               settings=protocol))
    return pd.DataFrame(output), warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--split', choices=['test', 'train'], default='test')
    parser.add_argument('--report-roots', nargs='+', type=Path)
    parser.add_argument('--metrics', nargs='+')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--voxel-bin-ms', type=float)
    parser.add_argument('--include-partial-fano', action='store_true')
    parser.add_argument('--shared-sequences', action='store_true', help='Use intersection across selected models per metric/protocol')
    parser.add_argument('--output', type=Path, help='Optionally save this summary as CSV')
    args = parser.parse_args(argv)
    if Path(args.dataset).name != args.dataset or args.dataset in ('', '.', '..'):
        parser.error('Expected a dataset directory name')
    if args.voxel_bin_ms is not None and (not np.isfinite(args.voxel_bin_ms) or args.voxel_bin_ms <= 0):
        parser.error('--voxel-bin-ms must be finite and positive')
    table, warnings = summarize_dataset(args.dataset, args.split, args.report_roots, args.metrics,
                                        args.methods, args.voxel_bin_ms, args.include_partial_fano,
                                        args.shared_sequences)
    for warning in warnings:
        print('WARNING:', warning, file=sys.stderr)
    if table.empty:
        print('No matching saved reports found. Check dataset, split, report roots and voxel bin.', file=sys.stderr)
        return 1
    print(f'{args.dataset}/{args.split}: equal-weight sequence means; no metric recomputation.')
    print('n = used sequences; available = saved sequence rows; excluded = invalid or outside shared cohort.')
    print('Distinct scoring settings are separate sections. Duplicate rows prefer valid, then newest reports.')
    for (metric, protocol), part in table.groupby(['metric', 'protocol'], sort=False):
        print(f'\n{metric} | protocol={protocol} | unit={part.iloc[0]["unit"]}')
        print('Settings:', part.iloc[0]['settings'])
        print(part[['method', 'mean', 'n', 'available', 'excluded', 'partial']].to_string(index=False, float_format=lambda v: f'{v:.7g}'))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f'\nSaved: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
