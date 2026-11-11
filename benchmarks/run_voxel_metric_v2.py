"""Isolated fixed-millisecond voxel benchmark runner; never overwrites v1 reports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks import run_metric as old
from benchmarks import voxel_metric_v2 as metric_v2

METRICS=("voxel_mae","voxel_rmse","voxel_relative_l1","pmse_2_v2ce_adapted","pmse_4_v2ce_adapted")


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
    args=parser.parse_args(argv)
    if Path(args.dataset).name != args.dataset or args.dataset in ('','.','..'):
        parser.error('Expected one dataset directory name')
    if not np.isfinite([args.voxel_bin_ms,args.voxel_start_s]).all() or args.voxel_bin_ms<=0 or args.voxel_start_s<0:
        parser.error('Require finite positive bin width and nonnegative start')
    if args.voxel_stop_s is not None and (not np.isfinite(args.voxel_stop_s) or args.voxel_stop_s<=args.voxel_start_s):
        parser.error('Require finite stop > start')
    if args.metric=='all':
        common=[args.dataset,args.split,'--voxel-bin-ms',str(args.voxel_bin_ms),
                '--voxel-start-s',str(args.voxel_start_s),'--pred-root',str(args.pred_root),
                '--report-root',str(args.report_root)]
        for name in ('voxel_stop_s','sequence','data_root'):
            value=getattr(args,name)
            if value is not None: common+=['--'+name.replace('_','-'),str(value)]
        if args.methods: common+=['--methods',*args.methods]
        return max(main([*common,'--metric',metric]) for metric in METRICS)
    options=dict(protocol=metric_v2.PROTOCOL,bin_ms=args.voxel_bin_ms,start_s=args.voxel_start_s,
                 stop_s=args.voxel_stop_s,final_bin='keep shorter',normalization='none',
                 pmse_boundary='fold overflow into last bin',pooling='partial means; original-cell weighted MSE',
                 cell_weight='equal per original voxel; short final bin is not duration-normalized',
                 provenance='https://arxiv.org/html/2309.08891v2; benchmark adaptation, not exact reproduction')
    files=[Path(__file__),Path(metric_v2.__file__),Path(old.__file__),Path(old.voxel_metric.__file__),Path(old.m.__file__),Path(old.v2.__file__)]
    config=dict(dataset=args.dataset,split=args.split,metric=args.metric,options=options,
                unit='squared events/voxel' if args.metric.startswith('pmse_') else old.UNITS[args.metric],
                selection=dict(methods=sorted(set(args.methods or old.v2.old.METHODS)),sequence=args.sequence),
                numpy_version=np.__version__,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    folder=args.report_root/args.dataset/args.metric/f'{args.split}-v2-{old.digest(config)[:12]}'
    folder.mkdir(parents=True,exist_ok=True)
    with old.atomic_file(folder/'config.json') as stream:
        json.dump(config,stream,indent=2)
    data=args.data_root or old.v2.old.setting('EVTWEEN_DATA',old.v2.old.LAB/'evtween/data')
    split_dir=data/args.dataset/args.split
    pred=args.pred_root/args.dataset/args.split
    dirs=[pred/m for m in old.v2.old.METHODS if (pred/m).is_dir()]
    names={p.name for d in dirs for p in d.iterdir() if p.is_dir()}
    if split_dir.is_dir():
        names.update(p.name for p in split_dir.iterdir() if p.is_dir())
    if args.sequence:
        if args.sequence not in names:
            parser.error('Unknown sequence')
        names={args.sequence}
    if not names:
        parser.error('No sequences found; check input paths')
    rows=[]; failed=False
    from csv import DictWriter
    fields=['sequence','method','value','value_on','value_off','value_aware','value_total','status','start_s','stop_s',
            'bin_ms','temporal_bins','last_bin_ms','height','width','error']
    def export():
        with old.atomic_file(folder/'results.csv') as stream:
            writer=DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    for name in sorted(names):
        seq=old.Sequence(old.source_for(name,split_dir,dirs,args))
        for method in dict.fromkeys(args.methods or old.v2.old.METHODS):
            row=dict(sequence=name,method=method,status='ok',bin_ms=args.voxel_bin_ms)
            try:
                if method=='v2v':
                    row['status']='not_applicable'
                else:
                    file=pred/method/name/'events.npz'
                    if not file.is_file():
                        row['status']='missing_prediction'
                    else:
                        paths=[file,file.parent/'input.json',seq.source/'frame_ts.npy',
                               *(seq.source/f'ev_{k}.npy' for k in 'txyp')]
                        before=[old.stamp(p) for p in paths]
                        source_edges,meta=seq.geometry
                        start,stop=metric_v2.scoring_window(source_edges,meta['absolute_t0'],
                                                          args.voxel_start_s,args.voxel_stop_s)
                        edges=metric_v2.time_edges(start,stop,args.voxel_bin_ms)
                        row.update(start_s=start,stop_s=stop,temporal_bins=len(edges)-1,
                                   last_bin_ms=float(np.diff(edges)[-1]*1000),height=meta['height'],width=meta['width'])
                        old.verify_metadata(file.parent/'input.json',meta,args,name)
                        events=old.read_events(file,meta)
                        scores=metric_v2.score(args.metric,events,seq.gt,edges=edges,height=meta['height'],
                                               width=meta['width'],bin_ms=args.voxel_bin_ms)
                        if before != [old.stamp(p) for p in paths]:
                            raise RuntimeError('Input changed during evaluation; rerun')
                        row.update({k:float(v) if np.isfinite(v) else None for k,v in scores.items()})
                        if row['value'] is None:
                            row['status']='undefined'
            except Exception as exc:
                row.update(status='error',value=None,error=str(exc)); failed=True
            rows.append(row); export()
            print(f'{name}/{method}: {row["status"]}',flush=True)
            if row.get('error'):
                print(f'  {row["error"]}',flush=True)
    print(f'Results: {folder/"results.csv"}')
    return int(failed)


if __name__=='__main__':
    raise SystemExit(main())
