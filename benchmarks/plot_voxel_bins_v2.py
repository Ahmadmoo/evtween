"""Export offline interactive metric-vs-bin plots, one HTML per dataset/protocol.

Read-only: never computes metrics or changes source reports. Requires Plotly.
"""
import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.voxel_metric_v2 import polarity_aware_from_scores

METRICS=('voxel_mae','voxel_rmse','voxel_relative_l1','pmse_2_v2ce_adapted','pmse_4_v2ce_adapted')
CHANNELS={'value':'Signed','value_on':'ON','value_off':'OFF',
          'value_aware':'A · Polarity-aware','value_total':'B · Total ON+OFF'}


def _signature(config):
    """Do not mix evaluator versions, windows, crops or method selections."""
    value=dict(config)
    value.pop('dataset',None); value.pop('metric',None); value.pop('unit',None)
    options=dict(value.get('options',{})); options.pop('bin_ms',None)
    value['options']=options
    return json.dumps(value,sort_keys=True)


def load_reports(root, *, split='test', datasets=None, runs=None):
    groups={}
    for path in sorted(Path(root).glob('*/*/*/config.json')):
        config=json.loads(path.read_text(encoding='utf-8'))
        if (config.get('split')!=split or config.get('metric') not in METRICS or
                config.get('options',{}).get('protocol')!='fixed-duration-full-coverage-v2'):
            continue
        if datasets and config['dataset'] not in datasets:
            continue
        if runs and path.parent.name not in runs:
            continue
        file=path.with_name('results.csv')
        if not file.is_file():
            continue
        width=float(config['options']['bin_ms'])
        if not math.isfinite(width) or width<=0:
            raise ValueError(f'Invalid bin width: {path}')
        with file.open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        ids=[(r['sequence'],r['method']) for r in rows]
        if len(ids)!=len(set(ids)):
            raise ValueError(f'Duplicate sequence/method rows: {file}')
        key=(config['dataset'],_signature(config))
        reports=groups.setdefault(key,{})
        setting=(config['metric'],width)
        if setting in reports:
            raise ValueError(f'Ambiguous duplicate runs for {key[0]}/{setting}; select exact folders with --runs')
        reports[setting]={'config':config,'rows':rows,'file':str(file)}
    return groups


def matched_curves(reports, methods=None):
    """Fixed sequence/method cohort across all widths of each metric/channel."""
    curves=[]; coverage=[]
    for metric in METRICS:
        settings={width:item for (m,width),item in reports.items() if m==metric}
        if not settings:
            continue
        fields=('value','value_aware','value_total','value_on','value_off') if metric in ('voxel_mae','voxel_rmse') else ('value',)
        for field in fields:
            grids={}
            for width,item in settings.items():
                grid={}
                for row in item['rows']:
                    if row.get('status')!='ok' or row['method']=='ground_truth':
                        continue
                    if methods and row['method'] not in methods:
                        continue
                    try:
                        if field=='value_aware':
                            # Derive A per sequence, including older ON/OFF CSVs.
                            val=polarity_aware_from_scores(float(row['value_on']),float(row['value_off']),squared=metric=='voxel_rmse')
                        else:
                            val=float(row[field])
                        bounds=(float(row['start_s']),float(row['stop_s']),int(row['height']),int(row['width']))
                        count=int(row['temporal_bins']); tail=float(row['last_bin_ms'])
                    except (KeyError,TypeError,ValueError):
                        continue
                    if not all(math.isfinite(x) for x in (val,*bounds[:2],tail)) or count<1:
                        continue
                    if not math.isclose(float(row['bin_ms']),width,rel_tol=1e-12):
                        raise ValueError('CSV bin width disagrees with config')
                    grid.setdefault(row['method'],{})[row['sequence']] = (val,bounds,count,tail)
                grids[width]=grid
            present=[set(g) for g in grids.values()]
            common_methods=set.intersection(*present) if present else set()
            observed=set.union(*present) if present else set()
            seq_sets=[set(g[method]) for g in grids.values() for method in common_methods]
            sequences=set.intersection(*seq_sets) if seq_sets else set()
            # A shared sequence must also have identical window and geometry everywhere.
            sequences={s for s in sequences if len({g[m][s][1] for g in grids.values() for m in common_methods})==1}
            coverage.append(dict(metric=metric,channel=field,widths_ms=sorted(settings),
                methods=sorted(common_methods),excluded_methods=sorted((set(methods) if methods else observed)-common_methods),
                sequences=sorted(sequences),n=len(sequences)))
            if not sequences:
                continue
            for method in sorted(common_methods):
                points=[]
                for width,g in sorted(grids.items()):
                    items=[g[method][s] for s in sorted(sequences)]
                    counts=[x[2] for x in items]
                    points.append(dict(bin_ms=width,error=statistics.mean(x[0] for x in items),
                        mean_bins=statistics.mean(counts),min_bins=min(counts),max_bins=max(counts),
                        n=len(sequences),mean_duration_s=statistics.mean(x[1][1]-x[1][0] for x in items)))
                channel=('ON + OFF' if metric.startswith('pmse_') else 'Signed') if field=='value' else CHANNELS[field]
                curves.append(dict(metric=metric,channel=channel,method=method,points=points))
    return curves,coverage


def make_figure(dataset, reports, methods=None):
    """Build an interactive figure without creating files; usable in Jupyter."""
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError as exc:
        raise RuntimeError('Install the plotting dependency: python -m pip install plotly') from exc
    curves,coverage=matched_curves(reports,methods)
    if not curves:
        raise ValueError(f'No matched completed cohorts for {dataset}')
    choices=list(dict.fromkeys((c['metric'],c['channel']) for c in curves))
    model_names=sorted({c['method'] for c in curves})
    palette=['#2563eb','#d97706','#059669','#dc2626','#7c3aed','#0891b2','#be185d','#4d7c0f','#475569','#9333ea','#c2410c','#0f766e']
    fig=go.Figure()
    for c in curves:
        points=c['points']; selected=(c['metric'],c['channel'])==choices[0]
        fig.add_trace(go.Scatter(x=[p['bin_ms'] for p in points],y=[p['error'] for p in points],
            name=c['method'],legendgroup=c['method'],mode='lines+markers',visible=selected,
            line={'color':palette[model_names.index(c['method'])%len(palette)]},
            customdata=[[p['bin_ms'],p['mean_bins'],p['min_bins'],p['max_bins'],p['n'],p['mean_duration_s']] for p in points],
            hovertemplate=('%{fullData.name}<br>Error: %{y:.6g}<br>Nominal bin: %{customdata[0]:g} ms'
                '<br>Mean bin count: %{customdata[1]:.2f} (%{customdata[2]}–%{customdata[3]})'
                '<br>Matched sequences: %{customdata[4]}<br>Mean duration: %{customdata[5]:.4g} s<extra></extra>')))
    units=lambda metric: 'Squared events/voxel ↓' if metric.startswith('pmse_') else ('Relative L1 ratio ↓' if metric=='voxel_relative_l1' else 'Events/voxel ↓')
    buttons=[]
    for metric,channel in choices:
        buttons.append(dict(label=f'{metric} · {channel}',method='update',args=[
            {'visible':[(c['metric'],c['channel'])==(metric,channel) for c in curves]},
            {'title.text':f'{escape(dataset)} · {metric} · {channel}', 'yaxis.title.text':units(metric)}]))
    width_x=[[p['bin_ms'] for p in c['points']] for c in curves]
    count_x=[[p['mean_bins'] for p in c['points']] for c in curves]
    fig.update_layout(template='plotly_white',height=620,
        title=f'{escape(dataset)} · {choices[0][0]} · {choices[0][1]}',
        xaxis_title='Nominal temporal-bin width (ms)',yaxis_title=units(choices[0][0]),
        margin=dict(t=150,b=65),hovermode='closest',
        updatemenus=[dict(buttons=buttons,x=0,y=1.25,xanchor='left',yanchor='top'),
            dict(type='buttons',direction='right',x=0,y=1.13,xanchor='left',yanchor='top',buttons=[
                dict(label='Bin width (ms)',method='update',args=[{'x':width_x},{'xaxis.title.text':'Nominal temporal-bin width (ms)'}]),
                dict(label='Mean bin count',method='update',args=[{'x':count_x},{'xaxis.title.text':'Mean temporal-bin count across matched sequences'}])]),
            dict(type='buttons',direction='right',x=1,y=1.13,xanchor='right',yanchor='top',buttons=[
                dict(label='Linear Y',method='relayout',args=[{'yaxis.type':'linear'}]),
                dict(label='Log Y',method='relayout',args=[{'yaxis.type':'log'}])])])
    fig.add_hline(y=0,line_color='#64748b',line_dash='dot')
    return fig,curves,coverage


def export_html(dataset, signature, reports, output, methods=None):
    from plotly.io import to_html
    fig,curves,coverage=make_figure(dataset,reports,methods)
    chart=to_html(fig,full_html=False,include_plotlyjs=True,config={'responsive':True,'displaylogo':False})
    # Untrusted report names are escaped; no external scripts, CDN or raw report JSON injection.
    details=''.join(f'<li>{escape(c["metric"])} / {escape(c["channel"])}: n={c["n"]}; '
                    f'widths={escape(str(c["widths_ms"]))} ms; excluded models={escape(str(c["excluded_methods"]))}</li>' for c in coverage)
    sources=''.join(f'<li>{escape(item["file"])}</li>' for item in reports.values())
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    doc+='<title>'+escape(dataset)+' voxel bin comparison</title><body style="font-family:system-ui;margin:24px;background:white;color:#172033">'
    doc+='<h1>'+escape(dataset)+' — temporal resolution comparison</h1>'
    doc+='<p>Read-only saved v2 scores. Select a metric/polarity, click legend names to toggle models, zoom or hover. Each curve averages the same matched sequence cohort across all available bin widths for that metric/channel. Zero error means a perfect match under that representation (relative L1 requires nonzero GT mass). Log Y hides zero values.</p>'
    doc+='<p><strong>Interpretation:</strong> raw values of different metrics have different scales. Coarser bins and PMSE pooling can hide temporal errors. A smaller number across bin settings does not by itself prove a more faithful simulator or a better metric. Mean bin count is dataset-dependent, not a fixed per-sequence count. No raw events are re-evaluated here.</p>'
    doc+=chart+'<details><summary>Matched coverage</summary><ul>'+details+'</ul></details><details><summary>Source reports and protocol</summary><ul>'+sources+'</ul><pre style="white-space:pre-wrap">'+escape(signature)+'</pre></details></body></html>'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(doc,encoding='utf-8')
    return curves,coverage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-root',type=Path,default=Path(__file__).resolve().parents[1]/'reports_voxel_v2')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'reports_voxel_v2/html')
    p.add_argument('--datasets',nargs='+');p.add_argument('--methods',nargs='+')
    p.add_argument('--runs',nargs='+',help='Optional exact report folder names to resolve duplicate runs')
    p.add_argument('--split',choices=('train','test'),default='test')
    args=p.parse_args()
    groups=load_reports(args.report_root,split=args.split,datasets=args.datasets,runs=args.runs)
    if not groups:p.error('No matching v2 reports found; calculate scores at the desired bin widths first')
    failures=0
    for (dataset,signature),reports in groups.items():
        suffix=hashlib.sha256(signature.encode()).hexdigest()[:10]
        # Dataset labels never become paths directly.
        safe=''.join(c if c.isascii() and (c.isalnum() or c in '-_') else '_' for c in dataset)
        output=args.output_dir/f'{safe}-{args.split}-{suffix}.html'
        try:
            curves,_=export_html(dataset,signature,reports,output,args.methods)
            print(f'{dataset}: {output} ({len(curves)} curves)')
        except (ValueError,RuntimeError) as exc:
            print(f'{dataset}: {exc}');failures+=1
    return int(failures>0)


if __name__=='__main__':raise SystemExit(main())
