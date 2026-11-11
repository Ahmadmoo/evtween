"""Read-only raw event-rate views for the v2 watchboard (not error metrics)."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import run_metric as runner
from .voxel_metric_v2 import time_edges


def rate_arrays(events, start, stop, bin_ms):
    """Counts / actual bin seconds. Include start, exclude stop; no sampling."""
    if not np.isfinite([start,stop,bin_ms]).all() or start>=stop or bin_ms<=0:
        raise ValueError('Require finite start < stop and positive bin_ms')
    if np.ceil((stop-start)*1000/bin_ms)>100000:
        raise ValueError('Rate view exceeds 100000 bins; increase rate-bin width or shorten the window')
    edges=time_edges(start,stop,bin_ms)
    t=np.asarray(events['t']);p=np.asarray(events['p'])
    if (t.ndim!=1 or p.shape!=t.shape or t.dtype.kind not in 'iuf' or
        not np.isfinite(t).all() or not np.isin(p,[-1,0,1]).all()):
        raise ValueError('Require matching finite timestamps and valid polarity')
    if np.any(t[1:]<t[:-1]):
        order=np.argsort(t,kind='stable');t=t[order];p=p[order]
    on_t=t[p>0];off_t=t[p<=0]
    on=np.diff(np.searchsorted(on_t,edges,side='left'))
    off=np.diff(np.searchsorted(off_t,edges,side='left'))
    widths=np.diff(edges)
    return dict(edges=edges,on=on/widths,off=off/widths,total=(on+off)/widths,
                on_counts=on,off_counts=off,total_counts=on+off)


def load_streams(dataset, split, pred_root, data_root=None, sequence='', methods=None):
    pred=Path(pred_root)/dataset/split
    folders=sorted(p for p in pred.glob('*') if p.is_dir())
    if not sequence:
        names=sorted({p.name for d in folders for p in d.iterdir()
                      if p.is_dir() and (p/'events.npz').is_file()})
        if not names:raise ValueError('No saved event streams found; set dataset/paths/sequence')
        sequence=names[0]
    root=Path(data_root) if data_root is not None else runner.v2.old.setting('EVTWEEN_DATA',runner.v2.old.LAB/'evtween/data')
    args=SimpleNamespace(dataset=dataset,split=split,data_root=data_root)
    source=runner.source_for(sequence,root/dataset/split,folders,args)
    seq=runner.Sequence(source);frames,meta=seq.geometry
    streams={'Ground truth':seq.gt};notes=[]
    requested=set(methods) if methods is not None else None
    for folder in folders:
        if requested is not None and folder.name not in requested:continue
        if folder.name=='v2v':
            notes.append('Native V2V omitted: signed voxel arrays cannot recover event counts/timestamps.');continue
        file=folder/sequence/'events.npz'
        if not file.is_file():
            notes.append(f'{folder.name}: missing events.npz');continue
        try:
            runner.verify_metadata(folder/sequence/'input.json',meta,args,sequence)
            streams[folder.name]=runner.read_events(file,meta)
        except Exception as exc:
            notes.append(f'{folder.name}: omitted ({exc})')
    if requested is not None:
        absent=requested-{f.name for f in folders}
        notes.extend(f'{m}: prediction folder unavailable' for m in sorted(absent))
    return sequence,frames,meta,streams,notes


def make_rate_figures(streams, start, stop, bin_ms, title=''):
    """Two interactive figures: separate ON/OFF, then total ON+OFF."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from html import escape
    rates={name:rate_arrays(ev,start,stop,bin_ms) for name,ev in streams.items()}
    separate=make_subplots(rows=2,cols=1,shared_xaxes=True,shared_yaxes='all',
                           subplot_titles=('ON events/s','OFF events/s'),vertical_spacing=.12)
    total=go.Figure()
    palette=['#2563eb','#d97706','#059669','#dc2626','#7c3aed','#0891b2','#be185d','#4d7c0f','#475569','#9333ea','#c2410c']
    for i,(name,rates_i) in enumerate(rates.items()):
        color='#111111' if name=='Ground truth' else palette[(i-1)%len(palette)]
        edges=rates_i['edges'];x=edges-start
        hover=np.column_stack([np.r_[edges[:-1]-start,edges[-2]-start],
                               np.r_[edges[1:]-start,edges[-1]-start],
                               np.r_[np.diff(edges)*1000,np.diff(edges)[-1]*1000]])
        def trace(key,showlegend):
            return go.Scatter(x=x,y=np.r_[rates_i[key],rates_i[key][-1]],mode='lines',
                name=name,legendgroup=name,showlegend=showlegend,
                line=dict(color=color,width=2.4 if name=='Ground truth' else 1.4,shape='hv'),
                customdata=hover,hovertemplate=('%{fullData.name}<br>[%{customdata[0]:.6f}, %{customdata[1]:.6f}) s'
                '<br>Width: %{customdata[2]:.4g} ms<br>Rate: %{y:,.4g} events/s<extra></extra>'))
        separate.add_trace(trace('on',True),row=1,col=1)
        separate.add_trace(trace('off',False),row=2,col=1)
        total.add_trace(trace('total',True))
    for fig,label,height in ((separate,'ON and OFF',850),(total,'Total ON + OFF',480)):
        fig.update_layout(template='plotly_white',height=height,title=f'{escape(title)} · {label}',
            hovermode='x unified',legend=dict(orientation='h',y=1.13,groupclick='togglegroup'),
            margin=dict(t=150,b=65),dragmode='zoom',
            updatemenus=[dict(type='buttons',direction='right',x=1,y=1.23,xanchor='right',buttons=[
                dict(label='Reset zoom',method='relayout',args=[{'xaxis.autorange':True,'xaxis2.autorange':True,
                                                              'yaxis.autorange':True,'yaxis2.autorange':True}]),
                dict(label='Linear Y',method='relayout',args=[{'yaxis.type':'linear','yaxis2.type':'linear'}]),
                dict(label='Log Y',method='relayout',args=[{'yaxis.type':'log','yaxis2.type':'log'}])])])
        fig.update_xaxes(title_text='Time since selected evaluation start (s)')
        fig.update_yaxes(title_text='Events/s',rangemode='tozero')
    return separate,total,rates
