import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from benchmarks import plot_voxel_bins_v2 as p


def fixture(root):
    for width in (10,30):
        folder=root/'toy/voxel_mae'/f'test-v2-{width}';folder.mkdir(parents=True)
        config=dict(dataset='toy',split='test',metric='voxel_mae',unit='events/voxel',
                    options=dict(protocol='fixed-duration-full-coverage-v2',bin_ms=width,start_s=0,stop_s=None),
                    code_sha256={'runner':'test-version'})
        (folder/'config.json').write_text(json.dumps(config))
        rows=[]
        for method in ('a','b'):
            for sequence,duration in (('s1',.1),('s2',.2)):
                error=(1 if method=='a' else 2)*width/10
                rows.append(dict(method=method,sequence=sequence,status='ok',value=error,
                    value_on=error/2,value_off=error/3,start_s=0,stop_s=duration,height=5,width=7,
                    temporal_bins=__import__('math').ceil(duration*1000/width),last_bin_ms=10,bin_ms=width))
        rows.append(dict(rows[0],method='failed',status='error',value=0))
        with (folder/'results.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


class BinPlots(unittest.TestCase):
    def test_matching_and_bin_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp));groups=p.load_reports(tmp)
            self.assertEqual(len(groups),1)
            reports=next(iter(groups.values()))
            curves,coverage=p.matched_curves(reports)
            self.assertEqual(len(curves),6)
            self.assertTrue(all(c['n']==2 for c in coverage))
            c=next(c for c in curves if c['method']=='a' and c['channel']=='Signed')
            self.assertEqual([v['error'] for v in c['points']],[1,3])
            self.assertEqual([v['mean_bins'] for v in c['points']],[15,5.5])
            reports[('voxel_mae',30.)]['rows']=[r for r in reports[('voxel_mae',30.)]['rows'] if r['method']!='b']
            curves,coverage=p.matched_curves(reports)
            self.assertTrue(all(c['method']=='a' for c in curves))
            self.assertTrue(all('b' in c['excluded_methods'] for c in coverage))

    def test_different_windows_and_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);reports=next(iter(p.load_reports(root).values()))
            for row in reports[('voxel_mae',30.)]['rows']:
                if row['sequence']=='s2':row['stop_s']=.19
            _,coverage=p.matched_curves(reports)
            self.assertTrue(all(c['n']==1 for c in coverage))
            path=root/'toy/voxel_mae/test-v2-30/config.json'
            config=json.loads(path.read_text());config['code_sha256']['runner']='new'
            path.write_text(json.dumps(config))
            self.assertEqual(len(p.load_reports(root)),2)

    @unittest.skipUnless(importlib.util.find_spec('plotly'),'Plotly optional dependency not installed')
    def test_standalone_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            (dataset,sig),reports=next(iter(p.load_reports(root).items()))
            output=root/'plot.html';p.export_html(dataset,sig,reports,output)
            html=output.read_text(encoding='utf-8')
            self.assertIn('plotly.js',html)
            self.assertIn('Mean bin count',html)
            self.assertIn('voxel_mae',html)
            self.assertNotIn('<script src=',html)
            result=subprocess.run([sys.executable,str(Path(p.__file__)),
                '--report-root',str(root),'--output-dir',str(root/'html'),'--datasets','toy'],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(len(list((root/'html').glob('toy-test-*.html'))),1)


if __name__=='__main__':unittest.main()
