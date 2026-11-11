import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import csv
import numpy as np
from benchmarks import voxel_metric_v2 as v
from benchmarks.test_voxel_metric import events


class FixedDurationMetrics(unittest.TestCase):
    def test_polarity_aware_and_total(self):
        def stream(on,off):
            return events([0.]*(on+off),p=[1]*on+[-1]*off)
        domain=dict(edges=np.array([0.,.03]),height=1,width=1,bin_ms=30)
        for metric in ('voxel_mae','voxel_rmse'):
            row=v.score(metric,stream(12,7),stream(10,5),**domain)
            self.assertEqual(row['value'],0.)
            self.assertAlmostEqual(row['value_aware'],2.)
            self.assertEqual(row['value_total'],4.)
            swap=v.score(metric,stream(0,10),stream(10,0),**domain)
            self.assertEqual(swap['value'],20.)
            self.assertAlmostEqual(swap['value_aware'],10.)
            self.assertEqual(swap['value_total'],0.)
        row=v.score('voxel_rmse',stream(13,9),stream(10,5),**domain)
        self.assertAlmostEqual(row['value_aware'],np.sqrt(12.5))
        self.assertEqual(row['value_total'],7.)

    def test_scoring_window_origin_contract(self):
        self.assertEqual(v.scoring_window([0,.5,1],100,.1,.8),(.1,.8))
        self.assertEqual(v.scoring_window([0,1],100), (0.,1.))
        for edges in ([.5,1.5],[0,0],[0,np.nan],[],[0,1,.5]):
            with self.assertRaises(ValueError): v.scoring_window(edges,100)
        with self.assertRaises(ValueError): v.scoring_window([0,1],100,0,2)

    def test_edges(self):
        np.testing.assert_allclose(v.time_edges(0,.1,30),[0,.03,.06,.09,.1])
        self.assertEqual(len(v.time_edges(0,.09,30)),4)
        np.testing.assert_allclose(v.time_edges(1,1.1,30),[1,1.03,1.06,1.09,1.1])
        for args in ((0,1,0),(1,0,30),(0,np.nan,30)):
            with self.assertRaises(ValueError): v.time_edges(*args)

    def test_forward_mass_and_boundaries(self):
        edges=v.time_edges(0,.1,30)
        ev=events([-.01,0,.015,.09,.099,.1],p=[1,1,1,-1,-1,1])
        grid=v.event_voxels(ev,edges,2,2,interpolation='forward',bin_ms=30)
        self.assertAlmostEqual(grid.sum(),4)
        np.testing.assert_allclose(grid[0,:,0,0],[1.5,.5,0,0])
        np.testing.assert_allclose(grid[1,:,0,0],[0,0,0,2])

    def test_last_corner_and_small_dimensions(self):
        edges=v.time_edges(0,.1,30)
        domain=dict(edges=edges,height=5,width=7,bin_ms=30)
        corner=events([.099],x=[6],y=[4])
        self.assertGreater(v.pmse_2_v2ce_adapted(corner,events(),**domain),0)
        self.assertGreater(v.pmse_4_v2ce_adapted(corner,events(),**domain),0)
        self.assertEqual(v.pmse_4_v2ce_adapted(corner,corner,**domain),0)
        self.assertEqual(v.pmse_4_v2ce_adapted(events([0]),events(),
            edges=np.array([0,.01]),height=1,width=1,bin_ms=30),.5)

    def test_weighted_partial_pool(self):
        a=np.arange(2*3*5*7).reshape(2,3,5,7).astype(float)
        pooled,weights=v._pool_full(a,2)
        self.assertEqual(weights.sum(),3*5*7)
        self.assertAlmostEqual(float((pooled*weights).sum()),float(a.sum()))
        pred=events([0]); empty=events()
        domain=dict(edges=v.time_edges(0,.12,30),height=4,width=4,bin_ms=30)
        self.assertEqual(v.pmse_2_v2ce_adapted(pred,empty,**domain),1/1024)

    def test_histogram_scores_and_validation(self):
        domain=dict(edges=v.time_edges(0,.1,30),height=2,width=2,bin_ms=30)
        on=events([.099]); off=events([.099],p=[0])
        self.assertEqual(v.score('voxel_mae',on,off,**domain)['value'],2/16)
        self.assertEqual(v.score('voxel_rmse',on,off,**domain)['value'],.5)
        self.assertTrue(np.isnan(v.score('voxel_relative_l1',on,events(),**domain)['value']))
        with self.assertRaises(ValueError):
            v.event_voxels(events([0],x=[2]),domain['edges'],2,2)

    def test_cli(self):
        from PIL import Image
        from benchmarks import run_metric as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'data/toy/test/seq'
            (source/'frames').mkdir(parents=True)
            for i in range(2): Image.new('L',(7,5)).save(source/'frames'/f'{i}.png')
            np.save(source/'frame_ts.npy',[10.,10.1])
            for k,a in events([10.099],x=[6],y=[4]).items(): np.save(source/f'ev_{k}.npy',a)
            _,_,_,meta,_=old.v2.sequence_info(source)
            dest=root/'pred/toy/test/v2ce/seq';dest.mkdir(parents=True)
            np.savez(dest/'events.npz',**events())
            (dest/'input.json').write_text(json.dumps(meta))
            script=Path(__file__).with_name('run_voxel_metric_v2.py')
            for metric in ('voxel_mae','voxel_rmse','voxel_relative_l1','pmse_2_v2ce_adapted','pmse_4_v2ce_adapted'):
                cmd=[sys.executable,str(script),'toy','test','--metric',metric,'--voxel-bin-ms','30',
                     '--methods','v2ce','v2v','--data-root',str(root/'data'),
                     '--pred-root',str(root/'pred'),'--report-root',str(root/'reports')]
                result=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                report=next((root/'reports/toy'/metric).glob('*/results.csv'))
                with report.open() as stream: rows=list(csv.DictReader(stream))
                self.assertEqual(rows[0]['status'],'ok')
                self.assertEqual(int(rows[0]['temporal_bins']),4)
                self.assertAlmostEqual(float(rows[0]['last_bin_ms']),10)
                self.assertGreater(float(rows[0]['value']),0)
                self.assertEqual(rows[1]['status'],'not_applicable')
                if metric in ('voxel_mae','voxel_rmse'):
                    self.assertTrue(rows[0]['value_aware'])
                    self.assertTrue(rows[0]['value_total'])
            all_run=subprocess.run([sys.executable,str(script),'toy','test','--metric','all',
                '--voxel-bin-ms','30','--methods','v2ce','--data-root',str(root/'data'),
                '--pred-root',str(root/'pred'),'--report-root',str(root/'all_reports')],
                capture_output=True,text=True)
            self.assertEqual(all_run.returncode,0,all_run.stdout+all_run.stderr)
            self.assertEqual(len(list((root/'all_reports').glob('*/*/*/results.csv'))),5)
            # A custom window keeps the absolute shared origin; no per-stream stretching.
            cmd=[sys.executable,str(script),'toy','test','--metric','voxel_mae','--voxel-bin-ms','30',
                 '--voxel-start-s','.09','--voxel-stop-s','.1','--methods','v2ce',
                 '--data-root',str(root/'data'),'--pred-root',str(root/'pred'),
                 '--report-root',str(root/'window_reports')]
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=next((root/'window_reports/toy/voxel_mae').glob('*/results.csv'))
            with report.open() as stream: row=next(csv.DictReader(stream))
            self.assertEqual(int(row['temporal_bins']),1)
            self.assertAlmostEqual(float(row['value']),1/35)


if __name__=='__main__': unittest.main()
