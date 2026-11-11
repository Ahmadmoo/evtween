import importlib.util
import unittest
import numpy as np
from benchmarks import event_rate_views_v2 as r


class RawRates(unittest.TestCase):
    def test_half_open_polarity_and_short_final_bin(self):
        events=dict(t=np.array([-.1,0,.015,.03,.099,.1]),p=np.array([1,1,0,1,-1,1]))
        result=r.rate_arrays(events,0,.1,30)
        np.testing.assert_array_equal(result['on_counts'],[1,1,0,0])
        np.testing.assert_array_equal(result['off_counts'],[1,0,0,1])
        np.testing.assert_allclose(result['total'],result['on']+result['off'])
        self.assertAlmostEqual(result['off'][-1],100.)
        self.assertEqual(result['total_counts'].sum(),4)
        self.assertAlmostEqual(float(np.sum(result['total']*np.diff(result['edges']))),4)

    def test_shifted_unsorted_and_empty(self):
        events=dict(t=np.array([1.09,1.,1.03]),p=np.array([-1,1,1]))
        result=r.rate_arrays(events,1,1.1,30)
        self.assertEqual(result['total_counts'].sum(),3)
        empty=r.rate_arrays(dict(t=np.array([]),p=np.array([])),0,.1,30)
        self.assertEqual(empty['total'].sum(),0)
        for width in (0,np.nan,.0000001):
            with self.assertRaises(ValueError):r.rate_arrays(events,0,1,width)

    @unittest.skipUnless(importlib.util.find_spec('plotly'),'Optional Plotly dependency missing')
    def test_two_figures_ground_truth_and_total(self):
        events=dict(t=np.array([0,.01]),p=np.array([1,-1]))
        onoff,total,data=r.make_rate_figures({'Ground truth':events,'model':events},0,.1,30,'Toy')
        self.assertEqual(len(onoff.data),4)
        self.assertEqual(len(total.data),2)
        self.assertEqual(total.data[0].name,'Ground truth')
        self.assertEqual(total.data[0].line.color,'#111111')
        np.testing.assert_allclose(total.data[0].y[:-1],data['Ground truth']['total'])
        self.assertEqual(onoff.data[0].legendgroup,onoff.data[1].legendgroup)
        self.assertEqual(onoff.layout.yaxis.matches,'y2')


if __name__=='__main__':unittest.main()
