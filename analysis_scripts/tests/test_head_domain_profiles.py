"""Focused checks for comparison weighting and conservative interval propagation."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_head_domain_profiles import Cell, bootstrap_cell, contrast_bounds, matched_cells


def cell(key, condition='C1', template='template_food_truck', rule='base_rule_ABA_exp_rule_ABB'):
    return Cell(key, condition, template, rule, '1', Path('unused'), '')


class ComparisonsTest(unittest.TestCase):
    def test_interval_subtraction_uses_opposite_endpoint(self):
        a, b = cell('a'), cell('b')
        a.mean, a.lower, a.upper = np.array([3.]), np.array([1.]), np.array([5.])
        b.mean, b.lower, b.upper = np.array([1.]), np.array([0.]), np.array([2.])
        mean, lo, hi = contrast_bounds({'a': a, 'b': b}, {'a': 1., 'b': -1.})
        np.testing.assert_allclose(mean, [2.])
        np.testing.assert_allclose(lo, [-1.])
        np.testing.assert_allclose(hi, [5.])

    def test_noop_cancellation_does_not_propagate_nan(self):
        a,b=cell('a'),cell('b')
        for c in [a,b]:
            c.mean,c.lower,c.upper=np.array([2.]),np.array([1.]),np.array([3.])
        a.lower[:]=np.nan
        _,lo,hi=contrast_bounds({'a':a,'b':b},{'a':0.,'b':1.})
        np.testing.assert_allclose(lo,[1.]);np.testing.assert_allclose(hi,[3.])

    def test_bootstrap_preserves_joint_head_variation(self):
        a=np.array([[1.,2.],[2.,4.],[4.,8.],[8.,16.]])
        r=bootstrap_cell(a,np.array([0,1]),np.random.default_rng(7),1000,.025)
        np.testing.assert_allclose(r['lower'][1],2*r['lower'][0])
        np.testing.assert_allclose(r['upper'][1],2*r['upper'][0])

    def test_zero_variance_uncertainty_is_unknown(self):
        a=np.array([[1.,2.],[1.,4.],[1.,8.]])
        r=bootstrap_cell(a,np.array([0,1]),np.random.default_rng(7),1000,.025)
        self.assertTrue(np.isnan(r['lower'][0]))
        self.assertTrue(np.isnan(r['point_lower'][0]))
        self.assertTrue(np.isfinite(r['lower'][1]))

    def test_controls_match_base_rule_not_full_rule(self):
        a=cell('a');b=cell('b','C5',rule='base_rule_ABA_exp_rule_ABA')
        c=cell('c','C5',rule='base_rule_ABB_exp_rule_ABB')
        self.assertEqual([(x.key,y.key) for x,y in matched_cells({'a':a,'b':b,'c':c},'C1','C5')],[('a','b')])

    def test_no_cross_story_template_pair(self):
        a=cell('a');b=cell('b','C2',template='template_library_book')
        self.assertEqual(matched_cells({'a':a,'b':b},'C1','C2'),[])


if __name__=='__main__':unittest.main()
