import unittest
from fractions import Fraction

from decidable.size_calculus.scales import (
    ScaleMonomial, ScaleShare, rewrite_scale, verify_scale_domination,
)


class ScaleComparisonTests(unittest.TestCase):
    def test_time_rewrite_retains_factor_and_free_atom(self):
        original = ScaleMonomial(1, {"K": 1, "t": -1})
        # Encode t = T/2 as an exact premise of the rewrite.
        rewritten = rewrite_scale(original, "t", ScaleMonomial(Fraction(1, 2), {"T": 1}))
        exact = ScaleMonomial(2, {"K": 1, "T": -1})
        self.assertEqual(rewritten, exact)
        ranges = {"K": "free", "T": "free"}
        self.assertEqual(verify_scale_domination((rewritten,), (exact,), ranges,
                                                (ScaleShare(0, 0, 2),)).status, "verified")
        self.assertEqual(verify_scale_domination(
            (rewritten,), (ScaleMonomial(1, exact.powers),), ranges, (ScaleShare(0, 0, 2),)
        ).status, "inconclusive")

    def test_two_terms_cannot_spend_the_same_exact_share(self):
        term = ScaleMonomial(1, {"r": 1})
        shares = ScaleShare(0, 0, 1), ScaleShare(1, 0, 1)
        self.assertEqual(verify_scale_domination((term, term), (term,), {"r": "small"},
                                                shares).status, "inconclusive")
        target = ScaleMonomial(2, {"r": 1})
        self.assertEqual(verify_scale_domination((term, term), (target,), {"r": "small"},
                                                shares).status, "verified")

    def test_small_and_large_ranges_and_generic_dependencies(self):
        derived = ScaleMonomial(3, {"r": 2, "R": 1}, frozenset({"d"}))
        target = ScaleMonomial(1, {"r": 1, "R": 2}, frozenset({"d", "s"}))
        ranges = {"r": "small", "R": "large"}
        self.assertEqual(verify_scale_domination((derived, derived), (target,), ranges,
                                                (ScaleShare(0, 0), ScaleShare(1, 0))).status,
                         "verified")
        for bad in (ScaleMonomial(4, target.powers),
                    ScaleMonomial(4, target.powers, frozenset({"s"}))):
            share = ScaleShare(0, 0, 4 if bad.dependencies is None else None)
            self.assertEqual(verify_scale_domination((derived,), (bad,), ranges, (share,)).status,
                             "inconclusive")
        with self.assertRaises(ValueError):
            verify_scale_domination((derived,), (ScaleMonomial(1, target.powers, frozenset({"r"})),),
                                    ranges, (ScaleShare(0, 0),))

    def test_rewrite_cannot_hide_a_varying_generic_dependency(self):
        dependent = ScaleMonomial(1, {"K": 1}, frozenset({"t"}))
        with self.assertRaisesRegex(ValueError, "generic constant dependencies"):
            rewrite_scale(dependent, "t", ScaleMonomial(Fraction(1, 2), {"T": 1}))
