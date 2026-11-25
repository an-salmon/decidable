from fractions import Fraction
import unittest

from decidable.size_calculus.stretched import (
    AverageProblem, bracket_average_scale, search_average_refutation,
    verify_average_refutation,
)


class StretchedSearchTests(unittest.TestCase):
    def test_search_finds_and_replays_a_conservative_two_atom_witness(self):
        problem = AverageProblem(Fraction(1, 2), (1, 1), (Fraction(1, 2),) * 2, 1)
        grid = ((0, 0), (1, 0), (0, 1), (0, 1), (0, 1000))
        result = search_average_refutation(problem, grid)
        self.assertEqual(result.status, "disproved")
        points, masses = result.witness
        self.assertEqual(set(points), {(1, 0), (0, 1)})
        self.assertEqual(sum(masses), 1)
        self.assertLessEqual(len(masses), 3)
        self.assertEqual(verify_average_refutation(problem, points, masses).status, "disproved")
        self.assertEqual(search_average_refutation(problem, grid, root_steps=0).status,
                         "inconclusive")

    def test_insufficient_grid_gives_no_universal_upper_bound(self):
        problem = AverageProblem(1, (1, 1), (Fraction(1, 2),) * 2, 1)
        result = search_average_refutation(problem, ((0, 0), (1, 0), (0, 1)))
        self.assertEqual(result.status, "inconclusive")
        self.assertIn("grid", result.scope)
        with self.assertRaises(ValueError):
            search_average_refutation(problem, ((1,),))

    def test_bracket_keeps_valid_endpoints_when_midpoint_is_unresolved(self):
        problem = AverageProblem(1, (1,), (1,), Fraction(1, 2))
        grid = ((Fraction(2, 3),),)
        result = bracket_average_scale(problem, grid, trials=6)
        self.assertEqual(result.upper, 1)
        self.assertGreater(result.lower, Fraction(15, 16))
        self.assertLess(result.lower, result.upper)
        self.assertEqual(result.lower_receipt.status, "disproved")
        self.assertEqual(result.upper_receipt.status, "verified")
        self.assertIsNone(bracket_average_scale(problem, ((0,),)))

    def test_nonconvex_aggregation_gives_an_exact_upper_enclosure(self):
        problem = AverageProblem(Fraction(1, 2), (1, 1), (Fraction(1, 2),) * 2, 1)
        grid = ((1, 0), (0, 1))
        result = bracket_average_scale(problem, grid, trials=0)
        self.assertEqual(result.lower, 1)
        self.assertEqual(result.upper, Fraction(25, 4))
        self.assertIsNone(bracket_average_scale(problem, grid, upper_scale=2, trials=0))
