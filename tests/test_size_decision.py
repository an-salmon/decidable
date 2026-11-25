"""Exact decision fixtures outside the scope of rational moment search."""

from fractions import Fraction as F
import unittest
from unittest.mock import patch

from sympy import Eq, Poly, Rational, symbols

from decidable.size_calculus import (
    MomentProblem, SemialgebraicMomentProblem, decide_moment, exponential_moment_problem,
    search_refutation,
)


class ExactMomentDecisionTests(unittest.TestCase):
    def setUp(self):
        available = SemialgebraicMomentProblem((symbols("x"),), True, (), (), 1, 1)
        if decide_moment(available, timeout_ms=2000).status == "unsupported":
            self.skipTest("the optional system Z3 backend is unavailable")

    def test_algebraic_refutation_at_a_boundary_budget(self):
        x = symbols("x")
        problem = MomentProblem((Poly((x**2 - 2)**2, x),), (0,), Poly(x, x), 1, lower=1, upper=2)
        self.assertEqual(search_refutation(problem, (1, F(3, 2), 2)).status, "inconclusive")
        result = decide_moment(problem, timeout_ms=5000)
        self.assertEqual(result.status, "disproved")
        self.assertIn("root-obj", result.witness.assignment)

    def test_affordable_majorant_may_require_an_algebraic_slope(self):
        x = symbols("x")
        g = Poly(x**2, x)
        f = Poly(2 + x * (x**2 - 2) - (x**2 - 2)**2, x)
        problem = MomentProblem((g,), (2,), f, 2, lower=1, upper=2)
        self.assertEqual(decide_moment(problem, timeout_ms=5000).status, "verified")

    def test_boolean_support_and_negative_integrand_guard(self):
        x, y = symbols("x y")
        support = (Eq(x, 1) & Eq(y, 0)) | (Eq(x, 0) & Eq(y, 1))
        problem = SemialgebraicMomentProblem((x, y), support, (), (), x + y, 1)
        self.assertEqual(decide_moment(problem, timeout_ms=5000).status, "verified")
        invalid = SemialgebraicMomentProblem((x, y), support, (), (), x - y, 1)
        with self.assertRaisesRegex(ValueError, "negative"):
            decide_moment(invalid, timeout_ms=5000)

    def test_interval_translation_ignores_symbol_assumptions(self):
        x = symbols("x", positive=True)
        problem = MomentProblem((), (), Poly(-x, x), 1, lower=-2, upper=-1)
        self.assertEqual(decide_moment(problem, timeout_ms=5000).status, "disproved")

    def test_common_exponential_chart_including_negative_rates(self):
        bounded = exponential_moment_problem((((1, 2),),), (4,), ((1, 1),), 2)
        self.assertEqual(decide_moment(bounded, timeout_ms=5000).status, "verified")
        decaying = exponential_moment_problem((), (), ((1, -F(1, 2)),), F(9, 10))
        self.assertEqual(decide_moment(decaying, timeout_ms=5000).status, "disproved")


def float_or_nonreal_fixtures():
    x = symbols("x")
    return (
        ((x,), True, (), (), x + 1.0, 1),
        ((symbols("x", positive=True),), True, (), (), 1, 1),
        ((x,), Eq(x, Rational(1, 2)), (), (), 1, 1.0),
    )


class DecisionBoundaryTests(unittest.TestCase):
    def test_rejects_nonexact_data_and_implicit_domain(self):
        for arguments in float_or_nonreal_fixtures():
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                SemialgebraicMomentProblem(*arguments)

    def test_backend_unavailable_and_unknown_are_not_a_proof(self):
        problem = SemialgebraicMomentProblem((symbols("x"),), True, (), (), 1, 1)
        with patch("decidable.size_calculus._nra.find_library", return_value=None):
            self.assertEqual(decide_moment(problem).status, "unsupported")
        with patch("decidable.size_calculus.semialgebraic.solve", return_value=("unknown", None)):
            self.assertEqual(decide_moment(problem).status, "inconclusive")
