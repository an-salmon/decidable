import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sympy import Poly, Rational, symbols

from decidable import exp_series, search
from decidable.kernel_domination import check_kernel_domination
from decidable.report import Obligation, Report


class ReviewRegressionTests(unittest.TestCase):
    def test_exp_tail_reports_the_early_negative_coefficient(self):
        x = symbols("x")
        evaluate = exp_series._forward_differences

        def guard(terms, cutoff, size):
            self.assertLessEqual(cutoff, 4)
            return evaluate(terms, cutoff, size)

        with patch.object(exp_series, "_forward_differences", side_effect=guard):
            result = exp_series.check_exp_series(
                Poly(x**2 / 10**12 - x, x), Poly(0, x), tail_start="auto"
            )
        self.assertEqual((result.status, result.witness), ("inconclusive", 1))

    def test_kernel_returns_early_disproof_without_growing_rational_powers(self):
        z = symbols("z")
        ratio = 1 - Rational(1, 10**9)
        upper = (Poly(1, z), Poly(1 - z, z))
        lower = (Poly(2, z), Poly(1 - ratio * z, z))
        rational_type = type(ratio)
        original_power = rational_type.__pow__

        def guard(base, exponent):
            if base == ratio:
                self.assertLessEqual(exponent, 4)
            return original_power(base, exponent)

        with patch.object(rational_type, "__pow__", guard):
            result = check_kernel_domination(upper, lower, tail_start=None)
        self.assertEqual(result.status, "disproved")
        self.assertEqual(result.witness, {"index": 0, "coefficient": -1})

    def test_report_keeps_numerical_evidence_out_of_exact_passes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            results = Report([
                Obligation("sample", lambda: True, kind="numeric"),
                Obligation("theorem", depends_on=("sample",)),
            ]).run(path)
            report = path.read_text(encoding="utf-8")
        self.assertFalse(results[1].passed)
        self.assertEqual(results[1].result.status, "numerical_evidence")
        self.assertIn("EVIDENCE [symbolic] theorem", report)
        self.assertIn("0 passed, 0 failed, 2 numerical evidence", report)

    def test_unrestricted_search_avoids_support_enumeration(self):
        variables = symbols("x:8")
        factors = tuple(Poly(variable, *variables) for variable in variables)
        target = Poly(sum(variable**2 for variable in variables), *variables)
        z = symbols("z")
        with patch.object(search, "combinations", side_effect=AssertionError("enumerated support")):
            polynomial = search.search_polynomial_certificate(target, squares=factors)
            series = search.search_series_certificate(
                (Poly(1 + z, z), Poly(1, z)), slack=((0, ()), (1, ()))
            )
        self.assertEqual(polynomial.status, "verified")
        self.assertEqual(series.status, "verified")
        self.assertEqual(
            search.search_polynomial_certificate(target, squares=factors, max_support=1).status,
            "inconclusive",
        )

    def test_unrestricted_search_recovers_feasible_free_weights(self):
        x = symbols("x")
        target = Poly(1, x)
        hypotheses = (Poly(x, x), Poly(x + 1, x), Poly(-x, x))
        result = search.search_polynomial_certificate(target, inequalities=hypotheses)
        self.assertEqual(result.status, "verified")
        self.assertTrue(all(weight >= 0 for weight, _ in result.witness.inequalities))

    def test_unrestricted_search_rejects_incompatible_weight_constraints(self):
        x = symbols("x")
        result = search.search_polynomial_certificate(
            Poly(-1, x), inequalities=(Poly(1, x), Poly(1, x))
        )
        self.assertEqual(result.status, "inconclusive")
