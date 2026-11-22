import unittest
from fractions import Fraction

from sympy import Poly, Rational, symbols

from decideable.series import verify_series_certificate


class SeriesCertificateTests(unittest.TestCase):
    def setUp(self):
        self.z, self.x, self.y = symbols("z x y")

    def series(self, numerator, denominator=1):
        generators = (self.z, self.x, self.y)
        return Poly(numerator, *generators), Poly(denominator, *generators)

    def test_symbolic_recurrence(self):
        kernel = 1 - self.z / 2
        residual = kernel * self.x - 1
        self.assertTrue(
            verify_series_certificate(
                self.series(residual, kernel),
                hypotheses=[([(1, 0, (Fraction(1, 2),))], self.series(residual))],
            )
        )

    def test_product_of_kernels_multiple_hypotheses_and_slack(self):
        z, x, y = self.z, self.x, self.y
        denominator = (1 - z / 2) * (1 - 2 * z)
        numerator = (z * denominator + z**2 / 2) * x + 2 * denominator * y
        numerator += z**3 * (1 - 2 * z)
        self.assertTrue(
            verify_series_certificate(
                self.series(numerator, denominator),
                hypotheses=[
                    ([(1, 1, ()), (Fraction(1, 2), 2, (Fraction(1, 2), 2))], self.series(x)),
                    ([(2, 0, ())], self.series(y)),
                ],
                slack=[(1, 3, (Fraction(1, 2),))],
            )
        )

    def test_certificate_must_match_all_coefficients(self):
        self.assertFalse(
            verify_series_certificate(self.series(1 + self.z**20), slack=[(1, 0, ())])
        )

    def test_negative_weight_or_ratio_rejects_an_exact_identity(self):
        self.assertFalse(verify_series_certificate(self.series(-1), slack=[(-1, 0, ())]))
        self.assertFalse(
            verify_series_certificate(self.series(1, 1 + self.z), slack=[(1, 0, (-1,))])
        )

    def test_denominator_is_a_unit_in_the_formal_series_ring(self):
        for denominator in (0, self.z, 1 + self.x):
            with self.subTest(denominator=denominator), self.assertRaises(ValueError):
                verify_series_certificate(self.series(1, denominator))
        with self.assertRaises(ValueError):
            verify_series_certificate(
                self.series(0), hypotheses=[([], self.series(1, self.z))]
            )
        self.assertTrue(verify_series_certificate(self.series(0, -2)))

    def test_rejects_nonlinear_series_symbols_and_mismatched_generators(self):
        with self.assertRaises(ValueError):
            verify_series_certificate(self.series(self.x * self.y))
        with self.assertRaises(ValueError):
            verify_series_certificate(
                (Poly(1, self.z, self.x), Poly(1, self.x, self.z))
            )

    def test_rejects_invalid_geometric_terms(self):
        invalid_terms = ((0.5, 0, ()), (0, 0, (True,)), (1, False, ()), (1, 0, (Rational(1, 2),)))
        for term in invalid_terms:
            with self.subTest(term=term), self.assertRaises(TypeError):
                verify_series_certificate(self.series(0), slack=[term])
        with self.assertRaises(ValueError):
            verify_series_certificate(self.series(0), slack=[(1, -1, ())])

    def test_rejects_inexact_polynomials(self):
        with self.assertRaises(TypeError):
            verify_series_certificate(self.series(0.5))
