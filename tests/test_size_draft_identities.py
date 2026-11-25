"""Exact arithmetic obligations extracted from the size-calculus draft."""

from fractions import Fraction
import unittest

from sympy import Poly, QQ, symbols


class DraftIdentityTests(unittest.TestCase):
    def test_rational_power_tangent_defect_through_degree_twenty_five(self):
        u = symbols("u")
        for maximum in range(1, 26):
            for q in range(1, maximum + 1):
                terms = {
                    (j,): (j + 1) * (maximum - q) if j < q else q * (maximum - 1 - j)
                    for j in range(maximum - 1)
                }
                quotient = Poly.from_dict(terms, u, domain=QQ)
                residual = Poly(q * u**maximum - maximum * u**q + maximum - q, u, domain=QQ)
                self.assertEqual(residual, Poly((u - 1)**2, u, domain=QQ) * quotient)
                self.assertTrue(all(value >= 0 for value in quotient.all_coeffs()))

    def test_bernstein_exponent_reference_points(self):
        for n, ceiling, mean, h in ((1, 1, 1, 1), (7, 4, 1, 2), (3, 11, 2, 23)):
            eta = Fraction(h, 1) / (ceiling * (mean + Fraction(h, 3)))
            self.assertGreater(1 - eta * ceiling / 3, 0)
            centered = n * h * eta - Fraction(n, 2) * eta**2 * ceiling * mean / (1 - eta * ceiling / 3)
            self.assertEqual(centered, Fraction(n * h**2, 2) / (ceiling * (mean + Fraction(h, 3))))
