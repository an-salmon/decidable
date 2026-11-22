import unittest
from fractions import Fraction

from sympy import Poly, Rational, symbols

from decideable.polynomial import verify_polynomial_certificate


class PolynomialCertificateTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = symbols("x y")

    def test_inequality_and_square_with_fractional_weights(self):
        x, y = self.x, self.y
        target = Poly(Rational(1, 3) * (x - y) + (x + y) ** 2 / 2, x, y)
        self.assertIs(
            verify_polynomial_certificate(
                target,
                inequalities=[(Fraction(1, 3), Poly(x - y, x, y))],
                squares=[(Fraction(1, 2), Poly(x + y, x, y))],
            ),
            True,
        )

    def test_equation_allows_polynomial_multiplier(self):
        x = self.x
        self.assertTrue(
            verify_polynomial_certificate(
                Poly(x - x**3 + 1, x),
                equations=[(Poly(-x, x), Poly(x**2 - 1, x))],
                squares=[(1, Poly(1, x))],
            )
        )

    def test_rejects_incorrect_coefficients(self):
        x = self.x
        self.assertIs(
            verify_polynomial_certificate(
                Poly(Rational(1, 3) * x**2, x),
                squares=[(Fraction(1, 2), Poly(x, x))],
            ),
            False,
        )

    def test_rejects_negative_weights_even_when_identity_matches(self):
        x = self.x
        self.assertIs(
            verify_polynomial_certificate(
                Poly(-x, x), inequalities=[(-1, Poly(x, x))]
            ),
            False,
        )
        self.assertIs(
            verify_polynomial_certificate(
                Poly(-x**2, x), squares=[(-1, Poly(x, x))]
            ),
            False,
        )

    def test_rejects_inexact_or_boolean_weights(self):
        x = self.x
        for bad_weight in (0.5, True):
            with self.subTest(weight=bad_weight), self.assertRaises(TypeError):
                verify_polynomial_certificate(
                    Poly(0, x), squares=[(bad_weight, Poly(0, x))]
                )

    def test_validates_all_polynomials_even_at_zero_weight(self):
        x, y = self.x, self.y
        target = Poly(0, x, y)
        with self.assertRaises(ValueError):
            verify_polynomial_certificate(target, squares=[(0, Poly(x, y, x))])
        with self.assertRaises(TypeError):
            verify_polynomial_certificate(
                target, inequalities=[(0, Poly(x, x, y, modulus=3))]
            )
        with self.assertRaises(TypeError):
            verify_polynomial_certificate(
                target, equations=[(Poly(0, x, y), Poly(0.5 * x, x, y))]
            )

    def test_rejects_nonpolynomial_inputs_and_malformed_pairs(self):
        x = self.x
        with self.assertRaises(TypeError):
            verify_polynomial_certificate(x)
        with self.assertRaises(TypeError):
            verify_polynomial_certificate(Poly(0, x), equations=[(1, Poly(x, x))])
        with self.assertRaises(ValueError):
            verify_polynomial_certificate(Poly(0, x), squares=[(1,)])
