import unittest
from fractions import Fraction

from sympy import Poly, Rational, symbols

from decidable.parameters import PositiveParameters
from decidable.polynomial import verify_polynomial_certificate


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

    def test_inequality_allows_multiple_weighted_square_factors(self):
        x, y = self.x, self.y
        target = Poly(x * ((x + y) ** 2 / 2 + y**2), x, y)
        hypothesis = Poly(x, x, y)
        self.assertTrue(
            verify_polynomial_certificate(
                target,
                inequality_squares=[
                    (Fraction(1, 2), Poly(x + y, x, y), hypothesis),
                    (1, Poly(y, x, y), hypothesis),
                ],
            )
        )

    def test_positive_parameter_weights_and_coefficients(self):
        x = self.x
        p = symbols("p")
        parameters = PositiveParameters((p,))
        weight = parameters.rational(Poly(p + 1, p), Poly(p, p))
        hypothesis = Poly(x, x)
        factor = Poly(x + p, x, domain=parameters.field)
        target = Poly((p + 1) / p * (1 + (x + p) ** 2) * x, x, domain=parameters.field)
        self.assertIs(
            verify_polynomial_certificate(
                target,
                inequalities=[(weight, hypothesis)],
                inequality_squares=[(weight, factor, hypothesis)],
                parameters=parameters,
            ),
            True,
        )

    def test_rejects_parameter_weight_without_proven_sign(self):
        x = self.x
        p = symbols("p")
        parameters = PositiveParameters((p,))
        weight = parameters.rational(Poly(p - 1, p), Poly(1, p))
        target = Poly((p - 1) * x**2, x, domain=parameters.field)
        self.assertIs(
            verify_polynomial_certificate(
                target, squares=[(weight, Poly(x, x))], parameters=parameters
            ),
            False,
        )

    def test_parameterized_polynomial_requires_parameter_declaration(self):
        x = self.x
        p = symbols("p")
        parameters = PositiveParameters((p,))
        with self.assertRaises(TypeError):
            verify_polynomial_certificate(Poly(p * x**2, x, domain=parameters.field))

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
        self.assertIs(
            verify_polynomial_certificate(
                Poly(-x**3, x), inequality_squares=[(-1, Poly(x, x), Poly(x, x))]
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
        with self.assertRaises(ValueError):
            verify_polynomial_certificate(
                Poly(0, x), inequality_squares=[(0, Poly(x, x), Poly(self.y, self.y))]
            )
