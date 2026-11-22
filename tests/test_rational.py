import unittest

from sympy import Poly, Rational, symbols

from decideable.rational import rational_identity


class RationalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = symbols("x y")

    def test_exact_multivariate_identity_with_mixed_domains(self):
        x, y = self.x, self.y
        self.assertTrue(
            rational_identity(
                Poly(Rational(1, 2) * x + y, x, y),
                Poly(x - y, x, y),
                Poly(x + 2 * y, x, y),
                Poly(2 * (x - y), x, y),
            )
        )

    def test_unequal_rational_functions(self):
        x = self.x
        self.assertFalse(
            rational_identity(Poly(x, x), Poly(x + 1, x), Poly(x, x), Poly(x - 1, x))
        )

    def test_cancellation_is_a_formal_identity(self):
        x = self.x
        self.assertTrue(
            rational_identity(
                Poly(x**2 - 1, x),
                Poly(x - 1, x),
                Poly(x + 1, x),
                Poly(1, x),
            )
        )

    def test_zero_numerators_are_allowed(self):
        x = self.x
        self.assertTrue(
            rational_identity(Poly(0, x), Poly(x, x), Poly(0, x), Poly(1, x))
        )

    def test_zero_denominators_are_rejected(self):
        zero = Poly(0, self.x)
        one = Poly(1, self.x)
        for arguments in ((one, zero, one, one), (one, one, one, zero)):
            with self.subTest(arguments=arguments), self.assertRaises(ZeroDivisionError):
                rational_identity(*arguments)

    def test_requires_polynomials(self):
        one = Poly(1, self.x)
        with self.assertRaises(TypeError):
            rational_identity(self.x, one, one, one)

    def test_rejects_nonrational_domains(self):
        x = self.x
        one = Poly(1, x)
        for polynomial in (Poly(0.5 * x, x), Poly(x, x, modulus=3)):
            with self.subTest(polynomial=polynomial), self.assertRaises(TypeError):
                rational_identity(polynomial, one, one, one)

    def test_rejects_different_generators_or_order(self):
        x, y = self.x, self.y
        one = Poly(1, x, y)
        for polynomial in (Poly(1, x), Poly(1, y, x)):
            with self.subTest(polynomial=polynomial), self.assertRaises(ValueError):
                rational_identity(polynomial, one, one, one)
