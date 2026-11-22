import unittest

from sympy import Poly, symbols

from decidable import PositiveParameters, rational_identity


class PositiveParameterTests(unittest.TestCase):
    def setUp(self):
        self.p, self.e, self.x = symbols("p e x")
        self.parameters = PositiveParameters((self.p, self.e))

    def fraction(self, numerator, denominator):
        p, e = self.parameters.symbols
        return self.parameters.rational(Poly(numerator, p, e), Poly(denominator, p, e))

    def test_signs_with_a_declared_gap(self):
        p, e = self.p, self.e
        inverse_difference = self.fraction(e, p * (p + e))
        self.assertTrue(self.parameters.nonnegative(inverse_difference))
        self.assertTrue(self.parameters.nonzero(inverse_difference))
        self.assertFalse(self.parameters.nonnegative(self.fraction(-e, p * (p + e))))
        self.assertFalse(self.parameters.nonnegative(self.fraction(p - e, 1)))

    def test_denominator_poles_are_rejected_before_cancellation(self):
        with self.assertRaises(ValueError):
            self.fraction(self.p - self.e, self.p - self.e)
        with self.assertRaises(ZeroDivisionError):
            self.fraction(self.p, 0)
        with self.assertRaises(ValueError):
            self.parameters.coefficient(self.parameters.field.from_sympy(1 / (self.p - self.e)))

    def test_rational_identity_with_parameter_coefficients(self):
        p, e, x = self.p, self.e, self.x
        domain = self.parameters.field
        inputs = tuple(Poly(value, x, domain=domain) for value in (e * x, p * (p + e), (1 / p - 1 / (p + e)) * x, 1))
        self.assertTrue(rational_identity(*inputs, parameters=self.parameters))
        with self.assertRaises(TypeError):
            rational_identity(*inputs)
