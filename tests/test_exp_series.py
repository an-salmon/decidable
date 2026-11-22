import unittest
from dataclasses import replace
from fractions import Fraction

from sympy import Poly, Rational, symbols

from decidable.exp_series import check_exp_series, verify_exp_series_certificate


x, y = symbols("x y")


class ExpSeriesTests(unittest.TestCase):
    def test_forward_differences_certify_the_whole_integer_tail(self):
        p = Poly(Rational(1, 2) * (x**2 - 3*x + 4), x)
        q = Poly(Rational(1, 2), x)
        early = check_exp_series(p, q)
        self.assertEqual(early.status, "inconclusive")

        result = check_exp_series(p, q, tail_start=2)
        self.assertEqual(result.status, "verified")
        self.assertIn("x >= 0", result.scope)
        certificate = result.witness
        self.assertEqual(certificate.tail_start, 2)
        self.assertEqual(certificate.prefix_coefficients, (Fraction(5, 2), Fraction(1, 2)))
        self.assertEqual(certificate.forward_differences, (Fraction(0), Fraction(1, 2), Fraction(1)))
        self.assertEqual(verify_exp_series_certificate(p, q, certificate), result)
        tampered = replace(certificate, forward_differences=(Fraction(0),) * 3)
        self.assertEqual(verify_exp_series_certificate(p, q, tampered).status, "inconclusive")

    def test_negative_taylor_coefficient_is_inconclusive_for_a_square(self):
        result = check_exp_series(Poly(0, x), Poly((x - 1)**2, x))
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.witness, 1)

    def test_auto_advances_to_a_replayable_tail_cutoff(self):
        p = Poly(Rational(1, 2) * (x**2 - 3*x + 4), x)
        q = Poly(Rational(1, 2), x)
        result = check_exp_series(p, q, tail_start="auto")
        self.assertEqual(result, check_exp_series(p, q, tail_start=2))
        self.assertEqual(verify_exp_series_certificate(p, q, result.witness), result)

    def test_auto_finds_a_negative_coefficient_in_a_tail_with_negative_leading_term(self):
        p, q = Poly(1024 - x, x), Poly(0, x)
        result = check_exp_series(p, q, tail_start="auto")
        self.assertEqual(result.status, "inconclusive")
        self.assertGreater(result.witness, 1024)
        self.assertLess(1024 - result.witness, 0)

    def test_auto_checks_prefix_and_zero_polynomial_tail(self):
        result = check_exp_series(Poly(0, x), Poly((x - 1)**2, x), tail_start="auto")
        self.assertEqual(result.status, "inconclusive")
        self.assertEqual(result.witness, 1)
        result = check_exp_series(Poly(0, x), Poly(x, x), tail_start="auto")
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.witness.tail_start, 2)

    def test_auto_keeps_a_negative_constant_as_disproof(self):
        result = check_exp_series(Poly(-1, x), Poly(0, x), tail_start="auto")
        self.assertEqual(result.status, "disproved")
        self.assertEqual(result.witness, 0)

    def test_negative_constant_disproves_at_zero_even_when_prefix_is_empty(self):
        result = check_exp_series(Poly(-1, x), Poly(0, x))
        self.assertEqual(result.status, "disproved")
        self.assertEqual(result.witness, 0)

    def test_zero_and_finite_polynomial(self):
        zero = check_exp_series(Poly(0, x), Poly(0, x))
        self.assertEqual(zero.status, "verified")
        self.assertEqual(zero.witness.prefix_coefficients, ())
        self.assertEqual(zero.witness.forward_differences, (Fraction(0),))

        result = check_exp_series(Poly(0, x), Poly(2*x**2, x))
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.witness.prefix_coefficients, (Fraction(0), Fraction(0), Fraction(4)))

    def test_input_domain_and_certificate_boundary(self):
        with self.assertRaises(TypeError):
            check_exp_series(Poly(0.5*x, x), Poly(0, x))
        with self.assertRaises(ValueError):
            check_exp_series(Poly(x+y, x, y), Poly(0, x, y))
        with self.assertRaises(ValueError):
            check_exp_series(Poly(x, x), Poly(0, y))
        with self.assertRaises(TypeError):
            check_exp_series(Poly(x, x), Poly(0, x), tail_start=True)
        with self.assertRaises(ValueError):
            check_exp_series(Poly(x, x), Poly(x, x), tail_start=1)

        p, q = Poly(x, x), Poly(x, x)
        result = check_exp_series(p, q)
        certificate = replace(result.witness, tail_start=1)
        self.assertEqual(verify_exp_series_certificate(p, q, certificate).status, "inconclusive")
        floats = replace(result.witness, forward_differences=(2., 1.))
        with self.assertRaises(TypeError):
            verify_exp_series_certificate(p, q, floats)
