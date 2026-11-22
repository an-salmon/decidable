import unittest
from fractions import Fraction

from sympy import Poly, Rational, symbols

from decidable.parameters import PositiveParameters
from decidable.series import rebase_series, shift_series, verify_series_certificate


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

    def test_geometric_certificate_with_positive_parameters(self):
        p, gap = symbols("p gap")
        parameters = PositiveParameters((p, gap))
        ratio = parameters.rational(Poly(p, p, gap), Poly(p + gap, p, gap))
        weight = parameters.rational(Poly(gap, p, gap), Poly(1, p, gap))
        z, x = self.z, self.x
        target = (
            Poly(gap * x, z, x, domain=parameters.field),
            Poly(1 - p * z / (p + gap), z, x, domain=parameters.field),
        )
        hypothesis = (Poly(x, z, x), Poly(1, z, x))
        self.assertTrue(
            verify_series_certificate(
                target,
                hypotheses=[([(weight, 0, (ratio,))], hypothesis)],
                parameters=parameters,
            )
        )

    def test_unproved_parameter_sign_rejects_a_matching_certificate(self):
        p = symbols("p")
        parameters = PositiveParameters((p,))
        weight = parameters.rational(Poly(p - 1, p), Poly(1, p))
        target = (Poly(p - 1, self.z, domain=parameters.field), Poly(1, self.z))
        self.assertFalse(
            verify_series_certificate(target, slack=[(weight, 0, ())], parameters=parameters)
        )

    def test_scale_denominator_constant_requires_uniform_nonvanishing(self):
        p = symbols("p")
        parameters = PositiveParameters((p,))
        for constant in (p, 1 - p):
            target = (Poly(0, self.z), Poly(constant - self.z, self.z, domain=parameters.field))
            with self.subTest(constant=constant):
                if constant == p:
                    self.assertTrue(verify_series_certificate(target, parameters=parameters))
                else:
                    with self.assertRaises(ValueError):
                        verify_series_certificate(target, parameters=parameters)

    def test_rebase_and_shift_with_multiple_known_initials(self):
        z, x, y = self.z, self.x, self.y
        raw = self.series(x + z * y, 2 - z)
        initials = {x: Fraction(3, 2), y: -1}

        rebased_numerator, rebased_denominator = rebase_series(raw, initials)
        self.assertEqual(rebased_numerator.as_expr(), z * x + z**2 * y - z + Rational(3, 2))
        self.assertEqual(rebased_denominator.as_expr(), 2 - z)

        shifted_numerator, shifted_denominator = shift_series(raw, initials)
        self.assertEqual(shifted_numerator.as_expr(), x + z * y - Rational(1, 4))
        self.assertEqual(shifted_denominator.as_expr(), 2 - z)

    def test_shift_without_series_symbols(self):
        z = self.z
        numerator, denominator = shift_series((Poly(1, z), Poly(1 - z, z)), {})
        self.assertEqual(numerator.as_expr(), 1)
        self.assertEqual(denominator.as_expr(), 1 - z)

    def test_shift_requires_exact_initials_for_every_symbol(self):
        raw = self.series(self.x)
        for initials in ({self.x: 0}, {self.x: 0, self.y: 0, self.z: 0}):
            with self.subTest(initials=initials), self.assertRaises(ValueError):
                shift_series(raw, initials)
        with self.assertRaises(TypeError):
            shift_series(raw, {self.x: 0.0, self.y: 0})

    def test_shift_with_parameter_initial_verifies_a_tail_inequality(self):
        p = symbols("p")
        parameters = PositiveParameters((p,))
        ratio = parameters.rational(Poly(p, p), Poly(1, p))
        initial = parameters.rational(Poly(p, p), Poly(1 + p, p))
        z, x = self.z, self.x

        raw = (Poly(x, z, x), Poly(1 - p * z, z, x, domain=parameters.field))
        tail = shift_series(raw, {x: initial}, parameters=parameters)
        x_tail = (Poly(x, z, x), Poly(1, z, x))
        self.assertTrue(
            verify_series_certificate(
                tail,
                hypotheses=[([(1, 0, (ratio,))], x_tail)],
                slack=[(ratio * initial, 0, (ratio,))],
                parameters=parameters,
            )
        )
