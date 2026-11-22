from fractions import Fraction
from unittest import TestCase

from decideable.gram import verify_gram_certificate


class GramCertificateTests(TestCase):
    def test_signed_equation_multiplier_and_square(self):
        self.assertTrue(
            verify_gram_certificate(
                2,
                {(1, 1): 2, (0, 1): -2},
                equations=[(-1, {(0, 0): 1, (1, 1): -1})],
                squares=[(1, (1, -1))],
            )
        )

    def test_fractional_inequality_multiplier_and_weight(self):
        self.assertTrue(
            verify_gram_certificate(
                1,
                {(0, 0): Fraction(5, 6)},
                inequalities=[(Fraction(1, 3), {(0, 0): 1})],
                squares=[(Fraction(1, 2), (1,))],
            )
        )

    def test_requires_full_cross_term(self):
        self.assertFalse(
            verify_gram_certificate(
                2, {(0, 0): 1, (0, 1): 1, (1, 1): 1}, squares=[(1, (1, 1))]
            )
        )

    def test_negative_inequality_multiplier_and_square_weight(self):
        target = {(0, 0): -1}
        self.assertFalse(
            verify_gram_certificate(1, target, inequalities=[(-1, {(0, 0): 1})])
        )
        self.assertFalse(verify_gram_certificate(1, target, squares=[(-1, (1,))]))

    def test_rejects_inexact_coefficients_and_boolean_inputs(self):
        with self.assertRaises(TypeError):
            verify_gram_certificate(1, {(0, 0): 0.0})
        with self.assertRaises(TypeError):
            verify_gram_certificate(1, {}, squares=[(0, (True,))])
        with self.assertRaises(TypeError):
            verify_gram_certificate(1, {}, inequalities=[(False, {})])
        with self.assertRaises(TypeError):
            verify_gram_certificate(1, {(False, 0): 1})
        with self.assertRaises(TypeError):
            verify_gram_certificate(True, {})

    def test_rejects_malformed_terms_with_zero_or_negative_weights(self):
        with self.assertRaises(ValueError):
            verify_gram_certificate(2, {}, equations=[(0, {(1, 0): 1})])
        with self.assertRaises(ValueError):
            verify_gram_certificate(2, {}, squares=[(0, (1,))])
        with self.assertRaises(ValueError):
            verify_gram_certificate(1, {}, inequalities=[(-1, {(0, 1): 1})])

    def test_zero_vectors(self):
        self.assertTrue(verify_gram_certificate(0, {}, squares=[(1, ())]))
        with self.assertRaises(ValueError):
            verify_gram_certificate(0, {(0, 0): 0})
        with self.assertRaises(ValueError):
            verify_gram_certificate(-1, {})
