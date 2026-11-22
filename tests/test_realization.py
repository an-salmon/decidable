from fractions import Fraction
from unittest import TestCase

from sympy import Poly, QQ, symbols

from decidable.realization import recurrence_to_series


class RecurrenceRealizationTests(TestCase):
    def test_exact_multiplication_replays_initials_and_recurrence(self):
        z = symbols("z")
        coefficients = Fraction(1, 2), Fraction(-1, 3)
        sequence = [2, Fraction(3, 5)]
        numerator, denominator = recurrence_to_series(coefficients, sequence, scale=z)
        self.assertEqual(numerator, Poly(2 - 2 * z / 5, z, domain=QQ))
        self.assertEqual(denominator, Poly(1 - z / 2 + z**2 / 3, z, domain=QQ))
        for n in range(2, 10):
            sequence.append(coefficients[0] * sequence[n - 1] + coefficients[1] * sequence[n - 2])
        prefix = Poly.from_list(sequence[::-1], z, domain=QQ)
        self.assertTrue((denominator * prefix - numerator).rem(Poly(z**10, z)).is_zero)

    def test_trailing_zero_coefficients_preserve_delayed_start(self):
        t = symbols("t")
        numerator, denominator = recurrence_to_series((2, 0), (1, 3), scale=t)
        self.assertEqual(numerator, Poly(1 + t, t, domain=QQ))
        self.assertEqual(denominator, Poly(1 - 2 * t, t, domain=QQ))
        self.assertEqual(
            recurrence_to_series((0, 0), (1, 3), scale=t),
            (Poly(1 + 3 * t, t, domain=QQ), Poly(1, t, domain=QQ)),
        )
        self.assertEqual(
            recurrence_to_series((), (), scale=t),
            (Poly(0, t, domain=QQ), Poly(1, t, domain=QQ)),
        )

    def test_reject_inexact_or_mismatched_input(self):
        with self.assertRaises(TypeError):
            recurrence_to_series((0.5,), (1,))
        with self.assertRaises(ValueError):
            recurrence_to_series((1, 2), (1,))
        with self.assertRaises(TypeError):
            recurrence_to_series((1,), (1,), scale="z")
