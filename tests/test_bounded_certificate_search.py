from dataclasses import asdict
from fractions import Fraction
from unittest import TestCase

from sympy import Poly, QQ, symbols

from decidable.polynomial import verify_polynomial_certificate
from decidable.search import search_polynomial_certificate, search_series_certificate
from decidable.series import verify_series_certificate


class GeometricSearchTests(TestCase):
    def test_recover_and_replay_hypothesis_and_slack_terms(self):
        z, a = symbols("z A")
        target = (Poly((2 + z) * a + z * (1 - z / 2) ** 2, z, a),
                  Poly(3 * (1 - z / 2) ** 2, z, a))
        hypothesis = Poly(a, z, a), Poly(1 - z / 2, z, a)
        candidates = dict(
            hypotheses=(([(0, (Fraction(1, 2),)), (1, (Fraction(1, 2),))], hypothesis),),
            slack=((1, ()), (2, ())),
        )

        result = search_series_certificate(target, **candidates)
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.witness.slack, ((Fraction(1, 3), 1, ()),))
        self.assertTrue(verify_series_certificate(target, **asdict(result.witness)))
        tampered = asdict(result.witness)
        tampered["slack"] = ((Fraction(1, 4), 1, ()),)
        self.assertFalse(verify_series_certificate(target, **tampered))
        exhausted = search_series_certificate(target, **candidates, max_support=2)
        self.assertEqual(exhausted.status, "inconclusive")
        self.assertIsNone(exhausted.witness)

    def test_reject_invalid_template_even_for_zero_target(self):
        z = symbols("z")
        target = Poly(0, z), Poly(1, z)
        with self.assertRaises(ValueError):
            search_series_certificate(target, slack=((0, (-1,)),))
        with self.assertRaises(TypeError):
            search_series_certificate(target, slack=((0, (0.5,)),))
        with self.assertRaises(ValueError):
            search_series_certificate(target, max_support=-1)
        self.assertEqual(search_series_certificate(target, max_support=0).status, "verified")


class PolynomialSearchTests(TestCase):
    def test_recover_and_replay_squares_and_signed_equation(self):
        x, y = symbols("x y")
        target = Poly((x + y) ** 2 / 2 + 3 * (x - y) ** 2 - x * (y - 1) / 3, x, y,
                      domain=QQ)
        squares = Poly(x + y, x, y), Poly(x - y, x, y)
        equations = ((Poly(x, x, y), Poly(y - 1, x, y)),)
        result = search_polynomial_certificate(target, equations=equations, squares=squares)
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.witness.equations[0][0], Poly(-x / 3, x, y, domain=QQ))
        self.assertTrue(verify_polynomial_certificate(target, **asdict(result.witness)))
        tampered = asdict(result.witness)
        tampered["squares"] = ((Fraction(-1, 2), squares[0]), result.witness.squares[1])
        self.assertFalse(verify_polynomial_certificate(target, **tampered))
        self.assertEqual(
            search_polynomial_certificate(
                target, equations=equations, squares=squares, max_support=2
            ).status,
            "inconclusive",
        )

    def test_recover_inequality_terms_after_dependent_candidates(self):
        x = symbols("x")
        inequality = Poly(x, x)
        result = search_polynomial_certificate(
            Poly(x + x**3, x),
            inequalities=(inequality, 2 * inequality),
            inequality_squares=((inequality, inequality),),
        )
        self.assertEqual(result.status, "verified")
        self.assertTrue(verify_polynomial_certificate(Poly(x + x**3, x), **asdict(result.witness)))

    def test_negative_candidate_is_inconclusive(self):
        x = symbols("x")
        result = search_polynomial_certificate(Poly(-x**2, x), squares=(Poly(x, x),))
        self.assertEqual(result.status, "inconclusive")
        self.assertIsNone(result.witness)
