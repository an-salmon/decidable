from fractions import Fraction
import unittest

from sympy import I, Poly, symbols

from decidable.size_calculus.orlicz import (
    PolynomialSizeBound, PolynomialSizeFunction, polynomial_change_class,
    polynomial_enlarge_scale, polynomial_monotone, polynomial_multiply,
    polynomial_weighted_aggregate,
)
from decidable.size_calculus.typing import PointwiseOrder


class PolynomialSizeFunctionTests(unittest.TestCase):
    def setUp(self):
        self.v, self.u, self.x, self.y = symbols("v u X Y")
        self.r, self.s = symbols("r s", positive=True)
        self.convex = PolynomialSizeFunction(Poly(self.v**2 + self.v, self.v))
        self.nonconvex = PolynomialSizeFunction(
            Poly(self.v**3 - 3 * self.v**2 + 3 * self.v, self.v)
        )

    def test_validates_size_axioms_and_checks_convexity_separately(self):
        self.assertTrue(self.convex.is_convex)
        self.assertFalse(self.nonconvex.is_convex)
        for expression in (0, 1, self.v - self.v**2, -self.v):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                PolynomialSizeFunction(Poly(expression, self.v))
        for expression in (0.5 * self.v, I * self.v):
            with self.subTest(expression=expression), self.assertRaises(TypeError):
                PolynomialSizeFunction(Poly(expression, self.v))
        with self.assertRaises(ValueError):
            PolynomialSizeFunction(Poly(self.u + self.v, self.u, self.v))

    def test_pointwise_order_deterministic_multiply_and_enlarge(self):
        bound = PolynomialSizeBound(self.x, self.nonconvex, self.r)
        smaller = PolynomialSizeBound(self.y, self.nonconvex, self.r)
        self.assertEqual(polynomial_monotone(bound, PointwiseOrder(self.y, self.x)), smaller)
        self.assertEqual(polynomial_multiply(smaller, self.s),
                         PolynomialSizeBound(self.s * self.y, self.nonconvex, self.s * self.r))
        self.assertEqual(polynomial_enlarge_scale(smaller, Fraction(3, 2)),
                         PolynomialSizeBound(self.y, self.nonconvex, self.r * Fraction(3, 2)))
        with self.assertRaises(ValueError):
            polynomial_monotone(bound, PointwiseOrder(self.x, self.y))
        with self.assertRaises(ValueError):
            polynomial_enlarge_scale(bound, Fraction(1, 2))
        with self.assertRaises(ValueError):
            polynomial_multiply(bound, 0)

    def test_class_change_uses_the_pointwise_scale_direction(self):
        bound = PolynomialSizeBound(self.x, self.convex, self.r)
        target = PolynomialSizeFunction(Poly(4 * self.u**2 + 2 * self.u, self.u))
        self.assertEqual(polynomial_change_class(bound, target, 2),
                         PolynomialSizeBound(self.x, target, 2 * self.r))
        with self.assertRaises(ValueError):
            polynomial_change_class(bound, target, 1)
        with self.assertRaises(ValueError):
            polynomial_change_class(bound, target, 0)

    def test_convex_weighted_sum_of_signed_inputs_at_weighted_scale(self):
        a = PolynomialSizeBound(self.x, self.convex, self.r)
        matching = PolynomialSizeFunction(Poly(self.u**2 + self.u, self.u))
        b = PolynomialSizeBound(self.y, matching, self.s)
        unused = PolynomialSizeBound(self.x, self.nonconvex, self.r)
        result = polynomial_weighted_aggregate((a, b, unused), (2, Fraction(1, 2), 0))
        self.assertEqual(result, PolynomialSizeBound(2 * self.x + self.y / 2, self.convex,
                                                    2 * self.r + self.s / 2))
        with self.assertRaisesRegex(ValueError, "convex"):
            polynomial_weighted_aggregate((unused, unused), (1, 1))
        with self.assertRaisesRegex(ValueError, "share"):
            polynomial_weighted_aggregate((a, unused), (1, 1))
        with self.assertRaises(ValueError):
            polynomial_weighted_aggregate((a,), (0,))
        with self.assertRaises(ValueError):
            polynomial_weighted_aggregate((a,), (-1,))
