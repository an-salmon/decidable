import unittest
from dataclasses import replace
from fractions import Fraction

from sympy import Float, Mul, Pow, log, symbols

from decidable.size_calculus.parametric import (
    AffineDomain,
    ParametricSizeBound,
    parametric_aggregate,
    parametric_bounded_raise,
    parametric_descend,
    parametric_power,
    parametric_product,
)
from decidable.size_calculus.typing import UpperBound


class ParametricSizeTests(unittest.TestCase):
    def setUp(self):
        self.p, self.q = symbols("p q", real=True)
        self.x, self.y = symbols("x y")
        self.r, self.u = symbols("r u", positive=True)
        self.domain = AffineDomain((self.p,), (2 - self.p,), (self.p,))
        self.bound = ParametricSizeBound(self.x, self.p, self.r, True, self.domain)

    def test_affine_context_preserves_strictness_and_nonvacuity(self):
        p = self.p
        self.assertTrue(self.domain.positive(p))
        self.assertTrue(self.domain.nonnegative(2 - p))
        self.assertFalse(self.domain.positive(2 - p))
        self.assertFalse(self.domain.nonnegative(p - 1))
        self.assertEqual(self.domain.affine(2 * p + Fraction(1, 3)), 2 * p + Fraction(1, 3))
        with self.assertRaises(ValueError):
            AffineDomain((p,), (-p,), (p,))
        with self.assertRaises(TypeError):
            AffineDomain((symbols("z", positive=True),))
        with self.assertRaises(TypeError):
            AffineDomain((p,), (Float("0.1") * p,))
        with self.assertRaises(ValueError):
            AffineDomain((p,), (self.q,))
        with self.assertRaises(ValueError):
            AffineDomain((p,), (p**2,))
        with self.assertRaises(ValueError):
            AffineDomain((p,), (Mul(p, 1 / p, evaluate=False),))
        with self.assertRaises(ValueError):
            replace(self.bound, reciprocal=2 - p)

    def test_power_product_and_descent_keep_the_whole_affine_domain(self):
        p, r, x = self.p, self.r, self.x
        self.assertEqual(parametric_power(self.bound, 2),
                         ParametricSizeBound(x**2, 2 * p, r**2, True, self.domain))
        self.assertEqual(parametric_power(replace(self.bound, scale=p**2), Fraction(1, 2)).scale, p)
        other = ParametricSizeBound(self.y, p + 1, self.u, True, self.domain)
        self.assertEqual(parametric_product((self.bound, other)),
                         ParametricSizeBound(x * self.y, 2 * p + 1, r * self.u, True, self.domain))
        self.assertEqual(parametric_descend(self.bound, 2),
                         ParametricSizeBound(x, 2, r * log(2)**(p - 2), True, self.domain))
        with self.assertRaises(ValueError):
            parametric_descend(self.bound, 1)
        with self.assertRaises(ValueError):
            parametric_product((self.bound, replace(other, domain=AffineDomain((p,), (), (p,)))))
        with self.assertRaises(ValueError):
            parametric_power(replace(self.bound, nonnegative=False), 2)

    def test_bounded_raise_retains_ratio_and_requires_positive_output(self):
        p, x, r, u = self.p, self.x, self.r, self.u
        bound = replace(self.bound, reciprocal=p + 1)
        self.assertEqual(parametric_bounded_raise(bound, p, UpperBound(x, u)),
                         ParametricSizeBound(x, p, r**(p / (p + 1)) * u**(1 - p / (p + 1)),
                                             True, self.domain))
        with self.assertRaises(ValueError):
            parametric_bounded_raise(bound, p, UpperBound(self.y, u))
        with self.assertRaises(ValueError):
            parametric_bounded_raise(bound, 2 - p, UpperBound(x, u))
        with self.assertRaises(ValueError):
            parametric_bounded_raise(bound, 0, UpperBound(x, u))

    def test_aggregation_explicitly_splits_and_keeps_the_boundary(self):
        p, r, u = self.p, self.r, self.u
        other = ParametricSizeBound(self.y, p, u, False, self.domain)
        small, large = parametric_aggregate((self.bound, other))
        self.assertEqual(small.expression, self.x + self.y)
        self.assertFalse(small.nonnegative)
        self.assertEqual(small.scale, r + u)
        self.assertEqual(small.domain.strict_inequalities, (p,))
        self.assertEqual(small.domain.inequalities, (2 - p, 1 - p))
        self.assertFalse(small.domain.positive(1 - p))
        self.assertTrue(small.domain.nonnegative(1 - p))
        self.assertEqual(large.scale, (1 + (p - 1) / log(2))**p * (r + u))
        self.assertEqual(large.domain.inequalities, (2 - p,))
        self.assertEqual(large.domain.strict_inequalities, (p, p - 1))
        self.assertTrue(large.domain.positive(p - 1))
        with self.assertRaises(ValueError):
            parametric_aggregate((self.bound, other), unit_factor=True)
        restricted = tuple(replace(bound, domain=small.domain) for bound in (self.bound, other))
        self.assertEqual(parametric_aggregate(restricted, unit_factor=True), (small,))
        restricted = tuple(replace(bound, domain=large.domain) for bound in (self.bound, other))
        self.assertEqual(parametric_aggregate(restricted), (large,))

    def test_equality_guards_use_the_affine_constraints(self):
        p, q = self.p, self.q
        domain = AffineDomain((p, q), (p - q, q - p, 1 - p), (p,))
        first = replace(self.bound, domain=domain)
        second = ParametricSizeBound(self.y, q, self.u, True, domain)
        result, = parametric_aggregate((first, second), unit_factor=True)
        self.assertEqual(result.reciprocal, p)
        self.assertEqual(result.domain, domain)
        with self.assertRaises(ValueError):
            parametric_aggregate((first, replace(second, reciprocal=q + 1)))

    def test_scale_exponent_denominators_must_be_defined_on_the_whole_domain(self):
        p = self.p
        self.assertEqual(replace(self.bound, scale=self.r**(1 / p)).scale, self.r**(1 / p))
        with self.assertRaises(ValueError):
            replace(self.bound, scale=self.r**(1 / (p - 1)))
        with self.assertRaises(ValueError):
            replace(self.bound, scale=Pow(2, 1 / self.u, evaluate=False))
        with self.assertRaises(ValueError):
            replace(self.bound, scale=symbols("p", positive=True))
