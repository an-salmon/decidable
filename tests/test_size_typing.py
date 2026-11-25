import unittest
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

from sympy import I, exp, log, oo, sqrt, symbols

from decidable.size_calculus.typing import (
    BernsteinTailBound,
    BoundedMean,
    DerivationNode,
    Independence,
    IntegralAggregationPremise,
    PointwiseOrder,
    SizeBound,
    TailBound,
    UpperBound,
    bounded_raise,
    enlarge_scale,
    independent_bernstein,
    independent_mean,
    integral_aggregate,
    monotone,
    multiply,
    power,
    size_from_tail,
    tail_from_size,
    verify_derivation,
    weighted_aggregate,
)


class SizeTypingTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.z = symbols("x y z")
        self.r, self.u = symbols("r u", positive=True)
        self.bound = SizeBound(self.x, Fraction(1, 2), self.r, True)

    def test_multiply_and_pointwise_order_keep_their_hypotheses(self):
        result = multiply(self.bound, self.u)
        self.assertEqual(result, SizeBound(self.u * self.x, Fraction(1, 2),
                                           self.u * self.r, True))
        with self.assertRaises(ValueError):
            multiply(self.bound, -1)

        premise = PointwiseOrder(self.y, result.expression)
        smaller = monotone(result, premise)
        self.assertEqual(smaller, SizeBound(self.y, Fraction(1, 2), self.u * self.r, False))
        with self.assertRaises(ValueError):
            monotone(result, PointwiseOrder(self.y, self.z))

        nodes = (DerivationNode("x", "given", argument="input"),
                 DerivationNode("scaled", "multiply", ("x",), self.u),
                 DerivationNode("pointwise", "monotone", ("scaled",), "order"))
        self.assertEqual(verify_derivation({"input": self.bound}, nodes, smaller,
                                           deterministic_symbols=(self.u,),
                                           pointwise_orders={"order": premise}).status,
                         "verified")
        self.assertEqual(verify_derivation({"input": self.bound}, nodes, smaller,
                                          deterministic_symbols=(self.u,)).status,
                         "inconclusive")
        self.assertEqual(verify_derivation({"input": self.bound}, nodes, smaller,
                                          pointwise_orders={"order": premise}).status,
                         "inconclusive")

    def test_scale_enlargement_uses_exact_arithmetic_and_pays_for_each_term(self):
        bound = replace(self.bound, scale=sqrt(2) * self.r)
        self.assertEqual(enlarge_scale(bound, 3 * self.r / 2).scale, 3 * self.r / 2)
        with self.assertRaises(ValueError):
            enlarge_scale(bound, 7 * self.r / 5)

        bound = replace(self.bound, scale=2 * self.r + self.u)
        self.assertEqual(enlarge_scale(bound, 3 * self.r + self.u).scale, 3 * self.r + self.u)
        with self.assertRaises(ValueError):
            enlarge_scale(bound, self.r + 2 * self.u)

        nodes = (DerivationNode("x", "given", argument="input"),
                 DerivationNode("scale", "enlarge_scale", ("x",), self.r + 2 * self.u))
        goal = replace(bound, scale=self.r + 2 * self.u)
        self.assertEqual(verify_derivation({"input": bound}, nodes, goal).status, "unsupported")

    def test_bounded_raise_checks_the_deterministic_ceiling_symbols(self):
        reciprocal = Fraction(1, 4)
        nodes = (DerivationNode("x", "given", argument="x"),
                 DerivationNode("raise", "bounded_raise", ("x",), (reciprocal, "ceiling")))
        options = {"upper_bounds": {"ceiling": UpperBound(self.x, self.u)}}
        goal = bounded_raise(self.bound, reciprocal, options["upper_bounds"]["ceiling"])
        self.assertEqual(verify_derivation({"x": self.bound}, nodes, goal, **options).status,
                         "inconclusive")
        self.assertEqual(verify_derivation({"x": self.bound}, nodes, goal,
                                           deterministic_symbols=(self.u,), **options).status,
                         "verified")

        existing = UpperBound(self.x, self.r)
        goal = bounded_raise(self.bound, reciprocal, existing)
        self.assertEqual(verify_derivation({"x": self.bound}, nodes, goal,
                                           upper_bounds={"ceiling": existing}).status,
                         "verified")

    def test_weighted_aggregation_retains_the_constant_and_ignores_zero_weights(self):
        left = replace(self.bound, reciprocal=2)
        right = SizeBound(self.y, 2, self.u, False)
        ignored = SizeBound(self.z, 1, self.r, True)
        result = weighted_aggregate((left, right, ignored), (2, Fraction(1, 3), 0))
        self.assertEqual(result, SizeBound(2 * self.x + self.y / 3, 2,
                                           (1 + 1 / log(2)) ** 2 * (2 * self.r + self.u / 3),
                                           False))
        with self.assertRaises(ValueError):
            weighted_aggregate((left, right), (1, 1), unit_factor=True)
        with self.assertRaises(ValueError):
            weighted_aggregate((left, right), (0, 0))
        with self.assertRaises(ValueError):
            weighted_aggregate((left, right), (1, -1))
        with self.assertRaises(ValueError):
            weighted_aggregate((left, right), (1,))

        nodes = (DerivationNode("x", "given", argument="x"),
                 DerivationNode("y", "given", argument="y"),
                 DerivationNode("z", "given", argument="z"),
                 DerivationNode("weighted", "weighted_aggregate", ("x", "y", "z"),
                                (2, Fraction(1, 3), 0)))
        self.assertEqual(verify_derivation({"x": left, "y": right, "z": ignored},
                                           nodes, result).status, "verified")

    def test_forward_and_reverse_tail_type_checks(self):
        forward = tail_from_size(self.bound)
        self.assertEqual(forward, TailBound(self.x, Fraction(1, 2), self.r, 2,
                                            include_threshold=True, nonnegative=True))
        nodes = (DerivationNode("x", "given", argument="x"),
                 DerivationNode("t", "tail_from_size", ("x",)))
        self.assertEqual(verify_derivation({"x": self.bound}, nodes, forward).status, "verified")
        with self.assertRaises(ValueError):
            size_from_tail(forward)
        with self.assertRaises(ValueError):
            size_from_tail(replace(forward, prefactor=1, threshold=1))
        with self.assertRaises(TypeError):
            power(forward, 2)

        tail = TailBound(self.y, Fraction(1, 2), self.r, 1, comparison=">")
        want = SizeBound(self.y, Fraction(1, 2), sqrt(2) * self.r, False)
        self.assertEqual(size_from_tail(tail), want)
        nodes = (DerivationNode("t", "tail_given", argument="tail"),
                 DerivationNode("o", "size_from_tail", ("t",)))
        self.assertEqual(verify_derivation({}, nodes, want, tail_givens={"tail": tail}).status,
                         "verified")
        forged = (nodes[0], DerivationNode("f", "enlarge_scale", ("t",), want.scale))
        self.assertEqual(verify_derivation({}, forged, want, tail_givens={"tail": tail}).status,
                         "inconclusive")

    def test_exact_real_quantities_and_typed_replay(self):
        for expression in (-1, I, oo):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                replace(self.bound, expression=expression)
        forged = SimpleNamespace(id="f", rule="given", operands=(), argument="x")
        self.assertEqual(verify_derivation({"x": self.bound}, (forged,), self.bound).status,
                         "inconclusive")
        given = DerivationNode("x", "given", argument="x")
        self.assertEqual(verify_derivation({"x": self.bound}, (given,), object()).status,
                         "inconclusive")

    def test_independent_mean_checks_the_exact_summands_and_parameter(self):
        declaration = Independence((self.x, self.y))
        premises = (BoundedMean(self.x, self.r, self.u), BoundedMean(self.y, self.r, self.u))
        parameter = Fraction(5, 4)
        goal = SizeBound((self.x + self.y) / 2 - 2 * self.u, 1, 2 * self.r / 5, False)
        self.assertEqual(independent_mean(declaration, premises, parameter), goal)
        with self.assertRaises(ValueError):
            independent_mean(Independence((self.x, self.z)), premises, parameter)
        with self.assertRaises(ValueError):
            independent_mean(declaration, (premises[0], replace(premises[1], ceiling=2)))
        with self.assertRaises(ValueError):
            independent_mean(declaration, premises, Fraction(3, 2))
        node = DerivationNode("i", "independent_mean",
                              argument=("mutual", ("mx", "my"), parameter))
        mean_givens = dict(zip(("mx", "my"), premises))
        self.assertEqual(verify_derivation({}, (node,), goal, bounded_means=mean_givens,
                                          deterministic_symbols=(self.r, self.u),
                                          independence={"mutual": declaration}).status, "verified")
        for declared in ((), (self.r,), (self.u,)):
            with self.subTest(declared=declared):
                self.assertEqual(verify_derivation({}, (node,), goal, bounded_means=mean_givens,
                                                  deterministic_symbols=declared,
                                                  independence={"mutual": declaration}).status,
                                 "inconclusive")
        self.assertEqual(verify_derivation({}, (node,), goal, bounded_means=mean_givens).status,
                         "inconclusive")

    def test_integral_aggregation_requires_the_named_measurable_hypothesis(self):
        premise = IntegralAggregationPremise(self.x, self.y, self.u, "measure", 2, self.r)
        goal = SizeBound(self.x, 2, (1 + 1 / log(2))**2 * self.r, True)
        self.assertEqual(integral_aggregate(premise), goal)
        self.assertEqual(integral_aggregate(replace(premise, reciprocal=1)).scale, self.r)
        with self.assertRaises(ValueError):
            replace(premise, scale_density=0)
        nodes = (DerivationNode("i", "integral_aggregate", argument="uniform integral"),)
        givens = {"uniform integral": premise}
        self.assertEqual(verify_derivation({}, nodes, goal, integral_aggregations=givens,
                                           deterministic_symbols=(self.r,)).status, "verified")
        self.assertEqual(verify_derivation({}, nodes, goal, integral_aggregations=givens).status,
                         "inconclusive")
        self.assertEqual(verify_derivation({}, nodes, goal, deterministic_symbols=(self.r,)).status,
                         "inconclusive")
        wrong_integral = replace(goal, expression=self.z)
        self.assertEqual(verify_derivation({}, nodes, wrong_integral, integral_aggregations=givens,
                                           deterministic_symbols=(self.r,)).status, "inconclusive")

    def test_independent_bernstein_checks_the_exact_threshold_and_exponent(self):
        declaration = Independence((self.x, self.y))
        premises = (BoundedMean(self.x, 3, 1), BoundedMean(self.y, 3, 1))
        threshold = Fraction(3, 2)
        goal = BernsteinTailBound((self.x + self.y) / 2 - 1, threshold, Fraction(1, 2))
        self.assertEqual(independent_bernstein(declaration, premises, threshold), goal)
        self.assertEqual(goal.probability_upper, exp(-Fraction(1, 2)))
        zero = tuple(replace(premise, mean_ceiling=0) for premise in premises)
        self.assertEqual(independent_bernstein(declaration, zero, threshold).probability_upper, 0)
        with self.assertRaises(ValueError):
            independent_bernstein(declaration, premises, 0)
        with self.assertRaises(ValueError):
            independent_bernstein(Independence((self.x, self.z)), premises, threshold)
        node = DerivationNode("b", "independent_bernstein",
                              argument=("mutual", ("mx", "my"), threshold))
        givens = {"bounded_means": dict(zip(("mx", "my"), premises)),
                  "independence": {"mutual": declaration}}
        self.assertEqual(verify_derivation({}, (node,), goal, **givens).status, "verified")
        self.assertEqual(verify_derivation({}, (node,), replace(goal, rate=1), **givens).status,
                         "inconclusive")
