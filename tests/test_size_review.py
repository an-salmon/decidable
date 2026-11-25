import unittest
from fractions import Fraction as F
from itertools import combinations
from random import Random
from unittest.mock import patch

from sympy import Float, Matrix, Poly, Pow, exp, symbols

from decidable.exp_series import check_exp_series
from decidable.size_calculus import (
    AverageProblem,
    DerivationNode,
    MomentProblem,
    SizeBound,
    UpperBound,
    check_scaling,
    exp_enclosure,
    grid_optimum,
    norm_weight,
    search_refutation,
    verify_average_refutation,
    verify_derivation,
)
from decidable.size_calculus.pointwise import check_exponential_residual
from decidable.size_calculus.receipts import replay_manifest


def dual_grid_value(inputs, budgets, output, grid):
    """Minimize the affine-majorant cost over vertices of the finite dual LP."""
    dimension = len(inputs) + 1
    inequalities = [
        ((1, *(poly.eval(point) for poly in inputs)), output.eval(point))
        for point in grid
    ]
    inequalities.extend(
        (tuple(int(i == j) for i in range(dimension)), 0)
        for j in range(1, dimension)
    )
    costs = []
    for active in combinations(inequalities, dimension):
        rows, rhs = zip(*active)
        coefficients = Matrix(rows)
        if coefficients.det() == 0:
            continue
        candidate = coefficients.inv() * Matrix(rhs)
        if all(sum(a * w for a, w in zip(row, candidate)) >= bound
               for row, bound in inequalities):
            costs.append(sum(w * b for w, b in zip(candidate, (1, *budgets))))
    return F(min(costs))


class GridOracleTests(unittest.TestCase):
    def test_exact_grid_search_agrees_with_dual_on_small_random_problems(self):
        rng = Random(20261124)
        z = symbols("z")
        grid = (0, F(2, 3), F(5, 4), 2)

        def random_nonnegative_polynomial():
            linear = rng.randint(0, 3) * z - rng.randint(0, 3)
            return Poly(linear**2 + rng.randint(0, 2), z)

        support_sizes = set()
        for trial in range(24):
            inputs = tuple(random_nonnegative_polynomial() for _ in range(trial % 4))
            output = random_nonnegative_polynomial()
            # A specified feasible atom ensures strong duality for the reference LP.
            feasible_point = rng.choice(grid)
            budgets = tuple(F(poly.eval(feasible_point)) + rng.choice((0, F(1, 3), 1))
                            for poly in inputs)
            with self.subTest(trial=trial, budgets=budgets):
                problem = MomentProblem(inputs, budgets, output, 0, upper=2)
                optimum = grid_optimum(problem, (*grid, grid[1]))
                self.assertIsNotNone(optimum)
                self.assertEqual(optimum.value, dual_grid_value(inputs, budgets, output, grid))
                self.assertEqual(optimum.points, grid)
                points, masses = optimum.law.points, optimum.law.masses
                support_sizes.add(len(points))
                self.assertLessEqual(len(points), len(inputs) + 1)
                self.assertTrue(set(points).issubset(grid))
                self.assertTrue(all(mass > 0 for mass in masses))
                self.assertEqual(sum(masses), 1)
                for polynomial, budget in zip(inputs, budgets):
                    self.assertLessEqual(sum(mass * polynomial.eval(point)
                                             for mass, point in zip(masses, points)), budget)
                self.assertEqual(sum(mass * output.eval(point)
                                     for mass, point in zip(masses, points)), optimum.value)
        self.assertTrue(any(size > 1 for size in support_sizes))

    def test_grid_moment_caps_can_be_jointly_infeasible(self):
        z = symbols("z")
        problem = MomentProblem((Poly(z, z), Poly(1 - z, z)), (F(1, 4),) * 2,
                                Poly(z**2, z), 0, upper=1)
        self.assertIsNone(grid_optimum(problem, (0, 1)))
        self.assertEqual(search_refutation(problem, (0, 1)).status, "inconclusive")


class ScalingReviewTests(unittest.TestCase):
    def test_reciprocal_keeps_small_and_large_regimes_separate(self):
        r = symbols("r")
        weights = {r: 1}
        for limit in ("small", "large", "both"):
            with self.subTest(limit=limit):
                result = check_scaling(1 / (1 + r), 1 / r, weights, limit=limit)
                self.assertEqual(result.status, "verified")
                self.assertEqual(result.witness, ((F(0), F(-1)), (F(-1), F(-1))))
        self.assertEqual(check_scaling(1 / (1 + r), 1, weights, limit="both").status,
                         "verified")
        self.assertEqual(check_scaling(1 / r, 1 / (1 + r), weights, limit="small").status,
                         "disproved")
        self.assertEqual(check_scaling(1 / r, 1 / (1 + r), weights, limit="large").status,
                         "verified")

    def test_scaling_cannot_override_literal_numbers_or_float_expressions(self):
        r = symbols("r")
        self.assertEqual(check_scaling(r, F(1, 2), {r: 1}).status, "disproved")
        self.assertEqual(check_scaling(r, Float("0.5") * r, {r: 1}).status, "unsupported")
        for invalid in ({r: 1, 1: 1}, {r: 1, Float("0.5") * r: 1}, {r: 1.0}):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                check_scaling(r, 1, invalid)


class PointwiseReviewTests(unittest.TestCase):
    def test_optional_sample_budget_cannot_block_global_e1_proof(self):
        v = symbols("v", real=True)
        result = check_exponential_residual(1 + exp(v), v, sample=30, degree=16)
        self.assertEqual(result.status, "verified")

    def test_pointwise_coordinate_domain_is_owned_by_the_checker(self):
        for assumptions in ({"real": False}, {"positive": True}, {"zero": True}):
            with self.subTest(assumptions=assumptions), self.assertRaises(ValueError):
                v = symbols("v", **assumptions)
                check_exponential_residual(exp(v), v)

    def test_denominator_checks_include_tangent_and_algebraic_zeros(self):
        v = symbols("v")
        tangent = Pow(1 - 2 * v + v**2, -1, evaluate=False)
        self.assertEqual(check_exponential_residual(tangent, v, upper=1).status, "verified")
        self.assertEqual(check_exponential_residual(tangent, v, upper=2).status, "unsupported")

        algebraic = Pow(2 - v**2, -1, evaluate=False)
        self.assertEqual(check_exponential_residual(algebraic, v, upper=F(7, 5)).status,
                         "verified")
        self.assertEqual(check_exponential_residual(algebraic, v, upper=F(10, 7)).status,
                         "unsupported")


class DraftFixtureReviewTests(unittest.TestCase):
    def test_bounded_raise_needs_its_premise_and_descent_keeps_the_scale(self):
        r = symbols("r", positive=True)
        x = symbols("X", nonnegative=True)
        given = DerivationNode("input", "given", argument="X")
        source = {"X": SizeBound(x, F(1, 2), r, True)}
        raise_node = DerivationNode("raise", "bounded_raise", ("input",), (F(1, 4), "X <= 1"))
        target = SizeBound(x, F(1, 4), r**F(1, 2), True)
        self.assertEqual(verify_derivation(source, (given, raise_node), target).status,
                         "inconclusive")
        self.assertEqual(verify_derivation(source, (given, raise_node), target,
                                           upper_bounds={"X <= 1": UpperBound(x, 1)}).status,
                         "verified")

        descent = DerivationNode("descent", "descend", ("input",), 1)
        wrong_scale = SizeBound(x, 1, r**2, True)
        self.assertEqual(verify_derivation(source, (given, descent), wrong_scale).status,
                         "inconclusive")
        # At r = 1/2, the constant X = r/2 makes the proposed budget fail.
        self.assertLessEqual(exp_enclosure(F(1, 4), 4)[1], 2)
        self.assertGreater(exp_enclosure(1, 4)[0], 2)

    def test_unit_factor_independent_bernoulli_average_is_refuted(self):
        problem = AverageProblem(F(1, 2), (1, 1), (F(1, 2), F(1, 2)), 1)
        points = ((0, 0), (0, 1), (1, 0), (1, 1))
        masses = (F(9, 49), F(12, 49), F(12, 49), F(16, 49))
        result = verify_average_refutation(problem, points, masses)
        self.assertEqual(result.status, "disproved")
        self.assertEqual(result.witness, (points, masses))

    def test_bernstein_control_fails_coefficient_and_pointwise_checks(self):
        v = symbols("v")
        p, q = Poly(2 * v - 8, v), Poly(2 * v**2 + 6 * v + 8, v)
        self.assertEqual(check_exp_series(p, q, tail_start="auto").status, "inconclusive")

        point = F(1, 10)
        incorrect_bound = 1 + point + point**2 / (2 * (1 - point / 4))
        self.assertGreater(exp_enclosure(point, 4)[0], incorrect_bound)

        # Without the positive-denominator guard the original bound fails too.
        outside = F(4)
        unguarded_bound = 1 + outside + outside**2 / (2 * (1 - outside / 3))
        self.assertLess(unguarded_bound, 0)

    def test_parabolic_forcing_weight_depends_on_norm_convention(self):
        r, solution, forcing = symbols("r solution forcing")
        for normalized in (False, True):
            with self.subTest(normalized=normalized):
                solution_weight = norm_weight(0, F(1, 2), 3, normalized=normalized)
                forcing_space_weight = norm_weight(-2, F(1, 2), 3, normalized=normalized,
                                                   derivative_order=-1)
                forcing_weight = norm_weight(forcing_space_weight, F(1, 2), 2,
                                             normalized=normalized)
                weights = {r: 1, solution: solution_weight, forcing: forcing_weight}
                rhs = r * forcing if normalized else forcing
                opposite_reading = forcing if normalized else r * forcing
                self.assertEqual(check_scaling(solution, rhs, weights).status, "verified")
                self.assertEqual(check_scaling(solution, opposite_reading, weights).status,
                                 "disproved")

    def test_unexpected_replay_exception_cannot_yield_a_proof_receipt(self):
        entry = {
            "id": "interrupted",
            "problem": {
                "version": 1,
                "coordinates": ["z"],
                "support": ["le", "0", ["ref", "z"]],
                "inputs": [],
                "output": {"integrand": "0", "budget": "0"},
            },
            "certificate": {"kind": "majorant", "c": "0", "lambda": []},
        }
        self.assertEqual(replay_manifest([entry])[0]["verdict"], "CERTIFIED")
        with patch("decidable.size_calculus.receipts.verify_majorant",
                   side_effect=RuntimeError("replay interrupted")):
            receipt, = replay_manifest([entry])
        self.assertEqual(receipt["verdict"], "FAIL")
        self.assertIn("replay interrupted", receipt["reason"])
