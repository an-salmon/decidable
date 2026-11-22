from fractions import Fraction
from unittest import TestCase

from decideable.linear import linear_feasible, linear_implies


class LinearFeasibilityTests(TestCase):
    def test_closed_bounds_can_meet(self):
        self.assertTrue(linear_feasible(1, inequalities=[(0, 1), (0, -1)]))

    def test_strict_bounds_cannot_meet(self):
        self.assertFalse(
            linear_feasible(1, inequalities=[(0, -1)], strict_inequalities=[(0, 1)])
        )

    def test_preserves_strict_copy_of_proportional_row(self):
        self.assertFalse(
            linear_feasible(
                1, inequalities=[(1, 1), (-1, -1)], strict_inequalities=[(2, 2)]
            )
        )

    def test_exact_rational_open_interval(self):
        lower = Fraction(1, 3)
        upper = lower + Fraction(1, 10**40)
        self.assertTrue(
            linear_feasible(1, strict_inequalities=[(-lower, 1), (upper, -1)])
        )
        self.assertFalse(linear_feasible(1, inequalities=[(-upper, 1), (lower, -1)]))

    def test_strict_cycle_across_multiple_variables(self):
        self.assertFalse(
            linear_feasible(
                3,
                inequalities=[(0, 1, -1, 0), (0, 0, 1, -1)],
                strict_inequalities=[(0, -1, 0, 1)],
            )
        )

    def test_one_sided_variable_does_not_drop_other_constraints(self):
        self.assertFalse(
            linear_feasible(2, inequalities=[(0, 0, 1), (-1, 1, 0), (0, -1, 0)])
        )
        self.assertTrue(linear_feasible(2, strict_inequalities=[(0, 0, 1), (0, -1, 0)]))

    def test_zero_variables_and_empty_system(self):
        self.assertTrue(linear_feasible(0))
        self.assertTrue(linear_feasible(5))
        self.assertTrue(
            linear_feasible(0, inequalities=[(0,)], strict_inequalities=[(1,)])
        )
        self.assertFalse(linear_feasible(0, inequalities=[(-1,)]))
        self.assertFalse(linear_feasible(0, strict_inequalities=[(0,)]))

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            linear_feasible(True)
        with self.assertRaises(ValueError):
            linear_feasible(-1)
        with self.assertRaises(ValueError):
            linear_feasible(1, inequalities=[(0,)])
        with self.assertRaises(TypeError):
            linear_feasible(1, strict_inequalities=[b"\x00\x01"])
        with self.assertRaises(TypeError):
            linear_feasible(1, inequalities=[(0, 0.5)])
        with self.assertRaises(TypeError):
            linear_feasible(1, strict_inequalities=[(0, True)])


class LinearImplicationTests(TestCase):
    def test_nonnegative_sum(self):
        assumptions = [(0, 1, 0), (0, 0, 1)]
        self.assertTrue(linear_implies(2, (0, 1, 1), inequalities=assumptions))
        self.assertFalse(linear_implies(2, (0, 1, -1), inequalities=assumptions))

    def test_equation_encoded_by_two_inequalities(self):
        equation = [(0, 1, -1), (0, -1, 1)]
        assumptions = [*equation, (0, 1, 0)]
        self.assertTrue(linear_implies(2, (0, 0, 1), inequalities=assumptions))

    def test_strict_hypothesis_with_nonstrict_target(self):
        self.assertTrue(linear_implies(1, (0, 1), strict_inequalities=[(0, 1)]))
        self.assertFalse(linear_implies(1, (-1, 1), strict_inequalities=[(0, 1)]))

    def test_vacuous_implication_and_constant_targets(self):
        self.assertTrue(linear_implies(0, (-1,), strict_inequalities=[(0,)]))
        self.assertTrue(linear_implies(0, (0,)))
        self.assertFalse(linear_implies(0, (-1,)))

    def test_checks_target_even_with_infeasible_hypotheses(self):
        with self.assertRaises(TypeError):
            linear_implies(0, (0.0,), inequalities=[(-1,)])
