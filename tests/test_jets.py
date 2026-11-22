from fractions import Fraction

import unittest
from sympy import Symbol, exp, symbols

from decidable.jets import Jet, JetSpace, check_jet_identity, total_derivative, two_scale_derivative


class JetTests(unittest.TestCase):
    def test_total_derivatives_commute_with_explicit_coordinates_and_parameters(self):
        x, y, a, u, ux, uy, uxy = symbols("x y a u ux uy uxy")
        space = JetSpace(
            (x, y),
            {Jet("u", (0, 0)): u, Jet("u", (1, 0)): ux,
             Jet("u", (0, 1)): uy, Jet("u", (1, 1)): uxy},
            parameters=(a,),
        )
        expression = a * x * y * u
        assert total_derivative(space, expression, 0) == a * x * y * ux + a * y * u
        assert check_jet_identity(
            space, expression, expression, lhs_derivatives=(0, 1), rhs_derivatives=(1, 0)
        ).status == "verified"


    def test_relation_membership_and_inconclusive_remainder(self):
        x, u, ux = symbols("x u ux")
        space = JetSpace((x,), {Jet("u", (0,)): u, Jet("u", (1,)): ux})
        assert check_jet_identity(
            space, u, x, lhs_derivatives=(0,), relations=(ux - x,)
        ).status == "verified"
        result = check_jet_identity(space, u, relations=(u**2,))
        assert result.status == "inconclusive"
        assert result.witness == u
        assert str(u) in result.scope


    def test_two_scale_derivative_and_localized_ideal(self):
        x, y, epsilon, u, ux, uy = symbols("x y epsilon u ux uy")
        space = JetSpace(
            (x,),
            {Jet("u", (0,), (0,)): u, Jet("u", (1,), (0,)): ux, Jet("u", (0,), (1,)): uy},
            fast_coordinates=(y,),
            epsilon=epsilon,
        )
        assert two_scale_derivative(space, x + 2 * y + u, 0) == 1 + ux + (2 + uy) / epsilon
        assert check_jet_identity(
            space, u, 0, relations=(epsilon**2 * ux, uy / epsilon),
            lhs_derivatives=(0,), two_scale=True,
        ).status == "verified"
        assert check_jet_identity(space, u / epsilon, relations=(u / epsilon**2,)).status == "verified"


    def test_invalid_jet_ring_inputs(self):
        x, epsilon, u = symbols("x epsilon u")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            Jet("u", (-1,))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            JetSpace((x,), {Jet("u", (0, 0)): u})
        with self.assertRaisesRegex(ValueError, "distinct names"):
            JetSpace((x,), {}, parameters=(Symbol("x", positive=True),))
        with self.assertRaisesRegex(ValueError, "declared together"):
            JetSpace((x,), {}, epsilon=epsilon)
        space = JetSpace((x,), {Jet("u", (0,)): u})
        assert check_jet_identity(space, Fraction(2, 3), 2 * Fraction(1, 3)).status == "verified"
        for expression in (1 / x, exp(x), x + epsilon, 0.5 * x):
            with self.assertRaises(ValueError):
                check_jet_identity(space, expression)
        with self.assertRaisesRegex(ValueError, "undeclared jet"):
            total_derivative(space, u, 0)
