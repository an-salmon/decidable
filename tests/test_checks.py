import unittest

from sympy import Add, Mul, Pow, Rational, symbols

from decidable.checks import loglinear_zero
from decidable.jets import Jet, JetSpace, check_jet_identity, total_derivative


class ProofCheckTests(unittest.TestCase):
    def test_nested_scale_powers(self):
        epsilon, alpha, d, p = symbols("epsilon alpha d p")
        exponent = (p - 2) / (2 * p)
        nested = Pow(Pow(epsilon, d, evaluate=False), exponent, evaluate=False)
        for offset, expected in ((0, True), (1, False)):
            with self.subTest(offset=offset):
                residual = Mul(
                    Pow(epsilon, alpha - 1, evaluate=False),
                    nested,
                    Pow(epsilon, -(alpha - 1 + d * exponent) + offset, evaluate=False),
                    evaluate=False,
                )
                self.assertIs(
                    loglinear_zero(residual, positive_bases=(epsilon,),
                                   exponent_parameters=(alpha, d, p)),
                    expected,
                )

    def test_positive_rational_bases(self):
        a = symbols("a")
        expression = Mul(Pow(Rational(2, 3), a), Pow(3, a), Pow(2, -a), evaluate=False)
        self.assertTrue(loglinear_zero(expression, exponent_parameters=(a,)))

    def test_rejects_nonpositive_or_unproved_bases_before_cancellation(self):
        x, y, a, b = symbols("x y a b")
        difference = x - y
        expression = Mul(
            Pow(Pow(difference, a, evaluate=False), b, evaluate=False),
            Pow(difference, -a * b, evaluate=False),
            evaluate=False,
        )
        for invalid in (expression, Pow(y, 0, evaluate=False)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    loglinear_zero(invalid, positive_bases=(x,), exponent_parameters=(a, b))
        with self.assertRaises(ValueError):
            loglinear_zero(expression, positive_bases=(x, y), exponent_parameters=(a, b))

    def test_rejects_an_identically_zero_exponent_denominator(self):
        x, p = symbols("x p")
        exponent = Pow(Add(p, -p, evaluate=False), -1, evaluate=False)
        with self.assertRaisesRegex(ValueError, "identically zero"):
            loglinear_zero(Pow(x, exponent, evaluate=False), positive_bases=(x,),
                           exponent_parameters=(p,))

    def test_replays_a_supplied_jet_certificate(self):
        for dimension in (2, 3, 4):
            with self.subTest(dimension=dimension):
                coordinates = symbols(f"x0:{dimension}")
                u, ux, uxx = symbols("u ux uxx")
                tail = (0,) * (dimension - 1)
                space = JetSpace(
                    coordinates,
                    {Jet("u", (order,) + tail): value
                     for order, value in enumerate((u, ux, uxx))},
                )
                relation = ux - u
                target = u * uxx - u**2
                certificate = u * (total_derivative(space, relation, 0) + relation)
                self.assertEqual(check_jet_identity(space, target - certificate).status, "verified")
                self.assertEqual(
                    check_jet_identity(space, target - certificate + ux).status, "inconclusive"
                )
