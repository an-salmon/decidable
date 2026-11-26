import unittest
from fractions import Fraction

from sympy import Add, Mul, Pow, Rational, Symbol, exp, log, oo, symbols

from decidable import PositiveParameters, quadrature_value, verify_quadrature
from decidable.report import Obligation, Report


class QuadratureTests(unittest.TestCase):
    def test_rational_powers_at_finite_and_improper_endpoints(self):
        self.assertEqual(quadrature_value(0, Fraction(1, 3), 1), Rational(2, 3))
        self.assertEqual(quadrature_value(Fraction(-1, 2), 0, 4), 4)
        self.assertEqual(quadrature_value(Fraction(-3, 2), 4, None), 1)
        self.assertEqual(verify_quadrature(0, Fraction(1, 3), 1, Fraction(2, 3)).status,
                         "verified")

    def test_logarithmic_primitive_and_oriented_finite_bounds(self):
        self.assertEqual(quadrature_value(-1, 1, 2), log(2))
        self.assertEqual(quadrature_value(-1, 2, 1), -log(2))
        self.assertEqual(verify_quadrature(-1, 2, 2, 0).status, "verified")

    def test_each_improper_endpoint_must_converge(self):
        for exponent, lower, upper in ((-2, 0, 1), (-1, 0, 1), (-1, 1, None),
                                        (0, 1, None), (-2, 0, None), (0, 0, None)):
            with self.subTest(exponent=exponent, lower=lower, upper=upper):
                self.assertEqual(verify_quadrature(exponent, lower, upper, 0).status,
                                 "disproved")
        self.assertEqual(verify_quadrature(0, 1, None, oo).status, "unsupported")

    def test_inverse_exponential_primitive_and_endpoint_limits(self):
        self.assertEqual(quadrature_value(-2, 0, None, decay=2), Rational(1, 2))
        finite_claim = (2 * exp(-1) - 3 * exp(-2)) / 4
        self.assertEqual(verify_quadrature(-3, 1, 2, finite_claim, decay=2).status,
                         "verified")
        self.assertEqual(verify_quadrature(-3, 0, 2, exp(-1) / 2, decay=2).status,
                         "verified")
        self.assertEqual(verify_quadrature(-3, 2, None, Rational(1, 4) - exp(-1) / 2,
                                           decay=2).status, "verified")
        self.assertEqual(verify_quadrature(-5, 0, None, Fraction(3, 8), decay=2).status,
                         "verified")
        self.assertEqual(verify_quadrature(-3, 1, 2, finite_claim + 1, decay=2).status,
                         "inconclusive")

    def test_positive_parameter_scales_and_decay(self):
        p, gap = symbols("p gap")
        parameters = PositiveParameters((p, gap))
        lower = parameters.field.from_sympy(p)
        upper = parameters.field.from_sympy(p + gap)
        decay = parameters.field.from_sympy(p**2 / 4)
        claimed = 4 * (exp(-p**2 / (4 * (p + gap))) - exp(-p / 4)) / p**2

        result = verify_quadrature(-2, lower, upper, claimed, decay=decay,
                                   parameters=parameters)
        self.assertEqual(result.status, "verified")
        self.assertEqual(verify_quadrature(-1, lower, upper, log(p + gap) - log(p),
                                           parameters=parameters).status, "verified")

    def test_unsupported_integrands_and_endpoints(self):
        cases = (
            (-1.0, 1, 2, {}),
            (-1, 1.0, 2, {}),
            (-1, -1, 2, {}),
            (-1, 1, 0, {}),
            (-1, None, None, {}),
            (-1, 1, 2, {"decay": 1}),
            (Fraction(-5, 2), 1, 2, {"decay": 1}),
            (-2, 1, 2, {"decay": 0}),
        )
        for exponent, lower, upper, options in cases:
            with self.subTest(exponent=exponent, lower=lower, upper=upper, options=options):
                self.assertEqual(verify_quadrature(exponent, lower, upper, 0, **options).status,
                                 "unsupported")

        p, q = symbols("p q")
        parameters = PositiveParameters((p, q))
        uncertain = parameters.field.from_sympy(p - q)
        pole = parameters.field.from_sympy(1 / (p - q))
        self.assertEqual(verify_quadrature(0, 0, uncertain, 1, parameters=parameters).status,
                         "unsupported")
        self.assertEqual(verify_quadrature(-2, 0, None, 1, decay=pole,
                                           parameters=parameters).status, "unsupported")

    def test_claim_domain_is_checked_before_formal_cancellation(self):
        p, q = symbols("p q")
        parameters = PositiveParameters((p, q))
        reciprocal = Pow(p - q, -1, evaluate=False)
        cancelled_pole = Add(reciprocal, Mul(-1, reciprocal, evaluate=False), 1,
                             evaluate=False)
        for claimed in (1.0, Symbol("foreign", positive=True), cancelled_pole,
                        log(-p, evaluate=False), Pow(p - q, Rational(1, 2), evaluate=False)):
            with self.subTest(claimed=claimed):
                self.assertEqual(verify_quadrature(0, 1, 2, claimed,
                                                   parameters=parameters).status, "unsupported")

    def test_unevaluated_unit_power_is_an_exact_claim(self):
        claimed = Pow(1, Rational(1, 2), evaluate=False)
        self.assertEqual(verify_quadrature(0, 1, 2, claimed).status, "verified")

    def test_cancelled_exponential_argument_keeps_original_domain_proof(self):
        p = Symbol("p")
        parameters = PositiveParameters((p,))
        upper = parameters.field.from_sympy(p**3 + 1)
        decay = parameters.field.from_sympy(p + 1)
        value = quadrature_value(-2, 0, upper, decay=decay, parameters=parameters)
        self.assertEqual(verify_quadrature(-2, 0, upper, value, decay=decay,
                                           parameters=parameters).status, "verified")

    def test_declared_positive_parameter_logarithm(self):
        p = Symbol("p")
        parameters = PositiveParameters((p,))
        upper = parameters.field.from_sympy(p**3)
        self.assertEqual(verify_quadrature(-1, 1, upper, 3 * log(p),
                                           parameters=parameters).status, "verified")
        nonnatural_log = log(p**3, 2, evaluate=False)
        self.assertEqual(verify_quadrature(-1, 1, upper, nonnatural_log,
                                           parameters=parameters).status, "unsupported")

    def test_report_can_record_symbolic_quadrature(self):
        report = Report([
            Obligation("inverse-scale integral",
                       lambda: verify_quadrature(-2, 0, None, Fraction(1, 2), decay=2)),
        ])
        (outcome,) = report.run()
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.result.status, "verified")
