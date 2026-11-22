from fractions import Fraction
from importlib.util import find_spec
from unittest import TestCase, skipUnless
from unittest.mock import patch

from decidable.gram import verify_gram_certificate
from decidable.search import search_gram_certificate


class GramSearchTests(TestCase):
    def assert_verified(self, vector_count, target, certificate):
        self.assertIsNotNone(certificate)
        self.assertTrue(
            verify_gram_certificate(
                vector_count,
                target,
                equations=certificate.equations,
                inequalities=certificate.inequalities,
                squares=certificate.squares,
            )
        )

    def test_exact_signed_equality_multiplier(self):
        hypothesis = {(0, 0): 1, (1, 1): -1}
        target = {(0, 0): Fraction(-2, 3), (1, 1): Fraction(2, 3)}
        certificate = search_gram_certificate(2, target, equations=(hypothesis,))
        self.assert_verified(2, target, certificate)
        self.assertEqual(certificate.equations, ((Fraction(-2, 3), hypothesis),))

    def test_exact_linear_solve_for_underdetermined_system(self):
        hypothesis = {(0, 0): 1, (1, 1): -1}
        target = {(0, 0): -1, (1, 1): 1}
        certificate = search_gram_certificate(2, target, equations=(hypothesis, hypothesis))
        self.assert_verified(2, target, certificate)

    def test_exact_nonnegative_inequality_multiplier(self):
        hypothesis = {(0, 0): -2}
        target = {(0, 0): -1}
        certificate = search_gram_certificate(1, target, inequalities=(hypothesis,))
        self.assert_verified(1, target, certificate)
        self.assertEqual(certificate.inequalities, ((Fraction(1, 2), hypothesis),))

    def test_exact_factorization_with_zero_pivot_and_nonsquare_weight(self):
        target = {(1, 1): 2, (1, 2): -4, (2, 2): 2}
        certificate = search_gram_certificate(3, target)
        self.assert_verified(3, target, certificate)
        self.assertEqual(certificate.squares, ((2, (0, 1, -1)),))

    def test_zero_pivot_with_nonzero_cross_term_is_indefinite(self):
        self.assertIsNone(search_gram_certificate(2, {(0, 1): 1}))

    def test_inexact_solver_proposal_is_rounded_and_verified(self):
        target = {(0, 0): -1, (1, 1): 4}
        hypothesis = {(0, 0): 1, (0, 1): 2}
        with patch("decidable.search._solve_sdp", return_value=(-2.0008,)):
            certificate = search_gram_certificate(2, target, equations=(hypothesis,))
        self.assert_verified(2, target, certificate)
        self.assertEqual(certificate.equations, ((-2, hypothesis),))

    def test_invalid_solver_proposal_is_inconclusive(self):
        target = {(0, 0): -1}
        hypothesis = {(0, 0): 2}
        with patch("decidable.search._solve_sdp", return_value=(-0.5,)):
            self.assertIsNone(search_gram_certificate(1, target, inequalities=(hypothesis,)))

    def test_optional_solver_may_be_absent(self):
        target = {(0, 0): 1, (1, 1): -1}
        hypothesis = {(0, 0): -1, (1, 1): -1}
        with patch.dict("sys.modules", {"cvxpy": None}):
            self.assertIsNone(search_gram_certificate(2, target, inequalities=(hypothesis,)))

    @skipUnless(find_spec("cvxpy"), "optional CVXPY is unavailable")
    def test_searches_for_an_inequality_multiplier_and_psd_residual(self):
        target = {(0, 0): 1, (1, 1): -1}
        hypothesis = {(0, 0): -1, (1, 1): -1}
        certificate = search_gram_certificate(2, target, inequalities=(hypothesis,))
        self.assert_verified(2, target, certificate)

    def test_malformed_inputs_raise(self):
        with self.assertRaises(TypeError):
            search_gram_certificate(1, {}, equations=({(0, 0): 0.0},))
        with self.assertRaises(ValueError):
            search_gram_certificate(1, {}, inequalities=({(1, 1): 0},))
        with self.assertRaises(ValueError):
            search_gram_certificate(1, {}, max_denominator=0)
