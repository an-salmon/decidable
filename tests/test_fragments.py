import unittest

from sympy import Poly, symbols

from decidable.fragments import (
    FRAGMENTS,
    ExpSeriesClaim,
    GramCertificateClaim,
    KernelDominationClaim,
    LinearImplicationClaim,
    MatrixRankClaim,
    PositiveStateClaim,
    dispatch,
)


class FragmentTests(unittest.TestCase):
    def test_gram_replay_proves_claim_or_leaves_it_unresolved(self):
        verified = dispatch("gram", "verify_gram_certificate", GramCertificateClaim(
            2, {(0, 0): 1, (0, 1): -2, (1, 1): 1}, squares=((1, (1, -1)),),
        ))
        self.assertEqual(verified.status, "verified")
        self.assertIn("2 vectors", verified.scope)
        self.assertIsInstance(verified.witness, GramCertificateClaim)

        rejected = dispatch("gram", "verify_gram_certificate", GramCertificateClaim(
            1, {(0, 0): 1}, squares=((2, (1,)),),
        ))
        self.assertEqual(rejected.status, "inconclusive")

    def test_affine_decision_distinguishes_disproof_from_failed_replay(self):
        result = dispatch("linear", "linear_implies", LinearImplicationClaim(
            1, (0, 1), inequalities=((1, 1),),
        ))
        self.assertEqual(result.status, "disproved")

    def test_routes_reject_unknown_operations_or_wrong_claims(self):
        claim = LinearImplicationClaim(1, (0, 1))
        self.assertEqual(dispatch("gram", "verify_gram_certificate", claim).status, "unsupported")
        self.assertEqual(dispatch("linear", "guess", claim).status, "unsupported")
        self.assertEqual(set(FRAGMENTS), {
            "rational", "noncommutative", "gram", "polynomial", "linear", "series", "linalg",
        })
        self.assertIn("linear_implies", FRAGMENTS["linear"].operations)

    def test_affine_route_rejects_inexact_coefficients(self):
        with self.assertRaises(TypeError):
            dispatch("linear", "linear_implies", LinearImplicationClaim(1, (0, 1.0)))

    def test_specialized_routes_run_the_declared_operations(self):
        z = symbols("z")
        one = Poly(1, z)
        zero = Poly(0, z)
        cases = (
            ("series", "check_kernel_domination",
             KernelDominationClaim((one, Poly(1 - z, z)), (zero, one))),
            ("series", "check_exp_series", ExpSeriesClaim(one, zero)),
            ("linear", "find_positive_state", PositiveStateClaim(((1,),), 1)),
        )
        for fragment, operation, claim in cases:
            with self.subTest(operation=operation):
                self.assertIn(operation, FRAGMENTS[fragment].operations)
                self.assertEqual(dispatch(fragment, operation, claim).status, "verified")

    def test_rank_route_returns_the_exact_value_with_finite_scope(self):
        result = dispatch("linalg", "matrix_rank", MatrixRankClaim(((1, 2), (2, 4))))
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.witness, 1)
        self.assertIn("finite", result.scope)
