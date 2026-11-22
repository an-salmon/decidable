import unittest

from sympy import Poly, QQ, Rational, symbols

from decidable.kernel_domination import check_kernel_domination


class KernelDominationTests(unittest.TestCase):
    def setUp(self):
        self.z = symbols("z")

    def kernel(self, numerator, denominator=1):
        return Poly(numerator, self.z), Poly(denominator, self.z)

    def test_mixed_sign_tail_with_polynomial_part(self):
        z = self.z
        upper = self.kernel(4 - z / 2, 1 - z / 2)
        lower = self.kernel(4, 1 - z / 4)
        result = check_kernel_domination(upper, lower)
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.witness["tail_start"], 1)
        self.assertEqual(result.witness["leading_amplitude"], 3)
        self.assertEqual(result.witness["negative_bound"], 2)

    def test_negative_prefix_returns_an_exact_index(self):
        z = self.z
        upper = self.kernel(3, 1 - z / 2)
        lower = self.kernel(4, 1 - z / 4)
        result = check_kernel_domination(upper, lower, tail_start=1)
        self.assertEqual(result.status, "disproved")
        self.assertEqual(result.witness, {"index": 0, "coefficient": -1})

    def test_inadequate_bound_can_be_improved_by_larger_cutoff(self):
        z = self.z
        upper = self.kernel(1, 1 - z)
        # The difference is 1 - 2/2**n + 1/4**n at each index.
        lower = self.kernel(2 * (1 - z / 4) - (1 - z / 2), (1 - z / 2) * (1 - z / 4))
        self.assertEqual(check_kernel_domination(upper, lower).status, "inconclusive")
        result = check_kernel_domination(upper, lower, tail_start=1)
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.witness["negative_bound"], 1)

    def test_automatic_cutoff_proves_a_nonnegative_sequence(self):
        z = self.z
        ratio = Rational(3, 4)
        # The difference is (1 - 2 * ratio**n)**2 at each index.
        upper_den = (1 - z) * (1 - ratio**2 * z)
        upper = self.kernel(1 - ratio**2 * z + 4 * (1 - z), upper_den)
        lower = self.kernel(4, 1 - ratio * z)
        self.assertEqual(check_kernel_domination(upper, lower).status, "inconclusive")
        result = check_kernel_domination(upper, lower, tail_start=None)
        self.assertEqual(result.status, "verified")
        self.assertGreaterEqual(result.witness["tail_start"], 5)

    def test_automatic_cutoff_finds_a_delayed_negative_coefficient(self):
        z = self.z
        ratio = Rational(3, 4)
        upper = self.kernel(4, 1 - ratio * z)
        lower = self.kernel(1, 1 - z)
        self.assertEqual(check_kernel_domination(upper, lower, tail_start=5).status, "inconclusive")
        result = check_kernel_domination(upper, lower, tail_start=None)
        self.assertEqual(result.status, "disproved")
        index = result.witness["index"]
        self.assertGreaterEqual(index, 5)
        self.assertEqual(result.witness["coefficient"], 4 * ratio**index - 1)
        self.assertLess(result.witness["coefficient"], 0)

    def test_unsupported_poles_after_exact_cancellation(self):
        z = self.z
        for denominator in ((1 - z)**2, 1 + z, 1 + z**2, 1 - 2 * z**2):
            with self.subTest(denominator=denominator):
                upper = self.kernel(1, denominator)
                lower = self.kernel(0)
                self.assertEqual(check_kernel_domination(upper, lower).status, "unsupported")
                self.assertEqual(check_kernel_domination(upper, upper).status, "verified")

    def test_polynomial_difference_retains_highest_coefficient(self):
        z = self.z
        shared = (1 - z)**2
        upper = self.kernel(1 - shared * z**5 / 3, shared)
        lower = self.kernel(1, shared)
        result = check_kernel_domination(upper, lower)
        self.assertEqual(result.status, "disproved")
        self.assertEqual(result.witness, {"index": 5, "coefficient": Rational(-1, 3)})

    def test_malformed_inputs(self):
        z = self.z
        unit = self.kernel(1)
        with self.assertRaises(ValueError):
            check_kernel_domination(self.kernel(1, z), unit)
        with self.assertRaises(ValueError):
            check_kernel_domination(unit, (Poly(1, z, symbols("y"), domain=QQ), unit[1]))
        with self.assertRaises(ValueError):
            check_kernel_domination(unit, unit, tail_start=-1)
        with self.assertRaises(TypeError):
            check_kernel_domination(unit, unit, tail_start=True)
