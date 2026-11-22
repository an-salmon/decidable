from fractions import Fraction

import unittest
from sympy import Poly, QQ, symbols

from decidable.gram import verify_gram_certificate
from decidable.reductions import expand_gram_inner_product, reduce_spectral_quadratic_form


class ReductionTests(unittest.TestCase):
    def test_gram_expansion_uses_symmetric_pairs(self):
        left = (Fraction(1, 2), 2)
        right = (2, -1)
        expected = {(0, 0): Fraction(1), (0, 1): Fraction(7, 2), (1, 1): Fraction(-2)}
        assert expand_gram_inner_product(left, right) == expected
        assert expand_gram_inner_product(right, left) == expected
        assert verify_gram_certificate(2, expand_gram_inner_product(left, left), squares=((1, left),))
        with self.assertRaisesRegex(ValueError, "same number"):
            expand_gram_inner_product((1,), (1, 2))


    def test_finite_support_spectral_reduction_is_exact(self):
        x = symbols("x")
        polynomial = Poly(x**2 - 1, x, domain=QQ)
        passing = reduce_spectral_quadratic_form(polynomial, (-1, Fraction(3, 2), -1))
        assert passing.samples == ((Fraction(-1), Fraction(0)), (Fraction(3, 2), Fraction(5, 4)))
        assert passing.nonnegative_on_support
        assert not reduce_spectral_quadratic_form(polynomial, (Fraction(1, 2),)).nonnegative_on_support
        with self.assertRaises(TypeError):
            reduce_spectral_quadratic_form(polynomial, (1.0,))
