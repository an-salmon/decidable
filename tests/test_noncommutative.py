from fractions import Fraction
from unittest import TestCase

from decideable.noncommutative import noncommutative_identity


class NoncommutativeIdentityTests(TestCase):
    def test_word_order_matters(self):
        result = noncommutative_identity({("A", "B"): 1}, {("B", "A"): 1})
        self.assertIs(result, False)

    def test_rational_coefficients_of_reduced_words_combine(self):
        self.assertIs(
            noncommutative_identity(
                {("P", "P"): Fraction(1, 2), ("P",): Fraction(1, 2), (): 1},
                {("P",): 1, (): 1},
                rules=[(("P", "P"), ("P",))],
            ),
            True,
        )

    def test_confluent_overlaps_allow_inequality(self):
        rules = [(("A", "a"), ()), (("a", "A"), ())]
        result = noncommutative_identity({("A", "a", "A"): 1}, {("A",): 1}, rules=rules)
        self.assertIs(result, True)
        self.assertIs(noncommutative_identity({("A",): 1}, {(): 1}, rules=rules), False)

    def test_nonconfluent_overlap_allows_only_positive_result(self):
        rules = [(("A", "B"), ("X",)), (("B", "C"), ("Y",))]
        left = {("X", "C"): 1}
        self.assertIsNone(noncommutative_identity(left, {("A", "Y"): 1}, rules=rules))
        result = noncommutative_identity({("A", "B", "C"): 1}, left, rules=rules)
        self.assertIs(result, True)

    def test_detects_containment_overlap(self):
        rules = [(("A", "B", "C"), ("X",)), (("B",), ())]
        result = noncommutative_identity({("X",): 1}, {("A", "C"): 1}, rules=rules)
        self.assertIsNone(result)

    def test_detects_self_overlap(self):
        rules = [(("A", "B", "A"), ("X",))]
        result = noncommutative_identity(
            {("X", "B", "A"): 1}, {("A", "B", "X"): 1}, rules=rules
        )
        self.assertIsNone(result)

    def test_rejects_malformed_input(self):
        with self.assertRaises(TypeError):
            noncommutative_identity({("A",): 0.0}, {})
        with self.assertRaises(TypeError):
            noncommutative_identity({("A",): True}, {})
        with self.assertRaises(TypeError):
            noncommutative_identity({("",): 0}, {})
        with self.assertRaises(TypeError):
            noncommutative_identity([], {})
        with self.assertRaises(ValueError):
            noncommutative_identity({}, {}, rules=[(("A",), ("B",))])
