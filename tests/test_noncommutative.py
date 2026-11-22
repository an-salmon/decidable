from fractions import Fraction
from unittest import TestCase

from decidable.noncommutative import bilinear_identity, noncommutative_identity


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


class BilinearIdentityTests(TestCase):
    def test_symmetric_symbol_allows_exchange_of_vectors(self):
        self.assertIs(
            bilinear_identity(
                {("p", ("S",), "q"): 1},
                {("q", ("S",), "p"): 1},
                adjoints={"S": "S"},
            ),
            True,
        )

    def test_confluent_forms_distinguish_nonsymmetric_symbol(self):
        self.assertIs(
            bilinear_identity(
                {("p", ("A",), "q"): 1},
                {("q", ("A",), "p"): 1},
                adjoints={"A": "At", "At": "A"},
            ),
            False,
        )

    def test_mixed_cross_terms_in_a_squared_norm_combine(self):
        self.assertIs(
            bilinear_identity(
                {("p", ("B", "A"), "q"): 1, ("q", ("At", "B"), "p"): 1},
                {("p", ("B", "A"), "q"): 2},
                adjoints={"A": "At", "At": "A", "B": "B"},
            ),
            True,
        )

    def test_transposed_rules_used_in_same_scalar_orientation(self):
        self.assertIs(
            bilinear_identity(
                {("p", ("At", "B"), "q"): 1},
                {("p", ("Ut",), "q"): 1},
                adjoints={"A": "At", "At": "A", "B": "B", "U": "Ut", "Ut": "U"},
                rules=[(("B", "A"), ("U",))],
            ),
            True,
        )

    def test_confluence_checked_after_adding_transposed_rules(self):
        self.assertIsNone(
            bilinear_identity(
                {("p", ("X", "A"), "q"): 1},
                {("p", ("A", "X"), "q"): 1},
                adjoints={"A": "A", "B": "B", "X": "X"},
                rules=[(("A", "B"), ("X",))],
            )
        )

    def test_unit_and_exact_coefficients(self):
        half = Fraction(1, 2)
        self.assertIs(
            bilinear_identity(
                {("p", (), "q"): half, ("q", (), "p"): half},
                {("p", (), "q"): 1},
                adjoints={},
            ),
            True,
        )

    def test_rejects_malformed_adjoint_declarations(self):
        with self.assertRaises(ValueError):
            bilinear_identity({}, {}, adjoints={"A": "At"})
        with self.assertRaises(ValueError):
            bilinear_identity({("p", ("A",), "q"): 0}, {}, adjoints={})
        with self.assertRaises(ValueError):
            bilinear_identity({}, {}, adjoints={}, rules=[(("A",), ())])
        with self.assertRaises(TypeError):
            bilinear_identity({("", (), "q"): 1}, {}, adjoints={})
