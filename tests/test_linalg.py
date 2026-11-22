from fractions import Fraction
from unittest import TestCase

from decidable.linalg import matrix_rank


class MatrixRankTests(TestCase):
    def test_rank_deficient_square_matrix(self):
        self.assertEqual(matrix_rank([[1, 2, 3], [2, 4, 6], [0, 1, 1]]), 2)

    def test_pivot_swaps_and_skips_zero_columns(self):
        self.assertEqual(matrix_rank(((0, 0, 1), (0, 1, 0))), 2)

    def test_tall_rectangular_matrix(self):
        self.assertEqual(matrix_rank([[1, 2], [1, 1], [0, 0]]), 2)

    def test_exact_fractional_dependence_and_near_dependence(self):
        third = Fraction(1, 3)
        self.assertEqual(matrix_rank([[1, third], [3, 1]]), 1)
        self.assertEqual(matrix_rank([[1, third], [3, 1 + Fraction(1, 10**40)]]), 2)

    def test_empty_and_zero_matrices(self):
        self.assertEqual(matrix_rank([]), 0)
        self.assertEqual(matrix_rank([[], []]), 0)
        self.assertEqual(matrix_rank([[0, 0], [0, 0], [0, 0]]), 0)

    def test_does_not_modify_input(self):
        rows = [[0, 1], [2, 3], [4, 5]]
        matrix_rank(rows)
        self.assertEqual(rows, [[0, 1], [2, 3], [4, 5]])

    def test_malformed_matrices(self):
        with self.assertRaises(ValueError):
            matrix_rank([[], [0]])
        with self.assertRaises(TypeError):
            matrix_rank("00")
        with self.assertRaises(TypeError):
            matrix_rank([b"00"])
        with self.assertRaises(TypeError):
            matrix_rank([[1.0]])
        with self.assertRaises(TypeError):
            matrix_rank([[False]])
