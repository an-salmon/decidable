from fractions import Fraction
from unittest import TestCase

from decidable.linalg import LinearSolution, matrix_nullspace, matrix_solve
from decidable.linear import find_farkas_certificate, verify_farkas_certificate
from decidable.positive_state import check_positive_state, find_positive_state


class ExactLinearStateTests(TestCase):
    def test_rectangular_matrix_solve_and_kernel(self):
        matrix = [[2, 4, 1], [4, 8, 2]]
        expected_kernel = ((-2, 1, 0), (Fraction(-1, 2), 0, 1))
        assert matrix_nullspace(matrix) == expected_kernel
        assert matrix_solve(matrix, [1, 2]) == LinearSolution(
            particular=(Fraction(1, 2), 0, 0),
            nullspace=expected_kernel,
        )
        assert matrix_solve(matrix, [1, 3]) is None
        assert matrix_solve([[2], [4]], [1, 2]) == LinearSolution(
            particular=(Fraction(1, 2),), nullspace=()
        )
        assert matrix_solve([], []) == LinearSolution(particular=(), nullspace=())
        assert matrix_solve([[], []], [0, 1]) is None

    def test_farkas_replay_and_feasible_system(self):
        matrix = [[1, -2], [0, 3], [-4, 0]]
        rhs = [0, 0, -1]
        multipliers = find_farkas_certificate(matrix, rhs)
        assert multipliers is not None
        assert verify_farkas_certificate(matrix, rhs, multipliers)
        assert not verify_farkas_certificate(matrix, rhs, [0, 0, 0])
        assert not verify_farkas_certificate(matrix, rhs, [-4, 0, 1])
        assert find_farkas_certificate(matrix, [0, 3, -1]) is None
        assert find_farkas_certificate([[]], [-1]) == (1,)

    def test_positive_state_supplied_and_recovered_weights(self):
        matrix = [[0, 2], [Fraction(1, 2), 0]]
        assert check_positive_state(matrix, 1, [Fraction(1, 2), 1]).status == "verified"
        perturbed = check_positive_state(matrix, 1, [Fraction(1, 2), Fraction(3, 4)])
        assert perturbed.status == "disproved"
        assert perturbed.witness == 1

        found = find_positive_state(matrix, 1)
        assert found.status == "verified"
        assert all(weight >= 1 for weight in found.witness)
        assert check_positive_state(matrix, 1, found.witness).status == "verified"

    def test_positive_state_infeasibility_at_spectral_radius(self):
        # rho = 1 is the spectral radius, but the first inequality forces w_1 <= 0.
        result = find_positive_state([[1, 0], [1, 1]], 1)
        assert result.status == "disproved"
        assert verify_farkas_certificate(
            [[0, 1], [0, 0], [-1, 0], [0, -1]], [0, 0, -1, -1], result.witness
        )

    def test_positive_state_rate_zero_and_invalid_inputs(self):
        assert find_positive_state([[0]], 0).status == "verified"
        assert find_positive_state([[1]], 0).status == "disproved"
        assert check_positive_state([[0]], 0, [0]).witness == 0
        with self.assertRaises(ValueError):
            find_positive_state([[0]], -1)
        with self.assertRaises(TypeError):
            find_positive_state([[0]], 1.0)
        with self.assertRaises(ValueError):
            find_positive_state([[-1]], 1)
        with self.assertRaises(ValueError):
            find_positive_state([[1, 0]], 1)
