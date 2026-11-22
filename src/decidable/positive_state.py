"""Exact certificates for a fixed-rate positive linear functional."""

from __future__ import annotations

from fractions import Fraction
from typing import Union

from .linalg import RationalMatrix, RationalVector, _exact_matrix, _exact_vector
from .linear import _closed_system_witness, verify_farkas_certificate
from .report import CheckResult


Rational = Union[int, Fraction]


def _positive_state_system(
    matrix: RationalMatrix, rate: Rational
) -> tuple[list[list[Fraction]], Fraction]:
    rows, size = _exact_matrix(matrix)
    if len(rows) != size:
        raise ValueError("the state matrix must be square")
    if any(entry < 0 for row in rows for entry in row):
        raise ValueError("the state matrix must be entrywise nonnegative")
    if type(rate) not in (int, Fraction):
        raise TypeError("the rate must be int or Fraction")
    if rate < 0:
        raise ValueError("the rate must be nonnegative")
    exact_rate = Fraction(rate)
    coefficients = [
        [rows[j][i] - (exact_rate if i == j else 0) for j in range(size)]
        for i in range(size)
    ]
    return coefficients, exact_rate


def check_positive_state(
    matrix: RationalMatrix, rate: Rational, weights: RationalVector
) -> CheckResult:
    """Check whether the supplied w > 0 satisfies M^T*w <= rate*w.

    Accept a nonnegative rational square matrix (including the empty matrix)
    and a nonnegative rational rate. All entries must be int or Fraction.
    A failure disproves the claim for the supplied weights and gives an index
    in the failed inequality. Malformed inputs raise TypeError or ValueError.
    """
    coefficients, exact_rate = _positive_state_system(matrix, rate)
    size = len(coefficients)
    vector = _exact_vector(weights, size)
    scope = f"Supplied weights for the {size}-dimensional state at fixed rate {exact_rate}"
    for index, weight in enumerate(vector):
        if weight <= 0:
            return CheckResult(
                status="disproved", scope=scope,
                reason=f"Weight {index} is not strictly positive.", witness=index,
            )
    for index, row in enumerate(coefficients):
        if sum(entry * weight for entry, weight in zip(row, vector)) > 0:
            return CheckResult(
                status="disproved", scope=scope,
                reason=f"(M^T*w)[{index}] exceeds rate*w[{index}].", witness=index,
            )
    return CheckResult(
        status="verified", scope=scope,
        reason="w^T M x <= rate*w^T x for every nonnegative state x.", witness=vector,
    )


def find_positive_state(matrix: RationalMatrix, rate: Rational) -> CheckResult:
    """Decide whether any w > 0 satisfies M^T*w <= rate*w at the fixed rate.

    Exact Fourier–Motzkin elimination returns a candidate or a separating
    multiplier. Homogeneity allows rescaling every positive vector to
    satisfy w_i >= 1. For replay, use (M^T - rate*I) followed by -I as the
    rows of A, with rhs equal to zero then minus one in each block.
    A verified result carries weights. A disproved result carries Farkas
    multipliers for this system. Inputs follow check_positive_state.
    """
    coefficients, exact_rate = _positive_state_system(matrix, rate)
    size = len(coefficients)
    matrix_rows = coefficients + [
        [-int(i == j) for j in range(size)] for i in range(size)
    ]
    bounds = [0] * size + [-1] * size
    scope = f"Existence of strictly positive weights in dimension {size} at fixed rate {exact_rate}"
    weights, multipliers = _closed_system_witness(matrix_rows, bounds)
    if weights is not None and all(weight >= 1 for weight in weights):
        replay = check_positive_state(matrix, exact_rate, weights)
        if replay.status == "verified":
            return CheckResult(
                status="verified", scope=scope, reason=replay.reason, witness=weights,
            )
    if multipliers is not None and verify_farkas_certificate(matrix_rows, bounds, multipliers):
        return CheckResult(
            status="disproved", scope=scope,
            reason="Exact Farkas multipliers contradict the inequalities for weights w_i >= 1.",
            witness=multipliers,
        )
    return CheckResult(
        status="inconclusive", scope=scope,
        reason="The candidate did not pass exact replay.",
    )
