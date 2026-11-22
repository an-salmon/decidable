"""Search for exact rational certificates of linear Gram inequalities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from math import isfinite

from sympy import Matrix

from .gram import GramForm, GramIndex, verify_gram_certificate


WeightedForm = tuple[Fraction, GramForm]
WeightedSquare = tuple[Fraction, tuple[Fraction, ...]]


@dataclass(frozen=True)
class GramCertificate:
    """A rational certificate in the form accepted by the Gram verifier."""

    equations: tuple[WeightedForm, ...]
    inequalities: tuple[WeightedForm, ...]
    squares: tuple[WeightedSquare, ...]


def search_gram_certificate(
    vector_count: int,
    target: GramForm,
    *,
    equations: Iterable[GramForm] = (),
    inequalities: Iterable[GramForm] = (),
    max_denominator: int = 1000,
) -> GramCertificate | None:
    """Try to prove nonnegativity of a target under exact rational hypotheses.

    Supply separate Gram forms assumed to be zero and nonnegative, respectively.
    Try an exact sum of squares and a linear solve before using CVXPY, if
    installed, to propose multipliers. Rationalize those multipliers using
    denominators up to ``max_denominator`` and check the residual exactly.

    Every returned certificate passes ``verify_gram_certificate``. A result
    of ``None`` is inconclusive. Malformed inputs raise TypeError or ValueError.
    """
    if type(max_denominator) is not int:
        raise TypeError("max_denominator must be an integer")
    if max_denominator < 1:
        raise ValueError("max_denominator must be positive")

    equations = tuple(equations)
    inequalities = tuple(inequalities)
    verify_gram_certificate(
        vector_count,
        target,
        equations=((0, form) for form in equations),
        inequalities=((0, form) for form in inequalities),
    )

    hypotheses = equations + inequalities
    certificate = _certify(
        vector_count, target, equations, inequalities, (Fraction(0),) * len(hypotheses)
    )
    if certificate is not None or not hypotheses:
        return certificate

    multipliers = _linear_multipliers(vector_count, target, hypotheses)
    if multipliers is not None:
        certificate = _certify(vector_count, target, equations, inequalities, multipliers)
        if certificate is not None:
            return certificate

    proposal = _solve_sdp(vector_count, target, equations, inequalities)
    if proposal is None:
        return None

    denominator_limit = 1
    while True:
        multipliers = tuple(
            Fraction(value).limit_denominator(denominator_limit) for value in proposal
        )
        certificate = _certify(vector_count, target, equations, inequalities, multipliers)
        if certificate is not None or denominator_limit == max_denominator:
            return certificate
        denominator_limit = min(10 * denominator_limit, max_denominator)


def _linear_multipliers(
    vector_count: int, target: GramForm, hypotheses: Sequence[GramForm]
) -> tuple[Fraction, ...] | None:
    indices = tuple(combinations_with_replacement(range(vector_count), 2))
    coefficients = Matrix(
        len(indices), len(hypotheses), lambda i, j: hypotheses[j].get(indices[i], 0)
    )
    values = Matrix([target.get(index, 0) for index in indices])
    try:
        solution, free_parameters = coefficients.gauss_jordan_solve(values)
    except ValueError:
        return None

    particular = solution.subs(dict.fromkeys(free_parameters, 0))
    return tuple(Fraction(value.p, value.q) for value in particular)


def _certify(
    vector_count: int,
    target: GramForm,
    equations: Sequence[GramForm],
    inequalities: Sequence[GramForm],
    multipliers: tuple[Fraction, ...],
) -> GramCertificate | None:
    equality_terms = tuple(zip(multipliers[: len(equations)], equations))
    inequality_terms = tuple(zip(multipliers[len(equations) :], inequalities))
    if any(weight < 0 for weight, _ in inequality_terms):
        return None

    residual: defaultdict[GramIndex, Fraction] = defaultdict(Fraction, target)
    for weight, form in equality_terms + inequality_terms:
        for index, coefficient in form.items():
            residual[index] -= weight * coefficient

    squares = _weighted_squares(_gram_matrix(vector_count, residual))
    if squares is None or not verify_gram_certificate(
        vector_count,
        target,
        equations=equality_terms,
        inequalities=inequality_terms,
        squares=squares,
    ):
        return None
    return GramCertificate(equality_terms, inequality_terms, squares)


def _gram_matrix(vector_count: int, form: GramForm) -> list[list[Fraction]]:
    matrix = [[Fraction(0) for _ in range(vector_count)] for _ in range(vector_count)]
    for (i, j), coefficient in form.items():
        entry = Fraction(coefficient) / (1 if i == j else 2)
        matrix[i][j] = matrix[j][i] = entry
    return matrix


def _weighted_squares(matrix: list[list[Fraction]]) -> tuple[WeightedSquare, ...] | None:
    """Factor a positive semidefinite matrix by exact symmetric elimination."""
    squares = []
    size = len(matrix)
    for k, row in enumerate(matrix):
        weight = row[k]
        if weight == 0 and not any(row[k + 1 :]):
            continue
        if weight <= 0:
            return None

        coefficients = (Fraction(0),) * k + tuple(value / weight for value in row[k:])
        squares.append((weight, coefficients))
        for i in range(k + 1, size):
            for j in range(i, size):
                entry = matrix[i][j] - weight * coefficients[i] * coefficients[j]
                matrix[i][j] = matrix[j][i] = entry
    return tuple(squares)


def _solve_sdp(
    vector_count: int,
    target: GramForm,
    equations: Sequence[GramForm],
    inequalities: Sequence[GramForm],
) -> tuple[float, ...] | None:
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError:
        return None

    positive_residual = cp.Variable((vector_count, vector_count), PSD=True)
    multipliers = tuple(cp.Variable() for _ in equations) + tuple(
        cp.Variable(nonneg=True) for _ in inequalities
    )
    reconstructed = positive_residual
    try:
        for variable, form in zip(multipliers, equations + inequalities):
            coefficients = np.asarray(_gram_matrix(vector_count, form), dtype=float)
            reconstructed = reconstructed + variable * coefficients
        target_matrix = np.asarray(_gram_matrix(vector_count, target), dtype=float)
        problem = cp.Problem(cp.Minimize(0), [reconstructed == target_matrix])
        problem.solve()
    except (cp.SolverError, OverflowError):
        return None

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None
    if any(variable.value is None for variable in multipliers):
        return None
    proposal = tuple(float(variable.value) for variable in multipliers)
    return proposal if all(isfinite(value) for value in proposal) else None
