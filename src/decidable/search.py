"""Search for rational certificates and replay them with exact verifiers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, combinations_with_replacement
from math import isfinite
from typing import Union

from sympy import Matrix, Poly, QQ

from .gram import GramForm, GramIndex, verify_gram_certificate
from .polynomial import verify_polynomial_certificate
from .rational import _validate_polynomial
from .report import CheckResult
from .series import (
    RationalSeries,
    _expand_geometric_sum,
    _scale_polynomial,
    _validate_series,
    verify_series_certificate,
)


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


RationalWeight = Union[int, Fraction]
GeometricAtom = tuple[int, Iterable[RationalWeight]]
WeightedGeometricAtom = tuple[Fraction, int, tuple[RationalWeight, ...]]


@dataclass(frozen=True)
class SeriesCertificate:
    """A series certificate with arguments in the exact verifier's format."""

    hypotheses: tuple[tuple[tuple[WeightedGeometricAtom, ...], RationalSeries], ...]
    slack: tuple[WeightedGeometricAtom, ...]


@dataclass(frozen=True)
class PolynomialCertificate:
    """A polynomial certificate with arguments in the exact verifier's format."""

    equations: tuple[tuple[Poly, Poly], ...]
    inequalities: tuple[tuple[Fraction, Poly], ...]
    squares: tuple[tuple[Fraction, Poly], ...]
    inequality_squares: tuple[tuple[Fraction, Poly, Poly], ...]


def search_series_certificate(
    target: RationalSeries,
    *,
    hypotheses: Iterable[tuple[Iterable[GeometricAtom], RationalSeries]] = (),
    slack: Iterable[GeometricAtom] = (),
    max_support: int | None = None,
) -> CheckResult:
    """Seek nonnegative rational weights on the supplied geometric terms.

    An atom is a pair (shift, ratios) representing z**shift divided by the
    product of the (1 - ratio*z) factors. Shifts and ratios must be
    nonnegative and ratios must be integers or Fractions. Every hypothesis
    has its own atom basis. Slack terms have the same form. Target and
    hypotheses have the exact rational domain of the series verifier.

    The optional support bound limits the total number of nonzero weights.
    With no bound, the search considers the full finite basis. A verified
    result includes a SeriesCertificate. An unsuccessful search is
    inconclusive for the target.
    """
    target = _validate_series(target)
    generators = target[0].gens
    scale = _scale_polynomial(target[0])
    one = scale**0
    bases = [(slack, (one, one))]
    bases.extend((atoms, _validate_series(series, generators)) for atoms, series in hypotheses)
    terms = []
    groups = []
    for atoms, (numerator, denominator) in bases:
        group = []
        for shift, ratios in atoms:
            ratios = tuple(ratios)
            (atom_numerator, atom_denominator), valid = _expand_geometric_sum(
                ((1, shift, ratios),), scale, None
            )
            if not valid:
                raise ValueError("geometric ratios must be nonnegative")
            terms.append((atom_numerator * numerator, atom_denominator * denominator))
            group.append((shift, ratios))
        groups.append(group)

    common_denominator = target[1]
    for _, denominator in terms:
        common_denominator = common_denominator.lcm(denominator)
    cleared_target = target[0] * common_denominator.exquo(target[1])
    columns = tuple(
        numerator * common_denominator.exquo(denominator) for numerator, denominator in terms
    )
    support = _support_limit(max_support, len(columns))
    scope = (
        f"Formal coefficientwise nonnegativity under the supplied series hypotheses "
        f"using {len(columns)} supplied terms and at most {support} nonzero weights"
    )
    weights = _find_weights(cleared_target, columns, max_support)
    if weights is not None:
        weight_iter = iter(weights)
        replay_groups = [
            tuple(
                (weight, shift, ratios)
                for (shift, ratios), weight in zip(group, weight_iter)
                if weight
            )
            for group in groups
        ]
        certificate = SeriesCertificate(
            hypotheses=tuple(
                (group, series) for group, (_, series) in zip(replay_groups[1:], bases[1:])
            ),
            slack=replay_groups[0],
        )
        if verify_series_certificate(
            target, hypotheses=certificate.hypotheses, slack=certificate.slack
        ):
            return CheckResult(
                status="verified", scope=scope, reason="Exact geometric certificate replayed",
                witness=certificate,
            )
    return CheckResult(
        status="inconclusive", scope=scope, reason="No certificate verified within the given support"
    )


def search_polynomial_certificate(
    target: Poly,
    *,
    equations: Iterable[tuple[Poly, Poly]] = (),
    inequalities: Iterable[Poly] = (),
    squares: Iterable[Poly] = (),
    inequality_squares: Iterable[tuple[Poly, Poly]] = (),
    max_support: int | None = None,
) -> CheckResult:
    """Seek exact weights in a supplied finite polynomial certificate template.

    Each equation entry is (multiplier_basis_polynomial, hypothesis) and
    has an unrestricted rational weight. Each inequality entry is a
    nonnegative hypothesis. Each square entry is a factor to square.
    Each inequality-square entry is (square_factor, nonnegative_hypothesis).
    Their weights must be nonnegative rationals. Choose the square factors
    explicitly, including any desired sums of monomials.

    The optional support bound limits the total number of nonzero weights.
    With no bound, the search considers the full finite template. A verified
    result includes a PolynomialCertificate. An unsuccessful search is
    inconclusive for the target.
    """
    target = _validate_polynomial(target).set_domain(QQ)
    equations = tuple(
        (
            _validate_polynomial(basis, target.gens).set_domain(QQ),
            _validate_polynomial(hypothesis, target.gens),
        )
        for basis, hypothesis in equations
    )
    inequalities = tuple(_validate_polynomial(poly, target.gens) for poly in inequalities)
    squares = tuple(_validate_polynomial(poly, target.gens) for poly in squares)
    inequality_squares = tuple(
        tuple(_validate_polynomial(poly, target.gens) for poly in (factor, hypothesis))
        for factor, hypothesis in inequality_squares
    )
    columns = (
        tuple(basis * hypothesis for basis, hypothesis in equations)
        + inequalities
        + tuple(factor**2 for factor in squares)
        + tuple(factor**2 * hypothesis for factor, hypothesis in inequality_squares)
    )
    support = _support_limit(max_support, len(columns))
    scope = (
        f"Real polynomial nonnegativity under the supplied hypotheses "
        f"using {len(columns)} supplied terms and at most {support} nonzero weights"
    )
    weights = _find_weights(target, columns, max_support, unrestricted=len(equations))
    if weights is not None:
        weight_iter = iter(weights)
        certificate = PolynomialCertificate(
            equations=tuple(
                (basis.mul_ground(weight), hypothesis)
                for (basis, hypothesis), weight in zip(equations, weight_iter) if weight
            ),
            inequalities=tuple(
                (weight, poly) for poly, weight in zip(inequalities, weight_iter) if weight
            ),
            squares=tuple(
                (weight, factor) for factor, weight in zip(squares, weight_iter) if weight
            ),
            inequality_squares=tuple(
                (weight, factor, hypothesis)
                for (factor, hypothesis), weight in zip(inequality_squares, weight_iter) if weight
            ),
        )
        if verify_polynomial_certificate(
            target,
            equations=certificate.equations,
            inequalities=certificate.inequalities,
            squares=certificate.squares,
            inequality_squares=certificate.inequality_squares,
        ):
            return CheckResult(
                status="verified", scope=scope, reason="Exact polynomial certificate replayed",
                witness=certificate,
            )
    return CheckResult(
        status="inconclusive", scope=scope, reason="No certificate verified within the given support"
    )


def _support_limit(max_support: int | None, term_count: int) -> int:
    if max_support is None:
        return term_count
    if type(max_support) is not int:
        raise TypeError("max_support must be an integer or None")
    if max_support < 0:
        raise ValueError("max_support must be nonnegative")
    return min(max_support, term_count)


def _find_weights(
    target: Poly, columns: tuple[Poly, ...], support: int | None, *, unrestricted: int = 0
) -> tuple[Fraction, ...] | None:
    if target.is_zero:
        return (Fraction(0),) * len(columns)
    if not columns or support == 0:
        return None

    monomials = sorted(set(target.monoms()).union(*(poly.monoms() for poly in columns)))
    matrix = Matrix([[poly.nth(*monomial) for poly in columns] for monomial in monomials])
    values = Matrix([target.nth(*monomial) for monomial in monomials])
    if support is None:
        return _solve_weight_feasibility(matrix, values, unrestricted)

    # A representation of minimal support has independent columns. In a
    # dependent support, move along a linear dependence until one weight
    # becomes zero, keeping all constrained weights nonnegative.
    for size in range(1, min(support, len(monomials)) + 1):
        for indices in combinations(range(len(columns)), size):
            try:
                solution, free_parameters = matrix[:, indices].gauss_jordan_solve(values)
            except ValueError:
                continue
            if free_parameters.rows:
                continue
            if any(value < 0 for index, value in zip(indices, solution) if index >= unrestricted):
                continue
            weights = [Fraction(0)] * len(columns)
            for index, value in zip(indices, solution):
                weights[index] = Fraction(value.p, value.q)
            return tuple(weights)
    return None


def _solve_weight_feasibility(
    matrix: Matrix, values: Matrix, unrestricted: int
) -> tuple[Fraction, ...] | None:
    from .linear import _closed_system_witness

    try:
        solution, parameters = matrix.gauss_jordan_solve(values)
    except ValueError:
        return None

    at_zero = dict.fromkeys(parameters, 0)
    candidate = solution.subs(at_zero)
    if any(weight < 0 for weight in candidate[unrestricted:]):
        if not parameters.rows:
            return None

        # The exact solution is affine in the free parameters. For each
        # constrained weight c + a*t >= 0, project -a*t <= c.
        rows = []
        bounds = []
        for expression, intercept in zip(solution[unrestricted:], candidate[unrestricted:]):
            rows.append(tuple(-Fraction(expression.coeff(parameter)) for parameter in parameters))
            bounds.append(Fraction(intercept))
        point, _ = _closed_system_witness(rows, bounds)
        if point is None:
            return None
        candidate = solution.subs(dict(zip(parameters, point)))
    return tuple(Fraction(weight) for weight in candidate)
