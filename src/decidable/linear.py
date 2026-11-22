"""Decide feasibility and implication for finite rational affine systems."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Union

from .linalg import ExactVector, RationalMatrix, RationalVector, _exact_matrix, _exact_vector


AffineForm = Sequence[Union[int, Fraction]]
ExactForm = tuple[Fraction, ...]
Constraint = tuple[ExactForm, bool]


def _exact_form(form: AffineForm, variable_count: int) -> ExactForm:
    if not isinstance(form, Sequence) or isinstance(form, (str, bytes, bytearray)):
        raise TypeError("an affine form must be a sequence")
    if len(form) != variable_count + 1:
        raise ValueError("an affine form needs a constant and a coefficient per variable")
    if any(type(value) not in (int, Fraction) for value in form):
        raise TypeError("affine coefficients must be int or Fraction")
    return tuple(Fraction(value) for value in form)


def _read_constraints(
    variable_count: int,
    inequalities: Iterable[AffineForm],
    strict_inequalities: Iterable[AffineForm],
) -> list[Constraint]:
    if type(variable_count) is not int:
        raise TypeError("variable_count must be an integer")
    if variable_count < 0:
        raise ValueError("variable_count must be nonnegative")

    return [
        (_exact_form(form, variable_count), strict)
        for forms, strict in ((inequalities, False), (strict_inequalities, True))
        for form in forms
    ]


def _add_constraint(rows: dict[ExactForm, bool], form: ExactForm, strict: bool) -> bool:
    """Normalize a row or return whether its constant inequality is true."""
    pivot = next((coefficient for coefficient in form[1:] if coefficient), None)
    if pivot is None:
        return form[0] > 0 if strict else form[0] >= 0

    scale = abs(pivot)
    normalized = tuple(coefficient / scale for coefficient in form)
    rows[normalized] = strict or rows.get(normalized, False)
    return True


def _feasible(constraints: list[Constraint]) -> bool:
    rows: dict[ExactForm, bool] = {}
    for form, strict in constraints:
        if not _add_constraint(rows, form, strict):
            return False

    while rows:
        projected: dict[ExactForm, bool] = {}
        lower_bounds: list[Constraint] = []
        upper_bounds: list[Constraint] = []
        for form, strict in rows.items():
            coefficient = form[-1]
            if coefficient == 0:
                projected[form[:-1]] = strict
                continue

            # The positive rows now have slope +1 and the negative rows -1.
            bound = tuple(value / abs(coefficient) for value in form[:-1])
            if coefficient > 0:
                lower_bounds.append((bound, strict))
            else:
                upper_bounds.append((bound, strict))

        # Every lower bound must meet every upper bound, strictly if either
        # bound is strict. This is the Fourier–Motzkin projection step.
        for lower, lower_strict in lower_bounds:
            for upper, upper_strict in upper_bounds:
                combined = tuple(left + right for left, right in zip(lower, upper))
                strict = lower_strict or upper_strict
                if not _add_constraint(projected, combined, strict):
                    return False
        rows = projected

    return True


def linear_feasible(
    variable_count: int,
    *,
    inequalities: Iterable[AffineForm] = (),
    strict_inequalities: Iterable[AffineForm] = (),
) -> bool:
    """Decide whether a finite conjunction of affine inequalities is feasible.

    A form ``(c, a_0, ...)`` denotes ``c + a_0*x_0 + ...``. Forms in
    ``inequalities`` must be nonnegative and those in ``strict_inequalities``
    must be positive. Encode an equation by adding its form and its negative
    to ``inequalities``. Use only ``int`` and ``Fraction`` coefficients.

    Fourier–Motzkin elimination uses exact arithmetic. The answer is the same
    over the reals and over the rationals. Malformed inputs raise ``TypeError``
    or ``ValueError``.
    """
    constraints = _read_constraints(variable_count, inequalities, strict_inequalities)
    return _feasible(constraints)


def linear_implies(
    variable_count: int,
    target: AffineForm,
    *,
    inequalities: Iterable[AffineForm] = (),
    strict_inequalities: Iterable[AffineForm] = (),
) -> bool:
    """Decide whether the hypotheses of ``linear_feasible`` imply ``target >= 0``.

    Negate the target as a strict inequality and check feasibility. Infeasible
    hypotheses imply every target.
    """
    constraints = _read_constraints(variable_count, inequalities, strict_inequalities)
    exact_target = _exact_form(target, variable_count)
    negated_target = tuple(-coefficient for coefficient in exact_target)
    constraints.append((negated_target, True))
    return not _feasible(constraints)


def verify_farkas_certificate(
    matrix: RationalMatrix, rhs: RationalVector, multipliers: RationalVector
) -> bool:
    """Replay infeasibility of the closed rational system A*x <= rhs.

    Check y >= 0, y^T A = 0 and y^T rhs < 0. Entries must be int or
    Fraction, with one multiplier and one right-hand entry per row. The
    empty row sequence represents a 0-by-0 matrix.
    """
    rows, column_count = _exact_matrix(matrix)
    bounds = _exact_vector(rhs, len(rows))
    weights = _exact_vector(multipliers, len(rows))
    return (
        all(weight >= 0 for weight in weights)
        and all(
            sum(weight * row[column] for weight, row in zip(weights, rows)) == 0
            for column in range(column_count)
        )
        and sum(weight * bound for weight, bound in zip(weights, bounds)) < 0
    )


def _keep_closed_row(
    rows: dict[ExactVector, tuple[Fraction, ExactVector]],
    form: ExactForm,
    provenance: ExactVector,
) -> None:
    """Keep the strongest row among identical normalized slopes."""
    scale = next((abs(entry) for entry in form[1:] if entry), Fraction(1))
    slope = tuple(entry / scale for entry in form[1:])
    constant = form[0] / scale
    previous = rows.get(slope)
    if previous is None or constant < previous[0]:
        rows[slope] = constant, tuple(weight / scale for weight in provenance)


def _closed_system_witness(
    matrix: RationalMatrix, rhs: RationalVector
) -> tuple[ExactVector | None, ExactVector | None]:
    """Return either a feasible point or a Farkas certificate."""
    matrix_rows, column_count = _exact_matrix(matrix)
    bounds = _exact_vector(rhs, len(matrix_rows))
    row_count = len(matrix_rows)
    rows: dict[ExactVector, tuple[Fraction, ExactVector]] = {}
    for index, (row, bound) in enumerate(zip(matrix_rows, bounds)):
        provenance = tuple(Fraction(index == source) for source in range(row_count))
        _keep_closed_row(rows, (bound, *(-entry for entry in row)), provenance)

    # Each row is c + a*x >= 0. Its provenance expresses it as a
    # nonnegative linear combination of the original rows.
    levels = []
    for dimension in range(column_count, -1, -1):
        projected = {}
        lower_bounds = []
        upper_bounds = []
        for slope, (constant, provenance) in rows.items():
            if not any(slope):
                if constant < 0:
                    return None, provenance
                continue
            coefficient = slope[-1]
            truncated = (constant, *slope[:-1])
            if not coefficient:
                _keep_closed_row(projected, truncated, provenance)
                continue
            scale = abs(coefficient)
            normalized = tuple(entry / scale for entry in truncated)
            combination = tuple(weight / scale for weight in provenance)
            if coefficient > 0:
                lower_bounds.append((normalized, combination))
            else:
                upper_bounds.append((normalized, combination))

        if dimension == 0:
            break
        levels.append(
            ([form for form, _ in lower_bounds], [form for form, _ in upper_bounds])
        )
        for lower, lower_weights in lower_bounds:
            for upper, upper_weights in upper_bounds:
                combined = tuple(left + right for left, right in zip(lower, upper))
                provenance = tuple(
                    left + right for left, right in zip(lower_weights, upper_weights)
                )
                _keep_closed_row(projected, combined, provenance)
        rows = projected

    point: list[Fraction] = []
    for lower_bounds, upper_bounds in reversed(levels):
        if lower_bounds:
            # From lower(x) + x_i >= 0, choose the largest lower bound.
            value = max(-_evaluate_affine(form, point) for form in lower_bounds)
        elif upper_bounds:
            value = min(_evaluate_affine(form, point) for form in upper_bounds)
        else:
            value = Fraction(0)
        point.append(value)
    return tuple(point), None


def _evaluate_affine(form: ExactForm, point: list[Fraction]) -> Fraction:
    return sum((entry * value for entry, value in zip(form[1:], point)), form[0])


def find_farkas_certificate(
    matrix: RationalMatrix, rhs: RationalVector
) -> ExactVector | None:
    """Find a rational infeasibility certificate for A*x <= rhs if one exists.

    Exact Fourier–Motzkin elimination can require exponentially many rows.
    Return None for a feasible system. This routine accepts closed
    inequalities with the same input contract as verify_farkas_certificate.
    """
    _, multipliers = _closed_system_witness(matrix, rhs)
    return multipliers
