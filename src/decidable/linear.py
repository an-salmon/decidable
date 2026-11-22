"""Decide feasibility and implication for finite rational affine systems."""

from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Union


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
