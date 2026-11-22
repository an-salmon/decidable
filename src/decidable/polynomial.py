"""Exact certificates for nonnegativity of real polynomial expressions."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Union

from sympy import Expr, Poly, Rational
from sympy.polys.fields import FracElement

from .parameters import PositiveParameters
from .rational import _validate_polynomial

Weight = Union[int, Fraction, FracElement]


def _checked_weight(
    weight: Weight, parameters: PositiveParameters | None
) -> tuple[Expr, bool]:
    if parameters is not None:
        coefficient = parameters.coefficient(weight)
        return parameters.field.to_sympy(coefficient), parameters.nonnegative(coefficient)
    if type(weight) not in (int, Fraction):
        raise TypeError("certificate weights must be int or Fraction")
    return Rational(weight.numerator, weight.denominator), weight >= 0


def verify_polynomial_certificate(
    target: Poly,
    *,
    equations: Iterable[tuple[Poly, Poly]] = (),
    inequalities: Iterable[tuple[Weight, Poly]] = (),
    squares: Iterable[tuple[Weight, Poly]] = (),
    inequality_squares: Iterable[tuple[Weight, Poly, Poly]] = (),
    parameters: PositiveParameters | None = None,
) -> bool:
    """Verify a supplied certificate for ``target >= 0`` over the reals.

    Each equation pair contains a polynomial multiplier and a polynomial
    assumed to be zero. Each inequality pair contains a nonnegative rational
    weight and a polynomial assumed to be nonnegative. Each square pair
    contains a nonnegative weight and a polynomial to square. An
    ``inequality_squares`` triple contains a nonnegative weight, a
    polynomial to square, and a polynomial assumed to be nonnegative.
    A sum of these triples permits a sum-of-squares multiplier on a
    hypothesis.

    All polynomials must be SymPy ``Poly`` objects with the same ordered
    generators. By default coefficients are over ``ZZ`` or ``QQ`` and
    weights are integers or ``Fraction`` values. If ``parameters`` is
    supplied, coefficients and weights may also be rational functions of
    the declared positive parameters. Every weight must have a verified
    nonnegativity proof over that parameter domain.

    The certificate succeeds if its weighted terms sum identically to the
    target and all weight restrictions hold. ``False`` rejects this supplied
    certificate and leaves the underlying implication unresolved. Malformed
    inputs raise ``TypeError`` or ``ValueError``.
    """
    remainder = _validate_polynomial(target, parameters=parameters)
    generators = remainder.gens

    for multiplier, hypothesis in equations:
        multiplier = _validate_polynomial(multiplier, generators, parameters=parameters)
        hypothesis = _validate_polynomial(hypothesis, generators, parameters=parameters)
        remainder -= multiplier * hypothesis

    valid_signs = True
    for weight, hypothesis in inequalities:
        scalar, nonnegative = _checked_weight(weight, parameters)
        valid_signs = valid_signs and nonnegative
        hypothesis = _validate_polynomial(hypothesis, generators, parameters=parameters)
        remainder -= scalar * hypothesis

    for weight, factor in squares:
        scalar, nonnegative = _checked_weight(weight, parameters)
        valid_signs = valid_signs and nonnegative
        factor = _validate_polynomial(factor, generators, parameters=parameters)
        remainder -= scalar * factor**2

    for weight, factor, hypothesis in inequality_squares:
        scalar, nonnegative = _checked_weight(weight, parameters)
        valid_signs = valid_signs and nonnegative
        factor = _validate_polynomial(factor, generators, parameters=parameters)
        hypothesis = _validate_polynomial(hypothesis, generators, parameters=parameters)
        remainder -= scalar * factor**2 * hypothesis

    return valid_signs and remainder.is_zero
