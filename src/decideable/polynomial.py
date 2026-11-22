"""Exact certificates for nonnegativity of real polynomial expressions."""

from collections.abc import Iterable
from fractions import Fraction

from sympy import Poly, Rational

from .rational import _validate_polynomial


def _rational_weight(weight: int | Fraction) -> Rational:
    if type(weight) not in (int, Fraction):
        raise TypeError("certificate weights must be int or Fraction")
    return Rational(weight.numerator, weight.denominator)


def verify_polynomial_certificate(
    target: Poly,
    *,
    equations: Iterable[tuple[Poly, Poly]] = (),
    inequalities: Iterable[tuple[int | Fraction, Poly]] = (),
    squares: Iterable[tuple[int | Fraction, Poly]] = (),
) -> bool:
    """Verify a supplied certificate for ``target >= 0`` over the reals.

    Each equation pair contains a polynomial multiplier and a polynomial
    assumed to be zero. Each inequality pair contains a nonnegative rational
    weight and a polynomial assumed to be nonnegative. Each square pair
    contains a nonnegative rational weight and a polynomial to square.
    All polynomials must be SymPy ``Poly`` objects over ``ZZ`` or ``QQ``
    with the same ordered generators.

    The certificate succeeds if its weighted terms sum identically to the
    target and all weight restrictions hold. ``False`` rejects this supplied
    certificate and leaves the underlying implication unresolved. Malformed
    inputs raise ``TypeError`` or ``ValueError``.
    """
    remainder = _validate_polynomial(target)
    generators = remainder.gens

    for multiplier, hypothesis in equations:
        multiplier = _validate_polynomial(multiplier, generators)
        hypothesis = _validate_polynomial(hypothesis, generators)
        remainder -= multiplier * hypothesis

    valid_signs = True
    for weight, hypothesis in inequalities:
        scalar = _rational_weight(weight)
        valid_signs = valid_signs and scalar.is_nonnegative
        hypothesis = _validate_polynomial(hypothesis, generators)
        remainder -= scalar * hypothesis

    for weight, factor in squares:
        scalar = _rational_weight(weight)
        valid_signs = valid_signs and scalar.is_nonnegative
        factor = _validate_polynomial(factor, generators)
        remainder -= scalar * factor**2

    return valid_signs and remainder.is_zero
