"""Verify exact coefficientwise inequalities between formal power series."""

from collections.abc import Iterable
from fractions import Fraction

from sympy import Expr, Poly, QQ, Rational

from .rational import _validate_polynomial, rational_identity


RationalSeries = tuple[Poly, Poly]
ExactScalar = int | Fraction
GeometricTerm = tuple[ExactScalar, int, Iterable[ExactScalar]]
GeometricSum = Iterable[GeometricTerm]


def _validate_series(
    series: RationalSeries, generators: tuple[Expr, ...] | None = None
) -> RationalSeries:
    numerator, denominator = series
    _validate_polynomial(numerator, generators)
    _validate_polynomial(denominator, numerator.gens)

    if any(any(powers[1:]) for powers in denominator.monoms()):
        raise ValueError("a series denominator may depend only on the scale generator")
    if denominator.nth(*(0,) * len(denominator.gens)) == 0:
        raise ValueError("a series denominator must have a nonzero constant term")
    if any(sum(powers[1:]) > 1 for powers in numerator.monoms()):
        raise ValueError("a series numerator must be affine in the formal series symbols")
    return numerator, denominator


def _add_series(left: RationalSeries, right: RationalSeries) -> RationalSeries:
    left_numerator, left_denominator = left
    right_numerator, right_denominator = right
    return (
        left_numerator * right_denominator + right_numerator * left_denominator,
        left_denominator * right_denominator,
    )


def _exact_scalar(value: ExactScalar) -> Rational:
    if type(value) not in (int, Fraction):
        raise TypeError("geometric weights and ratios must be int or Fraction")
    return Rational(value.numerator, value.denominator)


def _expand_geometric_sum(terms: GeometricSum, scale: Poly) -> tuple[RationalSeries, bool]:
    zero = scale * 0
    one = zero + 1
    result = (zero, one)
    valid_signs = True
    for weight, shift, ratios in terms:
        exact_weight = _exact_scalar(weight)
        if type(shift) is not int:
            raise TypeError("a geometric shift must be an integer")
        if shift < 0:
            raise ValueError("a geometric shift must be nonnegative")
        valid_signs = valid_signs and weight >= 0
        denominator = one
        for ratio in ratios:
            exact_ratio = _exact_scalar(ratio)
            valid_signs = valid_signs and ratio >= 0
            denominator *= one - exact_ratio * scale
        result = _add_series(result, (exact_weight * scale**shift, denominator))
    return result, valid_signs


def verify_series_certificate(
    target: RationalSeries,
    *,
    hypotheses: Iterable[tuple[GeometricSum, RationalSeries]] = (),
    slack: GeometricSum = (),
) -> bool:
    """Verify a coefficientwise nonnegativity certificate for ``target``.

    Target and hypothesis pairs ``(numerator, denominator)`` use exact SymPy
    Poly objects with the same ordered generators. The first generator is the
    scale variable z. Each other generator stands for an arbitrary formal
    power series, and the numerator must be affine in them. The denominator
    may depend on z alone and must have a nonzero constant term.

    Each hypothesis entry is a pair ``(multiplier, hypothesis)``. A multiplier
    or slack is a sum of terms ``(weight, shift, ratios)`` representing
    ``weight * z**shift / product(1 - ratio*z for ratio in ratios)``.
    Nonnegative weights, shifts, and ratios give coefficientwise nonnegative
    series. The certificate asserts that target equals the sum of the
    multiplied hypotheses and slack.

    Return False for a negative weight or ratio or an identity mismatch.
    Malformed inputs raise TypeError or ValueError. The check is purely formal
    and needs no analytic convergence of the series.
    """
    target_numerator, target_denominator = _validate_series(target)
    generators = target_numerator.gens
    scale_powers = (1,) + (0,) * (len(generators) - 1)
    scale = Poly.from_dict({scale_powers: 1}, generators, domain=QQ)
    result, valid_signs = _expand_geometric_sum(slack, scale)
    for multiplier, hypothesis in hypotheses:
        hypothesis_numerator, hypothesis_denominator = _validate_series(hypothesis, generators)
        (multiplier_numerator, multiplier_denominator), nonnegative = _expand_geometric_sum(
            multiplier, scale
        )
        valid_signs = valid_signs and nonnegative
        result = _add_series(
            result,
            (
                multiplier_numerator * hypothesis_numerator,
                multiplier_denominator * hypothesis_denominator,
            ),
        )
    return valid_signs and rational_identity(target_numerator, target_denominator, *result)
