"""Verify exact coefficientwise inequalities between formal power series."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Union

from sympy import Expr, Poly, QQ
from sympy.polys.fields import FracElement

from .parameters import PositiveParameters
from .rational import _validate_polynomial, rational_identity


RationalSeries = tuple[Poly, Poly]
ExactScalar = Union[int, Fraction, FracElement]
GeometricTerm = tuple[ExactScalar, int, Iterable[ExactScalar]]
GeometricSum = Iterable[GeometricTerm]


def _validate_series(
    series: RationalSeries,
    generators: tuple[Expr, ...] | None = None,
    *,
    parameters: PositiveParameters | None = None,
) -> RationalSeries:
    numerator, denominator = series
    numerator = _validate_polynomial(numerator, generators, parameters=parameters)
    denominator = _validate_polynomial(denominator, numerator.gens, parameters=parameters)

    domain = QQ if parameters is None else parameters.field
    numerator = numerator.set_domain(domain)
    denominator = denominator.set_domain(domain)

    if any(any(powers[1:]) for powers in denominator.monoms()):
        raise ValueError("a series denominator may depend only on the scale generator")
    constant = _constant_coefficient(denominator)
    nonzero = constant != 0 if parameters is None else parameters.nonzero(constant)
    if not nonzero:
        raise ValueError("a series denominator must have a provably nonzero constant term")
    if any(sum(powers[1:]) > 1 for powers in numerator.monoms()):
        raise ValueError("a series numerator must be affine in the formal series symbols")
    return numerator, denominator


def _constant_coefficient(polynomial: Poly):
    return polynomial.rep.nth(*(0,) * len(polynomial.gens))


def _scale_polynomial(polynomial: Poly) -> Poly:
    powers = (1,) + (0,) * (len(polynomial.gens) - 1)
    return Poly.from_dict({powers: 1}, polynomial.gens, domain=polynomial.domain)


def _add_series(left: RationalSeries, right: RationalSeries) -> RationalSeries:
    left_numerator, left_denominator = left
    right_numerator, right_denominator = right
    return (
        left_numerator * right_denominator + right_numerator * left_denominator,
        left_denominator * right_denominator,
    )


def _exact_coefficient(value: ExactScalar, parameters: PositiveParameters | None):
    if parameters is not None:
        return parameters.coefficient(value)
    if type(value) not in (int, Fraction):
        raise TypeError("series scalar inputs must be int or Fraction")
    return QQ.convert(value)


def _nonnegative(coefficient, parameters: PositiveParameters | None) -> bool:
    return coefficient >= 0 if parameters is None else parameters.nonnegative(coefficient)


def _expand_geometric_sum(
    terms: GeometricSum, scale: Poly, parameters: PositiveParameters | None
) -> tuple[RationalSeries, bool]:
    zero = scale * 0
    one = zero + 1
    result = (zero, one)
    valid_signs = True
    for weight, shift, ratios in terms:
        exact_weight = _exact_coefficient(weight, parameters)
        if type(shift) is not int:
            raise TypeError("a geometric shift must be an integer")
        if shift < 0:
            raise ValueError("a geometric shift must be nonnegative")
        valid_signs = valid_signs and _nonnegative(exact_weight, parameters)
        denominator = one
        for ratio in ratios:
            exact_ratio = _exact_coefficient(ratio, parameters)
            valid_signs = valid_signs and _nonnegative(exact_ratio, parameters)
            denominator *= one - scale.mul_ground(exact_ratio)
        result = _add_series(result, ((scale**shift).mul_ground(exact_weight), denominator))
    return result, valid_signs


def verify_series_certificate(
    target: RationalSeries,
    *,
    hypotheses: Iterable[tuple[GeometricSum, RationalSeries]] = (),
    slack: GeometricSum = (),
    parameters: PositiveParameters | None = None,
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

    With ``parameters``, coefficients may lie in the declared rational function
    field. A sign proof is required for each weight and ratio and the constant
    coefficient of every input denominator must be provably nonzero.

    Return False for an unverified weight or ratio or an identity mismatch.
    Malformed inputs raise TypeError or ValueError. The check is purely formal
    and needs no analytic convergence of the series.
    """
    target_numerator, target_denominator = _validate_series(target, parameters=parameters)
    generators = target_numerator.gens
    scale = _scale_polynomial(target_numerator)
    result, valid_signs = _expand_geometric_sum(slack, scale, parameters)
    for multiplier, hypothesis in hypotheses:
        hypothesis_numerator, hypothesis_denominator = _validate_series(
            hypothesis, generators, parameters=parameters
        )
        (multiplier_numerator, multiplier_denominator), nonnegative = _expand_geometric_sum(
            multiplier, scale, parameters
        )
        valid_signs = valid_signs and nonnegative
        result = _add_series(
            result,
            (
                multiplier_numerator * hypothesis_numerator,
                multiplier_denominator * hypothesis_denominator,
            ),
        )
    return valid_signs and rational_identity(
        target_numerator, target_denominator, *result, parameters=parameters
    )


def rebase_series(
    series: RationalSeries,
    initials: Mapping[Expr, ExactScalar],
    *,
    parameters: PositiveParameters | None = None,
) -> RationalSeries:
    """Express a series in terms of the tails of its formal-series symbols.

    Supply one exact constant coefficient in ``initials`` for every generator
    after the scale generator z. Substitute ``A -> initials[A] + z*A``
    simultaneously for all such generators. The returned polynomials retain
    the same generators, with each series symbol now denoting its tail.
    """
    numerator, denominator = _validate_series(series, parameters=parameters)
    series_symbols = numerator.gens[1:]
    if not isinstance(initials, Mapping):
        raise TypeError("series initials must map generators to exact coefficients")
    if set(initials) != set(series_symbols):
        raise ValueError("supply one initial coefficient for each formal series generator")

    values = [_exact_coefficient(initials[symbol], parameters) for symbol in series_symbols]
    domain = numerator.domain
    rebased = defaultdict(lambda: domain.zero)
    for powers, coefficient in numerator.rep.to_dict().items():
        scale_power, *series_powers = powers
        if any(series_powers):
            initial = values[series_powers.index(1)]
            rebased[(scale_power,) + (0,) * len(series_symbols)] += coefficient * initial
            rebased[(scale_power + 1, *series_powers)] += coefficient
        else:
            rebased[powers] += coefficient
    return Poly.from_dict(rebased, numerator.gens, domain=domain), denominator


def shift_series(
    series: RationalSeries,
    initials: Mapping[Expr, ExactScalar],
    *,
    parameters: PositiveParameters | None = None,
) -> RationalSeries:
    """Return ``(F - F(0))/z`` in the coordinates of :func:`rebase_series`.

    Use ``rebase_series`` with the same initials for any accompanying
    hypotheses or target that should keep their original coefficients.
    """
    numerator, denominator = rebase_series(series, initials, parameters=parameters)
    domain = numerator.domain
    initial = domain.exquo(_constant_coefficient(numerator), _constant_coefficient(denominator))
    shifted = (numerator - denominator.mul_ground(initial)).exquo(_scale_polynomial(numerator))
    return shifted, denominator
