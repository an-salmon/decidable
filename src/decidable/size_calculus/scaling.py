"""Exact rational weight propagation under a declared positive dilation."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from sympy import Add, Expr, Float, Mul, Pow, Rational, log

from ..report import CheckResult
from .moments import Exact, exact


class _UnknownWeight(ValueError):
    pass


def norm_weight(
    amplitude: Exact,
    reciprocal_p: Exact,
    volume: Exact,
    *,
    normalized: bool,
    derivative_order: Exact = 0,
    length_weight: Exact = 1,
    support_fraction: Exact = 0,
) -> Fraction:
    """Weight of one Lp norm or homogeneous Sobolev norm.

    Negative derivative orders use the matching homogeneous dual norm.
    A nonzero support fraction weight is a premise about the family.
    Pass reciprocal_p=0 for the supremum norm.
    """
    if type(normalized) is not bool:
        raise TypeError("normalized must be a bool")
    reciprocal_p = exact(reciprocal_p, "reciprocal p")
    if not 0 <= reciprocal_p <= 1:
        raise ValueError("reciprocal p must lie between zero and one")
    volume = exact(volume, "volume weight")
    measure_weight = exact(support_fraction, "support fraction weight")
    if not normalized:
        measure_weight += volume
    return (
        exact(amplitude, "amplitude weight")
        - exact(derivative_order, "derivative order") * exact(length_weight, "length weight")
        + reciprocal_p * measure_weight
    )


def _extrema(expression, weights: Mapping[Expr, Fraction]) -> tuple[Fraction, Fraction]:
    if expression == log(2):
        return Fraction(0), Fraction(0)
    if expression in weights:
        return weights[expression], weights[expression]
    if type(expression) in (int, Fraction) or isinstance(expression, Rational):
        if expression > 0:
            return Fraction(0), Fraction(0)
    if type(expression) is Mul:
        terms = tuple(_extrema(term, weights) for term in expression.args)
        return sum(low for low, _ in terms), sum(high for _, high in terms)
    if type(expression) is Add:
        terms = tuple(_extrema(term, weights) for term in expression.args)
        return min(low for low, _ in terms), max(high for _, high in terms)
    if type(expression) is Pow and isinstance(expression.exp, Rational):
        low, high = _extrema(expression.base, weights)
        exponent = Fraction(expression.exp)
        # These are the weights at zero and infinity, in that order.
        return low * exponent, high * exponent
    raise _UnknownWeight(f"No positive dilation weight for {expression}.")


def check_scaling(lhs, rhs, weights: Mapping[Expr, Exact], *, limit: str = "both") -> CheckResult:
    """Check the weight condition for lhs <= C*rhs, with C dilation-independent.

    Declared atoms represent positive quantities with the supplied weights.
    Products, rational powers and positive sums propagate the small- and
    large-dilation weights. The result concerns the weight condition itself.
    """
    if limit not in ("small", "large", "both"):
        raise ValueError("limit must be small, large or both")
    for atom in weights:
        if not isinstance(atom, Expr) or atom.has(Float):
            raise TypeError("declare exact symbolic weight keys")
        if atom.is_number or atom.is_commutative is not True or atom.is_positive is False:
            raise ValueError("declare weights for nonconstant positive quantities")
    weights = {atom: exact(weight, "dilation weight") for atom, weight in weights.items()}
    scope = f"Dilation weight condition ({limit}) for the supplied family"
    try:
        left = _extrema(lhs, weights)
        right = _extrema(rhs, weights)
    except _UnknownWeight as error:
        return CheckResult("unsupported", scope, str(error))
    if (limit != "large" and left[0] < right[0]) or (limit != "small" and left[1] > right[1]):
        return CheckResult("disproved", scope, "Dilation weights are incompatible.", (left, right))
    return CheckResult("verified", scope, "The necessary weight condition holds.", (left, right))
