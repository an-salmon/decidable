"""Exact local rules for polynomial size functions on the nonnegative ray."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from sympy import Add, Expr, Poly, QQ, Rational, Symbol

from ..rational import _validate_polynomial
from .moments import Exact, _nonnegative, exact
from .typing import PointwiseOrder, _positive_scale, _quantity


@dataclass(frozen=True)
class PolynomialSizeFunction:
    """A finite, nonzero rational polynomial with Φ(0) = 0 and Φ' >= 0 on [0, ∞)."""

    polynomial: Poly

    def __post_init__(self):
        polynomial = _validate_polynomial(self.polynomial)
        if (len(polynomial.gens) != 1 or type(polynomial.gen) is not Symbol
                or polynomial.gen.is_real is False):
            raise ValueError("a size function needs one real polynomial coordinate")
        if polynomial.is_zero or polynomial.eval(0) != 0:
            raise ValueError("a size function must vanish at zero and be positive somewhere")
        polynomial = polynomial.to_field()
        if not _nonnegative(polynomial.diff(), Fraction(0), None):
            raise ValueError("a size function must be nondecreasing on the nonnegative ray")
        object.__setattr__(self, "polynomial", polynomial)

    @property
    def is_convex(self) -> bool:
        """Decide convexity by checking Φ'' on the nonnegative ray."""
        return _nonnegative(self.polynomial.diff().diff(), Fraction(0), None)


@dataclass(frozen=True)
class PolynomialSizeBound:
    """The one-sided judgement E[Φ(max(expression, 0)/scale)] <= 1."""

    expression: Expr | Exact
    size_function: PolynomialSizeFunction
    scale: Expr | Exact

    def __post_init__(self):
        if type(self.size_function) is not PolynomialSizeFunction:
            raise TypeError("supply a validated polynomial size function")
        object.__setattr__(self, "expression", _quantity(self.expression))
        object.__setattr__(self, "scale", _positive_scale(self.scale))


def _require_bound(bound: PolynomialSizeBound) -> None:
    if type(bound) is not PolynomialSizeBound:
        raise TypeError("supply a polynomial size bound")


def polynomial_monotone(
    bound: PolynomialSizeBound, premise: PointwiseOrder
) -> PolynomialSizeBound:
    """Transfer a size bound to a quantity pointwise at most the input."""
    _require_bound(bound)
    if type(premise) is not PointwiseOrder or premise.upper != bound.expression:
        raise ValueError("monotonicity needs a matching pointwise order")
    return PolynomialSizeBound(premise.lower, bound.size_function, bound.scale)


def polynomial_multiply(
    bound: PolynomialSizeBound, coefficient: Expr | Exact
) -> PolynomialSizeBound:
    """Multiply the quantity and its scale by the same positive deterministic factor."""
    _require_bound(bound)
    coefficient = _positive_scale(coefficient)
    return PolynomialSizeBound(coefficient * bound.expression, bound.size_function,
                               coefficient * bound.scale)


def polynomial_enlarge_scale(bound: PolynomialSizeBound, factor: Exact) -> PolynomialSizeBound:
    """Enlarge the scale by an exact rational factor of at least one."""
    _require_bound(bound)
    factor = exact(factor, "scale factor")
    if factor < 1:
        raise ValueError("a scale enlargement factor must be at least one")
    return PolynomialSizeBound(bound.expression, bound.size_function, Rational(factor) * bound.scale)


def polynomial_change_class(
    bound: PolynomialSizeBound, target: PolynomialSizeFunction, factor: Exact
) -> PolynomialSizeBound:
    """Use the pointwise criterion Ψ(v/factor) <= Φ(v) for changing size function."""
    _require_bound(bound)
    if type(target) is not PolynomialSizeFunction:
        raise TypeError("supply a validated target size function")
    factor = exact(factor, "scale factor")
    if factor <= 0:
        raise ValueError("a scale factor must be positive")
    source = bound.size_function.polynomial
    target_at_scale = Poly.from_dict(
        {monomial: coefficient / Rational(factor)**monomial[0]
         for monomial, coefficient in target.polynomial.terms()},
        source.gen, domain=QQ,
    )
    if not _nonnegative(source - target_at_scale, Fraction(0), None):
        raise ValueError("the pointwise comparison fails on the nonnegative ray")
    return PolynomialSizeBound(bound.expression, target, Rational(factor) * bound.scale)


def polynomial_weighted_aggregate(
    bounds: Iterable[PolynomialSizeBound], weights: Iterable[Exact]
) -> PolynomialSizeBound:
    """Use Jensen's inequality for a finite weighted sum in a common convex class.

    Positive parts obey max(sum(w_i X_i), 0) <= sum(w_i max(X_i, 0)).
    After dividing by the weighted total scale, the right side is a convex
    combination of the input ratios with weights proportional to w_i scale_i.
    """
    bounds = tuple(bounds)
    for bound in bounds:
        _require_bound(bound)
    weights = tuple(exact(weight, "aggregation weight") for weight in weights)
    if len(bounds) != len(weights) or any(weight < 0 for weight in weights):
        raise ValueError("supply one nonnegative weight per input")
    active = tuple((bound, Rational(weight)) for bound, weight in zip(bounds, weights) if weight)
    if not active:
        raise ValueError("at least one aggregation weight must be positive")
    size_function = active[0][0].size_function
    if any(bound.size_function.polynomial.rep != size_function.polynomial.rep for bound, _ in active):
        raise ValueError("positive-weight inputs must share their size function")
    if not size_function.is_convex:
        raise ValueError("the unit-factor sum rule requires a convex size function")
    return PolynomialSizeBound(
        Add(*(weight * bound.expression for bound, weight in active)),
        size_function, Add(*(weight * bound.scale for bound, weight in active)),
    )
