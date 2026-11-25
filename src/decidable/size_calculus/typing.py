"""Exact local rules for stretched-exponential size bounds (Os)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from sympy import Add, Expr, Float, Mul, Pow, Rational, Symbol, log

from ..report import CheckResult
from .moments import Exact, exact


_LOG_TWO = log(2)
_SCOPE = "On the stated size and pointwise hypotheses"


def _positive_scale(scale) -> Expr:
    if type(scale) in (int, Fraction):
        value = exact(scale, "scale")
        scale = Rational(value.numerator, value.denominator)
    if isinstance(scale, Rational) and scale > 0:
        return scale
    if scale == _LOG_TWO or (type(scale) is Symbol and scale.is_positive is True):
        return scale
    if type(scale) in (Add, Mul):
        for term in scale.args:
            _positive_scale(term)
        return scale
    if type(scale) is Pow and isinstance(scale.exp, Rational):
        _positive_scale(scale.base)
        return scale
    raise ValueError("scale must be a positive exact expression")


@dataclass(frozen=True)
class SizeBound:
    """A given or derived judgement X <= O_s(scale), stored using 1/s."""

    expression: Expr
    reciprocal: Exact
    scale: Expr | Exact
    nonnegative: bool

    def __post_init__(self):
        if not isinstance(self.expression, Expr) or self.expression.is_commutative is not True:
            raise TypeError("the quantity must be a commutative SymPy expression")
        if self.expression.has(Float):
            raise TypeError("the quantity expression must be exact")
        if type(self.nonnegative) is not bool:
            raise TypeError("nonnegative must be a bool")
        reciprocal = exact(self.reciprocal, "reciprocal exponent")
        if reciprocal <= 0:
            raise ValueError("reciprocal exponent must be positive")
        object.__setattr__(self, "reciprocal", reciprocal)
        object.__setattr__(self, "scale", _positive_scale(self.scale))


@dataclass(frozen=True)
class UpperBound:
    """A caller-supplied pointwise hypothesis X <= ceiling."""

    expression: Expr
    ceiling: Expr | Exact


def power(bound: SizeBound, exponent: Exact) -> SizeBound:
    """Propagate a positive rational power of a nonnegative quantity."""
    exponent = exact(exponent, "power")
    if exponent <= 0 or not bound.nonnegative:
        raise ValueError("a power requires a positive exponent and nonnegative input")
    return SizeBound(
        bound.expression**exponent, bound.reciprocal * exponent, bound.scale**exponent, True
    )


def product(bounds: Iterable[SizeBound]) -> SizeBound:
    """Multiply nonnegative inputs with no dependence assumptions."""
    bounds = tuple(bounds)
    if not bounds or not all(bound.nonnegative for bound in bounds):
        raise ValueError("a product requires nonempty nonnegative inputs")
    return SizeBound(
        Mul(*(bound.expression for bound in bounds)),
        sum(bound.reciprocal for bound in bounds),
        Mul(*(bound.scale for bound in bounds)),
        True,
    )


def descend(bound: SizeBound, reciprocal: Exact) -> SizeBound:
    """Decrease s using the sharp uniform factor from Proposition 3.2."""
    reciprocal = exact(reciprocal, "reciprocal exponent")
    if reciprocal < bound.reciprocal:
        raise ValueError("descent requires an output tail exponent no larger than the input")
    scale = bound.scale * _LOG_TWO**(bound.reciprocal - reciprocal)
    return SizeBound(bound.expression, reciprocal, scale, bound.nonnegative)


def bounded_raise(bound: SizeBound, reciprocal: Exact, premise: UpperBound) -> SizeBound:
    """Increase s using a matching pointwise upper bound for a nonnegative input."""
    reciprocal = exact(reciprocal, "reciprocal exponent")
    if not 0 < reciprocal <= bound.reciprocal:
        raise ValueError("bounded raise requires a positive reciprocal no larger than the input")
    if not bound.nonnegative or premise.expression != bound.expression:
        raise ValueError("bounded raise needs a matching upper bound and nonnegative input")
    ceiling = _positive_scale(premise.ceiling)
    ratio = reciprocal / bound.reciprocal
    return SizeBound(bound.expression, reciprocal, ceiling**(1 - ratio) * bound.scale**ratio, True)


def aggregate(bounds: Iterable[SizeBound], *, unit_factor: bool = False) -> SizeBound:
    """Sum bounds at a common tail exponent, retaining the aggregation constant."""
    bounds = tuple(bounds)
    if type(unit_factor) is not bool:
        raise TypeError("unit_factor must be a bool")
    if not bounds:
        raise ValueError("a sum requires at least one bound")
    reciprocal = bounds[0].reciprocal
    if any(bound.reciprocal != reciprocal for bound in bounds):
        raise ValueError("sum inputs must share their tail exponent")
    if unit_factor and reciprocal > 1:
        raise ValueError("the unit-factor sum rule requires s >= 1")
    factor = (1 + (reciprocal - 1) / _LOG_TWO)**reciprocal if reciprocal > 1 else 1
    return SizeBound(
        Add(*(bound.expression for bound in bounds)),
        reciprocal,
        factor * Add(*(bound.scale for bound in bounds)),
        all(bound.nonnegative for bound in bounds),
    )


@dataclass(frozen=True)
class DerivationNode:
    id: str
    rule: str
    operands: tuple[str, ...] = ()
    argument: object = None


def verify_derivation(
    givens: Mapping[str, SizeBound],
    nodes: Iterable[DerivationNode],
    goal: SizeBound,
    *,
    upper_bounds: Mapping[str, UpperBound] | None = None,
) -> CheckResult:
    """Check a topologically ordered derivation whose last node is its goal.

    Given nodes name an input. Power and descent take a rational argument.
    Bounded raise takes (reciprocal, upper-bound identifier). Product and
    sum have no argument, and sum_unit selects the unit-factor theorem.
    """
    proved = {}
    upper_bounds = {} if upper_bounds is None else upper_bounds
    for node in nodes:
        try:
            if type(node.id) is not str or not node.id or node.id in proved:
                raise ValueError("node identifiers must be nonempty and unique")
            operands = tuple(proved[operand] for operand in node.operands)
            if node.rule == "given" and not operands:
                result = givens[node.argument]
            elif node.rule == "power" and len(operands) == 1:
                result = power(operands[0], node.argument)
            elif node.rule == "descend" and len(operands) == 1:
                result = descend(operands[0], node.argument)
            elif node.rule == "bounded_raise" and len(operands) == 1:
                reciprocal, premise = node.argument
                result = bounded_raise(operands[0], reciprocal, upper_bounds[premise])
            elif node.rule == "product" and node.argument is None:
                result = product(operands)
            elif node.rule in ("sum", "sum_unit") and node.argument is None:
                result = aggregate(operands, unit_factor=node.rule == "sum_unit")
            else:
                raise ValueError("unknown rule or invalid operands")
            if type(result) is not SizeBound:
                raise TypeError("a given must be a SizeBound")
        except (KeyError, TypeError, ValueError) as error:
            return CheckResult("inconclusive", _SCOPE, f"Node {node.id}: {error}")
        proved[node.id] = result
    if not proved or result != goal:
        return CheckResult("inconclusive", _SCOPE, "The final judgement does not match the goal.")
    return CheckResult("verified", _SCOPE, "The derivation proves the requested size bound.", result)
