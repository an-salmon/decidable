"""Stretched-exponential rules over exact affine real parameter domains."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction

from sympy import Abs, Add, Expr, Integer, Mul, Pow, Rational, Symbol, log

from ..linear import ExactForm, linear_feasible, linear_implies
from .moments import Exact, exact
from .typing import UpperBound, _quantity, _sign


_LOG_TWO = log(2)


def _form(expression: Expr, parameters: tuple[Symbol, ...]) -> ExactForm:
    if isinstance(expression, Rational):
        return (Fraction(expression.p, expression.q),) + (Fraction(0),) * len(parameters)
    if expression in parameters:
        return (Fraction(0), *(Fraction(expression == parameter) for parameter in parameters))
    if type(expression) is Add:
        return tuple(sum(entries) for entries in zip(*(_form(term, parameters)
                                                       for term in expression.args)))
    if type(expression) is Mul:
        scalar = Fraction(1)
        affine = None
        for term in expression.args:
            form = _form(term, parameters)
            if not any(form[1:]):
                scalar *= form[0]
            elif affine is None:
                affine = form
            else:
                raise ValueError("expected a rational affine form")
        if affine is None:
            return (scalar,) + (Fraction(0),) * len(parameters)
        return tuple(scalar * entry for entry in affine)
    raise ValueError("expected a rational affine form in the declared parameters")


def _affine_expression(row: ExactForm, parameters: tuple[Symbol, ...]) -> Expr:
    return Rational(row[0]) + Add(*(Rational(value) * symbol
                                  for value, symbol in zip(row[1:], parameters)))


@dataclass(frozen=True)
class AffineDomain:
    """A nonempty subset of real parameter space with exact affine constraints.

    Each expression in inequalities stands for >= 0, and expressions in
    strict_inequalities stand for > 0. Use plain or real SymPy symbols with
    distinct names. All sign restrictions belong in the constraint lists.
    """

    parameters: tuple[Symbol, ...]
    inequalities: tuple[Expr | Exact, ...] = ()
    strict_inequalities: tuple[Expr | Exact, ...] = ()
    _rows: tuple[ExactForm, ...] = field(init=False, repr=False, compare=False)
    _strict_rows: tuple[ExactForm, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        parameters = tuple(self.parameters)
        if any(type(symbol) is not Symbol or
               symbol not in (Symbol(symbol.name), Symbol(symbol.name, real=True))
               for symbol in parameters):
            raise TypeError("use plain or real parameter symbols with explicit affine constraints")
        if len({symbol.name for symbol in parameters}) != len(parameters):
            raise ValueError("parameter names must be distinct")

        rows = tuple(_form(_quantity(expression), parameters) for expression in self.inequalities)
        strict_rows = tuple(_form(_quantity(expression), parameters)
                            for expression in self.strict_inequalities)
        if not linear_feasible(len(parameters), inequalities=rows, strict_inequalities=strict_rows):
            raise ValueError("the affine parameter domain is empty")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "inequalities", tuple(_affine_expression(row, parameters) for row in rows))
        object.__setattr__(self, "strict_inequalities",
                           tuple(_affine_expression(row, parameters) for row in strict_rows))
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_strict_rows", strict_rows)

    def affine(self, expression: Expr | Exact) -> Expr:
        """Validate and normalize a rational affine expression in the parameters."""
        return _affine_expression(_form(_quantity(expression), self.parameters), self.parameters)

    def nonnegative(self, expression: Expr | Exact) -> bool:
        """Decide whether the affine expression is everywhere nonnegative."""
        return linear_implies(len(self.parameters), _form(_quantity(expression), self.parameters),
                              inequalities=self._rows, strict_inequalities=self._strict_rows)

    def positive(self, expression: Expr | Exact) -> bool:
        """Decide strict positivity by checking the weak negation for infeasibility."""
        negation = tuple(-value for value in _form(_quantity(expression), self.parameters))
        return not linear_feasible(len(self.parameters), inequalities=(*self._rows, negation),
                                   strict_inequalities=self._strict_rows)

    def _restrict(self, expression: Expr, *, strict: bool) -> AffineDomain:
        if strict:
            return AffineDomain(self.parameters, self.inequalities,
                                (*self.strict_inequalities, expression))
        return AffineDomain(self.parameters, (*self.inequalities, expression), self.strict_inequalities)


def _parameter_exponent(expression: Expr, domain: AffineDomain) -> None:
    if isinstance(expression, Rational) or expression in domain.parameters:
        return
    if type(expression) in (Add, Mul):
        for term in expression.args:
            _parameter_exponent(term, domain)
        return
    if type(expression) is Pow and isinstance(expression.exp, Integer):
        _parameter_exponent(expression.base, domain)
        if expression.exp < 0 and not (domain.positive(expression.base)
                                       or domain.positive(-expression.base)):
            raise ValueError("a scale exponent denominator must be nonzero on the entire domain")
        return
    raise ValueError("scale exponents must be rational expressions in the declared parameters")


def _positive_scale(expression: Expr | Exact, domain: AffineDomain) -> Expr:
    expression = _quantity(expression)
    if expression == _LOG_TWO or (isinstance(expression, Rational) and expression > 0):
        return expression
    if type(expression) is Symbol and expression not in domain.parameters:
        if expression.is_positive is True and all(expression.name != p.name for p in domain.parameters):
            return expression
        raise ValueError("scale symbols outside the affine domain must be independently positive")
    if expression.free_symbols.issubset(domain.parameters):
        try:
            affine = domain.affine(expression)
        except ValueError:
            pass
        else:
            if domain.positive(affine):
                return affine
            raise ValueError("the affine scale must be strictly positive on the entire domain")
    if type(expression) in (Add, Mul):
        for term in expression.args:
            _positive_scale(term, domain)
        return expression
    if expression.func is Abs:
        argument = expression.args[0]
        try:
            return _positive_scale(argument, domain)
        except ValueError:
            return _positive_scale(-argument, domain)
    if type(expression) is Pow:
        _positive_scale(expression.base, domain)
        _parameter_exponent(expression.exp, domain)
        return expression
    raise ValueError("the scale must be a certified positive exact expression")


@dataclass(frozen=True)
class ParametricSizeBound:
    """X <= O_s(scale) for every parameter value in domain, with reciprocal 1/s affine."""

    expression: Expr
    reciprocal: Expr | Exact
    scale: Expr | Exact
    nonnegative: bool
    domain: AffineDomain

    def __post_init__(self):
        if type(self.domain) is not AffineDomain:
            raise TypeError("a parametric size bound needs an AffineDomain")
        expression = _quantity(self.expression)
        _sign(expression, self.nonnegative)
        reciprocal = self.domain.affine(self.reciprocal)
        if not self.domain.positive(reciprocal):
            raise ValueError("the reciprocal exponent must be positive on the entire domain")
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "reciprocal", reciprocal)
        object.__setattr__(self, "scale", _positive_scale(self.scale, self.domain))


def _require_bound(bound: ParametricSizeBound) -> None:
    if type(bound) is not ParametricSizeBound:
        raise TypeError("parametric rules require ParametricSizeBound inputs")


def _common_domain(bounds: tuple[ParametricSizeBound, ...]) -> AffineDomain:
    if not bounds:
        raise ValueError("supply at least one parametric bound")
    for bound in bounds:
        _require_bound(bound)
    domain = bounds[0].domain
    if any(bound.domain != domain for bound in bounds):
        raise ValueError("all bounds must use the same parameter domain")
    return domain


def parametric_power(bound: ParametricSizeBound, exponent: Exact) -> ParametricSizeBound:
    """Apply a fixed positive rational power to a nonnegative input."""
    _require_bound(bound)
    exponent = Rational(exact(exponent, "power"))
    if exponent <= 0 or not bound.nonnegative:
        raise ValueError("a power requires a positive exponent and nonnegative input")
    return ParametricSizeBound(bound.expression**exponent, bound.reciprocal * exponent,
                               bound.scale**exponent, True, bound.domain)


def parametric_product(bounds: Iterable[ParametricSizeBound]) -> ParametricSizeBound:
    """Apply the dependence-free product rule on a common parameter domain."""
    bounds = tuple(bounds)
    domain = _common_domain(bounds)
    if not all(bound.nonnegative for bound in bounds):
        raise ValueError("a product requires nonnegative inputs")
    return ParametricSizeBound(Mul(*(bound.expression for bound in bounds)),
                               Add(*(bound.reciprocal for bound in bounds)),
                               Mul(*(bound.scale for bound in bounds)), True, domain)


def parametric_descend(bound: ParametricSizeBound, reciprocal: Expr | Exact) -> ParametricSizeBound:
    """Check the direction of descent everywhere and retain its sharp scale factor."""
    _require_bound(bound)
    domain = bound.domain
    reciprocal = domain.affine(reciprocal)
    if not domain.nonnegative(reciprocal - bound.reciprocal):
        raise ValueError("descent cannot increase the tail exponent anywhere in the domain")
    scale = bound.scale * _LOG_TWO**(bound.reciprocal - reciprocal)
    return ParametricSizeBound(bound.expression, reciprocal, scale, bound.nonnegative, domain)


def parametric_bounded_raise(
    bound: ParametricSizeBound, reciprocal: Expr | Exact, premise: UpperBound
) -> ParametricSizeBound:
    """Use a matching uniform ceiling to raise the tail exponent."""
    _require_bound(bound)
    domain = bound.domain
    reciprocal = domain.affine(reciprocal)
    if not domain.positive(reciprocal):
        raise ValueError("the output reciprocal must be positive throughout the domain")
    if not domain.nonnegative(bound.reciprocal - reciprocal):
        raise ValueError("bounded raise cannot decrease the tail exponent anywhere in the domain")
    if type(premise) is not UpperBound or not bound.nonnegative or premise.expression != bound.expression:
        raise ValueError("bounded raise requires a matching upper bound and nonnegative input")
    ceiling = _positive_scale(premise.ceiling, domain)
    ratio = reciprocal / bound.reciprocal
    scale = ceiling**(1 - ratio) * bound.scale**ratio
    return ParametricSizeBound(bound.expression, reciprocal, scale, True, domain)


def parametric_aggregate(
    bounds: Iterable[ParametricSizeBound], *, unit_factor: bool = False
) -> tuple[ParametricSizeBound, ...]:
    """Sum bounds on one common domain and return one or two exhaustive cases.

    A crossing domain is explicitly split at reciprocal <= 1 and reciprocal
    > 1. Unit factor requires that reciprocal <= 1 everywhere in the original
    domain. The boundary itself belongs to the first branch.
    """
    bounds = tuple(bounds)
    domain = _common_domain(bounds)
    if type(unit_factor) is not bool:
        raise TypeError("unit_factor must be a bool")
    reciprocal = bounds[0].reciprocal
    for bound in bounds[1:]:
        difference = bound.reciprocal - reciprocal
        if not domain.nonnegative(difference) or not domain.nonnegative(-difference):
            raise ValueError("sum inputs must share a tail exponent throughout the domain")

    expression = Add(*(bound.expression for bound in bounds))
    scale = Add(*(bound.scale for bound in bounds))
    nonnegative = all(bound.nonnegative for bound in bounds)
    if domain.nonnegative(1 - reciprocal):
        return (ParametricSizeBound(expression, reciprocal, scale, nonnegative, domain),)
    if unit_factor:
        raise ValueError("unit-factor aggregation requires reciprocal <= 1 throughout the domain")

    large_scale = (1 + (reciprocal - 1) / _LOG_TWO)**reciprocal * scale
    if domain.positive(reciprocal - 1):
        return (ParametricSizeBound(expression, reciprocal, large_scale, nonnegative, domain),)
    small = domain._restrict(1 - reciprocal, strict=False)
    large = domain._restrict(reciprocal - 1, strict=True)
    return (ParametricSizeBound(expression, reciprocal, scale, nonnegative, small),
            ParametricSizeBound(expression, reciprocal, large_scale, nonnegative, large))
