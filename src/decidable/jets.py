"""Exact differential expressions in a caller-declared finite jet ring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from sympy import QQ, Add, Dummy, Expr, Integer, Mul, Poly, Pow, Rational, Symbol
from sympy import cancel, groebner, preorder_traversal

from .report import CheckResult


@dataclass(frozen=True)
class Jet:
    """A field derivative with slow and, if present, independent fast indices."""

    field: str
    index: tuple[int, ...]
    fast_index: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("a jet needs a nonempty field identifier")
        for index in (self.index, self.fast_index):
            if not isinstance(index, tuple) or any(type(value) is not int for value in index):
                raise TypeError("jet indices must be tuples of integers")
            if any(value < 0 for value in index):
                raise ValueError("jet indices must be nonnegative")


class JetSpace:
    """Declare a finite collection of jet symbols and independent coordinates.

    All symbols must be distinct, including by name. Parameters are
    constant under differentiation. A second coordinate tuple requires
    a distinct epsilon symbol, which is then a constant unit. In a
    two-scale space each jet has a full multi-index for each scale.
    """

    def __init__(
        self,
        coordinates: Sequence[Symbol],
        jets: Mapping[Jet, Symbol],
        *,
        parameters: Sequence[Symbol] = (),
        fast_coordinates: Sequence[Symbol] = (),
        epsilon: Symbol | None = None,
    ) -> None:
        coordinates = tuple(coordinates)
        fast_coordinates = tuple(fast_coordinates)
        parameters = tuple(parameters)
        if not coordinates:
            raise ValueError("at least one coordinate is required")
        if fast_coordinates and len(fast_coordinates) != len(coordinates):
            raise ValueError("the two coordinate tuples must have equal dimensions")
        if bool(fast_coordinates) != (epsilon is not None):
            raise ValueError("fast coordinates and an invertible epsilon must be declared together")
        if not isinstance(jets, Mapping):
            raise TypeError("jets must map Jet objects to symbols")
        jets = dict(jets)
        for jet in jets:
            if not isinstance(jet, Jet):
                raise TypeError("jet keys must be Jet objects")
            if len(jet.index) != len(coordinates) or len(jet.fast_index) != len(fast_coordinates):
                raise ValueError("jet indices must match the declared coordinate dimensions")
        symbols = coordinates + fast_coordinates + parameters + tuple(jets.values())
        symbols += () if epsilon is None else (epsilon,)
        if any(not isinstance(symbol, Symbol) or symbol.is_commutative is not True for symbol in symbols):
            raise TypeError("jet space generators must be commuting SymPy symbols")
        if len({symbol.name for symbol in symbols}) != len(symbols):
            raise ValueError("jet space generators must have distinct names")
        if epsilon is not None and epsilon.is_zero is True:
            raise ValueError("epsilon must be nonzero")
        self._coordinates = coordinates
        self._fast_coordinates = fast_coordinates
        self._jets = MappingProxyType(jets)
        self._epsilon = epsilon
        self._symbols = symbols

    def _expression(self, expression: Expr | int | Fraction) -> Expr:
        if type(expression) is int:
            expression = Integer(expression)
        elif type(expression) is Fraction:
            expression = Rational(expression.numerator, expression.denominator)
        if not isinstance(expression, Expr):
            raise TypeError("jet expressions must be exact SymPy expressions, int, or Fraction")
        for term in preorder_traversal(expression):
            if isinstance(term, Symbol):
                if term not in self._symbols:
                    raise ValueError(f"undeclared jet ring symbol {term}")
            elif isinstance(term, Pow):
                if not isinstance(term.exp, Integer):
                    raise ValueError("jet expressions require integer exponents")
                if term.exp < 0:
                    self._check_unit(term.base)
            elif not isinstance(term, (Add, Mul, Rational)):
                raise ValueError("jet expressions require rational coefficients and polynomial operations")
        return expression

    def _check_unit(self, base: Expr) -> None:
        coefficient, factors = cancel(base).as_coeff_Mul()
        if coefficient and isinstance(coefficient, Rational):
            if factors == 1:
                return
            factor, exponent = factors.as_base_exp()
            if self._epsilon is not None and factor == self._epsilon and isinstance(exponent, Integer):
                return
        raise ValueError("only nonzero rational constants and powers of declared epsilon can be inverted")

    def _derivative(self, expression: Expr, axis: int, *, fast: bool = False) -> Expr:
        coordinates = self._fast_coordinates if fast else self._coordinates
        if type(axis) is not int:
            raise TypeError("derivative axis must be an integer")
        if not 0 <= axis < len(coordinates):
            raise ValueError("derivative axis is outside the declared coordinates")
        result = expression.diff(coordinates[axis])
        free_symbols = expression.free_symbols
        for jet, symbol in self._jets.items():
            if symbol not in free_symbols:
                continue
            partial = expression.diff(symbol)
            if partial == 0:
                continue
            index = jet.fast_index if fast else jet.index
            next_index = index[:axis] + (index[axis] + 1,) + index[axis + 1 :]
            successor = (
                Jet(jet.field, jet.index, next_index)
                if fast
                else Jet(jet.field, next_index, jet.fast_index)
            )
            if successor not in self._jets:
                raise ValueError(f"derivative requires undeclared jet {successor}")
            result += partial * self._jets[successor]
        return result


def total_derivative(space: JetSpace, expression: Expr | int | Fraction, axis: int) -> Expr:
    """Apply the chain rule along the indicated zero-based slow coordinate."""
    return space._derivative(space._expression(expression), axis)


def two_scale_derivative(space: JetSpace, expression: Expr | int | Fraction, axis: int) -> Expr:
    """Apply the slow total derivative plus epsilon inverse times the fast one."""
    if space._epsilon is None:
        raise ValueError("two-scale derivatives require fast coordinates and invertible epsilon")
    expression = space._expression(expression)
    return space._derivative(expression, axis) + space._derivative(expression, axis, fast=True) / space._epsilon


def check_jet_identity(
    space: JetSpace,
    lhs: Expr | int | Fraction,
    rhs: Expr | int | Fraction = 0,
    *,
    relations: Iterable[Expr | int | Fraction] = (),
    lhs_derivatives: Iterable[int] = (),
    rhs_derivatives: Iterable[int] = (),
    two_scale: bool = False,
) -> CheckResult:
    """Test ideal membership of the differentiated residual over QQ.

    Each side's derivative axes are applied in the given order. Relations
    are expressions assumed zero and are used as supplied. With epsilon
    declared invertible, denominators are cleared and an extra polynomial
    relation enforces its invertibility. Malformed expressions, missing
    successor jets and invalid axes raise TypeError or ValueError.
    A nonzero remainder leaves the analytic identity inconclusive.
    """
    if type(two_scale) is not bool:
        raise TypeError("two_scale must be a bool")
    if two_scale and space._epsilon is None:
        raise ValueError("two-scale derivatives require fast coordinates and invertible epsilon")
    derivative = two_scale_derivative if two_scale else total_derivative
    lhs = space._expression(lhs)
    rhs = space._expression(rhs)
    for axis in lhs_derivatives:
        lhs = derivative(space, lhs, axis)
    for axis in rhs_derivatives:
        rhs = derivative(space, rhs, axis)

    generators = space._symbols
    equations = [cancel(space._expression(relation)).as_numer_denom()[0] for relation in relations]
    relation_count = len(equations)
    scale_scope = ""
    if space._epsilon is not None:
        inverse = Dummy("epsilon_inverse")
        equations.append(space._epsilon * inverse - 1)
        generators += (inverse,)
        scale_scope = f" with {space._epsilon} invertible"
    names = ", ".join(map(str, space._symbols))
    scope = f"Finite jet ring over QQ on ({names}){scale_scope}, with {relation_count} supplied relations"
    residual = cancel(lhs - rhs).as_numer_denom()[0]
    if equations:
        _, remainder = groebner(equations, *generators, domain=QQ).reduce(residual)
    else:
        remainder = Poly(residual, *generators, domain=QQ).as_expr()
    if not remainder:
        return CheckResult(
            status="verified", scope=scope, reason="Residual belongs to the supplied finite relation ideal."
        )
    if space._epsilon is not None:
        remainder = remainder.subs(inverse, 1 / space._epsilon)
    return CheckResult(
        status="inconclusive",
        scope=scope,
        reason="A nonzero ideal remainder leaves the analytic identity unresolved.",
        witness=remainder,
    )
