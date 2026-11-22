"""Exact realizations of finite scalar recurrences as formal series."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction

from sympy import Poly, QQ, Symbol

from .series import RationalSeries, _exact_coefficient


def recurrence_to_series(
    coefficients: Iterable[int | Fraction],
    initials: Iterable[int | Fraction],
    *,
    scale: Symbol = Symbol("z"),
) -> RationalSeries:
    """Return the formal series for a rational constant-coefficient recurrence.

    Supply the d coefficients (a_1, ..., a_d) and d initial terms
    (s_0, ..., s_{d-1}) as integers or Fractions. The recurrence
    s_n = sum(a_j*s_{n-j}) is imposed from n=d onward. The numerator
    is the truncation of the denominator times the initial polynomial
    to degrees below d. Trailing zero coefficients preserve the given
    start index. When d is zero, the series is identically zero.

    Both polynomials are over QQ in the supplied scale symbol. This is
    a formal identity with no sign or convergence conditions.
    """
    if not isinstance(scale, Symbol):
        raise TypeError("scale must be a SymPy Symbol")
    coefficients = tuple(_exact_coefficient(value, None) for value in coefficients)
    initials = tuple(_exact_coefficient(value, None) for value in initials)
    order = len(coefficients)
    if len(initials) != order:
        raise ValueError("supply one initial term per recurrence coefficient")

    z = Poly(scale, scale, domain=QQ)
    denominator = Poly.from_dict(
        {(0,): QQ.one, **{(j,): -a for j, a in enumerate(coefficients, start=1)}},
        scale,
        domain=QQ,
    )
    initial_polynomial = Poly.from_dict(
        {(j,): value for j, value in enumerate(initials)}, scale, domain=QQ
    )
    numerator = (denominator * initial_polynomial).rem(z**order)
    return numerator, denominator
