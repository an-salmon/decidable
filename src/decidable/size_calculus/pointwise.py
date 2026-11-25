"""Guarded E1 pointwise certificates for exponential residuals."""

from __future__ import annotations

from fractions import Fraction

from sympy import Add, Dummy, Integer, Mul, Poly, Pow, QQ, Rational, Symbol, exp
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from ..exp_series import check_exp_series
from ..report import CheckResult
from .enclosures import exp_enclosure
from .moments import Exact, exact


class _OutsideE1(ValueError):
    pass


def _positive(polynomial: Poly, upper: Fraction | None) -> bool:
    return polynomial.eval(0) > 0 and not any(
        root > 0 and (upper is None or root < upper)
        for root in polynomial.real_roots()
    )


def check_exponential_residual(
    expression,
    coordinate: Symbol,
    *,
    upper: Exact | None = None,
    sample: Exact | None = None,
    degree: int = 16,
) -> CheckResult:
    """Check expression >= 0 on [0, upper) or the nonnegative half-line.

    Explicit inverses require a positive polynomial base throughout the
    interval. The chart supports exp(v) and exp(-v). After clearing the
    guarded denominators the residual must be Q(v) + P(v) exp(v).
    Give unevaluated powers and products to preserve original denominators.
    An optional exact sample may refute the original inequality.
    """
    if type(coordinate) is not Symbol:
        raise TypeError("use one real coordinate symbol")
    if coordinate not in (Symbol(coordinate.name), Symbol(coordinate.name, real=True)):
        raise ValueError("put coordinate restrictions in the interval, not on the symbol")
    upper = None if upper is None else exact(upper, "open upper endpoint")
    if upper is not None and upper <= 0:
        raise ValueError("the upper endpoint must exceed zero")
    scope = (f"All real {coordinate} in [0, {upper})" if upper is not None
             else f"All real {coordinate} >= 0")
    e = Dummy("exp")

    def lower(term):
        if type(term) in (int, Fraction):
            return Rational(term.numerator, term.denominator)
        if isinstance(term, Rational) or term == coordinate:
            return term
        if type(term) in (Add, Mul):
            return type(term)(*(lower(item) for item in term.args))
        if type(term) is exp and term.args[0] in (coordinate, -coordinate):
            return e if term.args[0] == coordinate else 1 / e
        if type(term) is Pow and isinstance(term.exp, Integer):
            base = lower(term.base)
            if term.exp < 0:
                if base != e:
                    try:
                        divisor = Poly(base, coordinate, domain=QQ)
                    except (CoercionFailed, PolynomialError, ValueError) as error:
                        raise _OutsideE1("an explicit inverse requires a polynomial base") from error
                    if not _positive(divisor, upper):
                        raise _OutsideE1("a denominator is not strictly positive on the interval")
            return base**term.exp
        raise _OutsideE1("expression contains a term outside the E1 chart")

    try:
        numerator, denominator = lower(expression).as_numer_denom()
        # Exp is positive. Verify any remaining polynomial denominator.
        denominator = Poly(denominator, e, domain=QQ.poly_ring(coordinate))
        terms = denominator.terms()
        if len(terms) != 1 or not _positive(Poly(terms[0][1], coordinate, domain=QQ), upper):
            raise _OutsideE1("the chart could not establish a positive denominator")
        polynomial = Poly(numerator, e, domain=QQ.poly_ring(coordinate))
        if polynomial.degree() > 1:
            raise _OutsideE1("the cleared residual has multiple exponential powers")
        p = Poly(polynomial.nth(1), coordinate, domain=QQ)
        q = Poly(polynomial.nth(0), coordinate, domain=QQ)
    except (CoercionFailed, PolynomialError, ValueError) as error:
        return CheckResult("unsupported", scope, str(error))

    if sample is not None:
        sample = exact(sample, "sample point")
        if type(degree) is not int or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if sample < 0 or (upper is not None and sample >= upper):
            raise ValueError("sample must lie in the declared interval")
        if sample < degree + 2:
            low, high = exp_enclosure(sample, degree)
            a = Fraction(p.eval(sample))
            b = Fraction(q.eval(sample))
            if a * (high if a > 0 else low) + b < 0:
                return CheckResult("disproved", scope, "An exact sample violates the residual.", sample)
    result = check_exp_series(p, q, tail_start="auto")
    return CheckResult(result.status, scope, result.reason, result.witness)
