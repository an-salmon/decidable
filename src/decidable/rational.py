"""Exact identities in a rational-function field."""

from __future__ import annotations

from sympy import Expr, Poly, QQ, ZZ

from .parameters import PositiveParameters


def _validate_polynomial(
    polynomial: Poly,
    generators: tuple[Expr, ...] | None = None,
    *,
    parameters: PositiveParameters | None = None,
) -> Poly:
    if parameters is not None and type(parameters) is not PositiveParameters:
        raise TypeError("parameters must be PositiveParameters")
    if not isinstance(polynomial, Poly):
        raise TypeError("polynomial arguments must be SymPy Poly objects")
    if polynomial.domain not in (ZZ, QQ):
        if parameters is None or polynomial.domain != parameters.field:
            raise TypeError("use ZZ, QQ or the declared parameter field for coefficients")
        for coefficient in polynomial.rep.to_dict().values():
            parameters.coefficient(coefficient)
    if parameters is not None and set(polynomial.gens).intersection(parameters.symbols):
        raise ValueError("a parameter cannot also be a polynomial generator")
    if generators is not None and polynomial.gens != generators:
        raise ValueError("polynomials must have the same ordered generators")
    return polynomial


def rational_identity(
    left_numerator: Poly,
    left_denominator: Poly,
    right_numerator: Poly,
    right_denominator: Poly,
    *,
    parameters: PositiveParameters | None = None,
) -> bool:
    """Decide whether two fractions represent the same rational function.

    Supply polynomials over ZZ or QQ with the same ordered generators. The
    equality holds as a field identity and makes no claim about definedness
    at points where an input denominator vanishes.

    Raises TypeError for non-polynomials or unsupported coefficient domains,
    ValueError for inconsistent generators, and ZeroDivisionError for a zero
    denominator polynomial.
    """
    generators = _validate_polynomial(left_numerator, parameters=parameters).gens
    for polynomial in (left_denominator, right_numerator, right_denominator):
        _validate_polynomial(polynomial, generators, parameters=parameters)
    if left_denominator.is_zero or right_denominator.is_zero:
        raise ZeroDivisionError("denominators must be nonzero polynomials")

    cross_product = (
        left_numerator * right_denominator - right_numerator * left_denominator
    )
    return cross_product.is_zero
