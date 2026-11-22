"""Exact rational coefficients on an explicitly declared positive orthant."""

from dataclasses import dataclass, field
from fractions import Fraction

from sympy import Poly, QQ, Symbol, ZZ
from sympy.polys.domains import FractionField
from sympy.polys.fields import FracElement


def _coefficient_sign(coefficients):
    """Return a proved sign at strictly positive parameter values, or None."""
    coefficients = tuple(coefficients)
    if all(c >= 0 for c in coefficients):
        return int(any(coefficients))
    if all(c <= 0 for c in coefficients):
        return -1
    return None


@dataclass(frozen=True)
class PositiveParameters:
    """Exact rational functions of independently strictly positive symbols.

    The sign check is sufficient and may leave a sign unresolved. To model
    sigma > rho > 0, declare rho and a positive gap e, and encode sigma as
    rho + e before constructing the polynomials and certificate.
    """

    symbols: tuple[Symbol, ...]
    field: FractionField = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        symbols = tuple(self.symbols)
        if not symbols or any(type(symbol) is not Symbol for symbol in symbols):
            raise TypeError("declare one or more parameter symbols")
        if len(set(symbols)) != len(symbols):
            raise ValueError("parameter symbols must be distinct")
        if any(symbol.is_positive is False for symbol in symbols):
            raise ValueError("parameter assumptions must allow strictly positive values")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "field", QQ.frac_field(*symbols))

    def rational(self, numerator: Poly, denominator: Poly) -> FracElement:
        """Construct a fraction from exact polynomials in the declared parameters."""
        for polynomial in (numerator, denominator):
            if not isinstance(polynomial, Poly) or polynomial.domain not in (ZZ, QQ):
                raise TypeError("a parameter numerator and denominator must be Poly over ZZ or QQ")
            if polynomial.gens != self.symbols:
                raise ValueError("use the declared parameter symbols in order")
        sign = _coefficient_sign(denominator.rep.to_dict().values())
        if sign == 0:
            raise ZeroDivisionError("the parameter denominator is zero")
        if sign is None:
            raise ValueError("the parameter denominator is not certified nonzero")
        value = self.field.from_sympy(numerator.as_expr() / denominator.as_expr())
        return self.coefficient(value)

    def coefficient(self, value) -> FracElement:
        """Validate an exact scalar, including its parameter poles."""
        if type(value) in (int, Fraction):
            return self.field.convert(value)
        if type(value) is not FracElement or value.field != self.field.field:
            raise TypeError("use int, Fraction or an element of the declared parameter field")
        if _coefficient_sign(value.denom.values()) not in (-1, 1):
            raise ValueError("the parameter denominator is not certified nonzero")
        return value

    def nonnegative(self, value) -> bool:
        """Check that all coefficients of the numerator have the needed sign."""
        value = self.coefficient(value)
        numerator_sign = _coefficient_sign(value.numer.values())
        denominator_sign = _coefficient_sign(value.denom.values())
        return numerator_sign in (0, denominator_sign)

    def nonzero(self, value) -> bool:
        """Check that an exact scalar has a strict sign throughout the orthant."""
        value = self.coefficient(value)
        return _coefficient_sign(value.numer.values()) in (-1, 1)
