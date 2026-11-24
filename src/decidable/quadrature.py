"""Exact definite integrals of powers and inverse-scale exponentials."""

from __future__ import annotations

from fractions import Fraction

from sympy import Dummy, Expr, Poly, Rational, S, ZZ, exp, log

from ._quadrature_identity import same_closed_form
from .parameters import PositiveParameters
from .report import CheckResult
from .series import ExactScalar


class _DivergentIntegral(ValueError):
    """A supported integrand has a divergent improper endpoint."""


def _scale(
    value: ExactScalar,
    name: str,
    parameters: PositiveParameters | None,
    *,
    allow_zero: bool = False,
) -> Expr:
    if parameters is None:
        if type(value) not in (int, Fraction):
            raise TypeError(f"{name} must be an int or Fraction without parameters")
        if value > 0 or (allow_zero and value == 0):
            return Rational(value.numerator, value.denominator)
    else:
        scalar = parameters.coefficient(value)
        if allow_zero and scalar == 0:
            return S.Zero
        if parameters.nonnegative(scalar) and parameters.nonzero(scalar):
            return scalar.as_expr()
    raise ValueError(f"{name} must be certified strictly positive"
                     + (" or exactly zero" if allow_zero else ""))


def _power_value(exponent: Rational, lower: Expr, upper: Expr | None) -> Expr:
    primitive_power = exponent + 1
    if lower == 0 and primitive_power <= 0:
        raise _DivergentIntegral("The integral diverges at zero.")
    if upper is None and primitive_power >= 0:
        raise _DivergentIntegral("The integral diverges at positive infinity.")
    if primitive_power == 0:
        return log(upper) - log(lower)

    lower_value = S.Zero if lower == 0 else lower**primitive_power
    upper_value = S.Zero if upper is None else upper**primitive_power
    return (upper_value - lower_value) / primitive_power


def _exponential_value(
    degree: int, lower: Expr, upper: Expr | None, decay: Expr
) -> Expr:
    variable = Dummy("u")
    coefficients = [1]
    for index in range(degree, 0, -1):
        coefficients.append(index * coefficients[-1])
    polynomial = Poly.from_list(coefficients, variable, domain=ZZ)

    # For u = c/t, this identity verifies the derivative of
    # c^(-degree-1) exp(-u) P(u) is t^(-degree-2) exp(-c/t).
    if polynomial - polynomial.diff() != Poly(variable**degree, variable, domain=ZZ):
        raise ArithmeticError("inverse-scale exponential primitive failed derivative replay")

    expression = polynomial.as_expr()

    def at_finite_positive(endpoint: Expr) -> Expr:
        substitution = decay / endpoint
        return exp(-substitution) * expression.subs(variable, substitution)

    # exp(-u) P(u) tends to zero as u tends to infinity, and to P(0) at u=0.
    lower_value = S.Zero if lower == 0 else at_finite_positive(lower)
    upper_value = polynomial.eval(0) if upper is None else at_finite_positive(upper)
    return decay**(-degree - 1) * (upper_value - lower_value)


def quadrature_value(
    exponent: int | Fraction,
    lower: ExactScalar,
    upper: ExactScalar | None,
    *,
    decay: ExactScalar | None = None,
    parameters: PositiveParameters | None = None,
) -> Expr:
    """Compute the exact definite integral of t**exponent over the bounds.

    A supplied decay c multiplies the integrand by exp(-c/t) and requires an
    integer exponent at most -2. Lower zero and upper None denote improper
    endpoints at zero and positive infinity. Other bounds and c must be
    certifiably strictly positive. Two finite positive bounds are oriented.

    Scalars are int or Fraction, or exact rational-function coefficients in
    the supplied positive parameter field. An unsupported argument or a
    divergent integral raises TypeError or ValueError.
    """
    if type(exponent) not in (int, Fraction):
        raise TypeError("exponent must be an int or Fraction")
    if parameters is not None and type(parameters) is not PositiveParameters:
        raise TypeError("parameters must be PositiveParameters")
    power = Rational(exponent.numerator, exponent.denominator)
    if decay is not None and (power.q != 1 or power > -2):
        raise ValueError("inverse-scale exponential integration requires an integer exponent at most -2")

    lower_value = _scale(lower, "lower bound", parameters, allow_zero=True)
    upper_value = None if upper is None else _scale(upper, "upper bound", parameters)
    if decay is None:
        return _power_value(power, lower_value, upper_value)
    coefficient = _scale(decay, "decay", parameters)
    return _exponential_value(-int(power) - 2, lower_value, upper_value, coefficient)


def verify_quadrature(
    exponent: int | Fraction,
    lower: ExactScalar,
    upper: ExactScalar | None,
    claimed: Expr | int | Fraction,
    *,
    decay: ExactScalar | None = None,
    parameters: PositiveParameters | None = None,
) -> CheckResult:
    """Check a finite closed form against the supported exact definite integral.

    A formally unequal expression is inconclusive because distinct formal
    atoms can agree as analytic functions. A divergent supported integral
    disproves the assertion of a finite value.
    """
    scope = "Exact definite integral with the supplied bounds"
    if type(parameters) is PositiveParameters:
        names = ", ".join(map(str, parameters.symbols))
        scope += f" for all positive values of {names}"
    divergence = None
    try:
        try:
            expected = quadrature_value(exponent, lower, upper, decay=decay, parameters=parameters)
        except _DivergentIntegral as error:
            divergence = str(error)
            # Use a finite reference to validate the claim's syntax and domain.
            expected = S.Zero
        matches = same_closed_form(expected, claimed, parameters)
    except (TypeError, ValueError) as error:
        return CheckResult("unsupported", scope, str(error))
    if divergence is not None:
        return CheckResult("disproved", scope, divergence)
    if matches:
        return CheckResult("verified", scope, "The endpoint difference equals the claimed value.")
    return CheckResult("inconclusive", scope, "The formal closed forms differ.")
