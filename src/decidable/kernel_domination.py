"""Certify coefficientwise domination between scalar formal series."""

from __future__ import annotations

from sympy import S

from .report import CheckResult
from .series import RationalSeries, _scale_polynomial, _validate_series


_SCOPE = "[z^n](upper - lower) >= 0 for all integers n >= 0"


def _negative_coefficient(index, coefficient) -> CheckResult:
    return CheckResult(
        status="disproved",
        scope=_SCOPE,
        reason=f"The difference has coefficient {coefficient} at index {index}.",
        witness={"index": index, "coefficient": coefficient},
    )


def check_kernel_domination(
    upper: RationalSeries, lower: RationalSeries, *, tail_start: int | None = 0
) -> CheckResult:
    """Check that `upper` dominates `lower` coefficientwise.

    Each kernel is a pair of univariate SymPy Poly objects over ZZ or QQ.
    All four polynomials must use the same generator and each input
    denominator must have a nonzero constant coefficient. Malformed inputs
    raise TypeError or ValueError. After exact cancellation of the difference,
    polynomials and fractions with distinct simple positive rational poles
    are supported. Other poles yield `unsupported`.

    For a nonpolynomial difference, check each index below `tail_start`,
    increasing the cutoff if needed to include the whole polynomial part.
    The remaining coefficients have the form sum(c * r**n). Dividing by
    the largest ratio R**n gives a uniform lower bound by evaluating each
    negative lower-ratio contribution at the cutoff and dropping the
    nonnegative lower-ratio contributions. An inadequate bound yields
    `inconclusive`. A checked negative coefficient disproves domination
    and the witness contains its index and exact value.

    With `tail_start=None`, automatically find a finite cutoff and decide
    domination for every supported difference. A negative leading amplitude
    gives a negative coefficient once the lower-ratio terms are too small to
    offset it. The automatic search and prefix check have no resource budget.

    Verified domination holds after convolution against any common scalar
    formal series with nonnegative coefficients.
    """
    if tail_start is not None:
        if type(tail_start) is not int:
            raise TypeError("tail_start must be an integer or None")
        if tail_start < 0:
            raise ValueError("tail_start must be nonnegative")

    upper_num, upper_den = _validate_series(upper)
    if len(upper_num.gens) != 1:
        raise ValueError("kernel polynomials must use exactly one generator")
    lower_num, lower_den = _validate_series(lower, upper_num.gens)

    numerator, denominator = (upper_num * lower_den - lower_num * upper_den).cancel(
        upper_den * lower_den, include=True
    )
    polynomial, remainder = numerator.div(denominator)
    if remainder.is_zero:
        for (index,), coefficient in sorted(polynomial.terms()):
            if coefficient < 0:
                return _negative_coefficient(index, coefficient)
        return CheckResult(
            status="verified",
            scope=_SCOPE,
            reason="The difference is a polynomial with nonnegative coefficients.",
        )

    _, factors = denominator.factor_list()
    if any(
        multiplicity != 1 or factor.degree() != 1 or factor.nth(0) * factor.nth(1) >= 0
        for factor, multiplicity in factors
    ):
        return CheckResult(
            status="unsupported",
            scope=_SCOPE,
            reason="The reduced denominator must have distinct simple positive rational poles.",
        )

    scale = _scale_polynomial(denominator)
    one = scale * 0 + 1
    amplitudes = []
    reconstructed = polynomial * denominator
    for factor, _ in factors:
        pole = -factor.nth(0) / factor.nth(1)
        ratio = 1 / pole
        quotient = denominator.exquo(one - scale.mul_ground(ratio))
        amplitude = remainder.eval(pole) / quotient.eval(pole)
        amplitudes.append((ratio, amplitude))
        reconstructed += quotient.mul_ground(amplitude)

    if reconstructed != numerator:
        return CheckResult(
            status="inconclusive",
            scope=_SCOPE,
            reason="The partial fraction identity did not replay.",
        )

    def coefficient_at(index):
        return polynomial.nth(index) + sum(
            amplitude * ratio**index for ratio, amplitude in amplitudes
        )

    amplitudes.sort()
    largest_ratio, leading_amplitude = amplitudes[-1]
    lower_terms = [(ratio / largest_ratio, amplitude) for ratio, amplitude in amplitudes[:-1]]

    def negative_bound_at(index):
        return sum(
            (-amplitude * ratio**index for ratio, amplitude in lower_terms if amplitude < 0),
            S.Zero,
        )

    polynomial_end = 0 if polynomial.is_zero else polynomial.degree() + 1
    if tail_start is None:
        tail_start = polynomial_end
        if leading_amplitude < 0:
            while sum(
                abs(amplitude) * ratio**tail_start for ratio, amplitude in lower_terms
            ) >= -leading_amplitude:
                tail_start = max(1, 2 * tail_start)
            return _negative_coefficient(tail_start, coefficient_at(tail_start))
        while negative_bound_at(tail_start) > leading_amplitude:
            tail_start = max(1, 2 * tail_start)
    else:
        tail_start = max(tail_start, polynomial_end)

    for index in range(tail_start):
        coefficient = coefficient_at(index)
        if coefficient < 0:
            return _negative_coefficient(index, coefficient)

    negative_bound = negative_bound_at(tail_start)
    bound_witness = {
        "tail_start": tail_start,
        "leading_amplitude": leading_amplitude,
        "negative_bound": negative_bound,
    }
    if leading_amplitude < negative_bound:
        return CheckResult(
            status="inconclusive",
            scope=_SCOPE,
            reason=f"The prefix passed, but the tail bound at index {tail_start} is negative.",
            witness=bound_witness,
        )
    return CheckResult(
        status="verified",
        scope=_SCOPE,
        reason=f"The checked prefix and the tail bound from index {tail_start} cover every coefficient.",
        witness=bound_witness,
    )
