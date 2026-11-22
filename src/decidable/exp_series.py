"""Exact Taylor-series certificates for polynomial times exponential sums.

The normalized coefficient n! [x^n](P exp(x) + Q) equals H(n) + n! q_n,
where H(t) is the sum of p_j times the falling factorial (t)_j. For
integer k >= 0, H(N + k) equals the sum of binomial(k, j) times the
j-th forward difference of H at N. Nonnegative differences therefore
control the entire integer tail after Q's last coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial, perm
from typing import Literal

from sympy import Poly, Symbol

from .rational import _validate_polynomial
from .report import CheckResult


_SCOPE = "P(x) exp(x) + Q(x) >= 0 for all real x >= 0"


@dataclass(frozen=True)
class ExpSeriesCertificate:
    """Normalized Taylor coefficients below tail_start and forward differences at it."""

    tail_start: int
    prefix_coefficients: tuple[Fraction, ...]
    forward_differences: tuple[Fraction, ...]


def _polynomials(P: Poly, Q: Poly) -> tuple[Poly, Poly]:
    P = _validate_polynomial(P)
    Q = _validate_polynomial(Q, P.gens)
    if len(P.gens) != 1 or not isinstance(P.gen, Symbol):
        raise ValueError("use one common symbolic polynomial generator")
    return P, Q


def _minimum_tail_start(Q: Poly) -> int:
    return 0 if Q.is_zero else int(Q.degree()) + 1


def _validate_tail_start(tail_start: int) -> None:
    if type(tail_start) is not int:
        raise TypeError("tail_start must be an integer")
    if tail_start < 0:
        raise ValueError("tail_start must be nonnegative")


def _tail_size(P: Poly) -> int:
    return 1 if P.is_zero else int(P.degree()) + 1


def _falling_terms(P: Poly) -> tuple[tuple[int, Fraction], ...]:
    return tuple((j, Fraction(c)) for (j,), c in P.terms() if c)


def _tail_value(terms: tuple[tuple[int, Fraction], ...], n: int) -> Fraction:
    return sum((coefficient * perm(n, j) for j, coefficient in terms), Fraction(0))


def _forward_differences(
    terms: tuple[tuple[int, Fraction], ...], tail_start: int, size: int
) -> tuple[Fraction, ...]:
    # H(t) has degree at most deg P. Its first deg P + 1 forward
    # differences determine its Newton expansion at tail_start.
    row = [_tail_value(terms, tail_start + i) for i in range(size)]
    differences = []
    while row:
        differences.append(row[0])
        row = [right - left for left, right in zip(row, row[1:])]

    return tuple(differences)


def _certificate(P: Poly, Q: Poly, tail_start: int) -> ExpSeriesCertificate:
    terms = _falling_terms(P)
    q_terms = {j: factorial(j) * Fraction(c) for (j,), c in Q.terms() if c}
    prefix = tuple(_tail_value(terms, n) + q_terms.get(n, 0) for n in range(tail_start))
    differences = _forward_differences(terms, tail_start, _tail_size(P))
    return ExpSeriesCertificate(tail_start, prefix, differences)


def _negative_coefficient(index: int) -> CheckResult:
    return CheckResult(
        status="inconclusive", scope=_SCOPE,
        reason=f"The normalized Taylor coefficient at index {index} is negative.",
        witness=index,
    )


def verify_exp_series_certificate(
    P: Poly, Q: Poly, certificate: ExpSeriesCertificate
) -> CheckResult:
    """Replay a certificate against univariate SymPy Poly inputs over ZZ or QQ.

    A rejected certificate leaves the function inequality inconclusive. Use
    check_exp_series to detect a violation at x=0 independently of replay.
    Inputs outside the polynomial or certificate types raise TypeError or
    ValueError. A structurally valid certificate with incorrect data, negative
    coefficients or negative forward differences is rejected.
    """
    P, Q = _polynomials(P, Q)
    if not isinstance(certificate, ExpSeriesCertificate):
        raise TypeError("certificate must be an ExpSeriesCertificate")
    _validate_tail_start(certificate.tail_start)
    for entries in (certificate.prefix_coefficients, certificate.forward_differences):
        if type(entries) is not tuple or any(type(value) is not Fraction for value in entries):
            raise TypeError("certificate coefficients must be tuples of Fraction values")
    if certificate.tail_start < _minimum_tail_start(Q):
        return CheckResult(
            status="inconclusive", scope=_SCOPE,
            reason="The proposed tail still contains coefficients from Q.",
        )
    if (
        len(certificate.prefix_coefficients) != certificate.tail_start
        or len(certificate.forward_differences) != _tail_size(P)
        or certificate != _certificate(P, Q, certificate.tail_start)
    ):
        return CheckResult(
            status="inconclusive", scope=_SCOPE,
            reason="The certificate does not match the exact Taylor prefix and forward differences.",
        )
    return _check_signs(certificate)


def _check_signs(certificate: ExpSeriesCertificate) -> CheckResult:
    for index, coefficient in enumerate(certificate.prefix_coefficients):
        if coefficient < 0:
            return _negative_coefficient(index)
    if any(value < 0 for value in certificate.forward_differences):
        return CheckResult(
            status="inconclusive", scope=_SCOPE,
            reason=f"A forward difference at index {certificate.tail_start} is negative.",
        )
    return CheckResult(
        status="verified",
        scope=_SCOPE,
        reason=(
            f"Exact Taylor coefficients below index {certificate.tail_start} and Newton's "
            "binomial expansion from that index give a nonnegative, absolutely convergent series."
        ),
        witness=certificate,
    )


def _check_auto(P: Poly, Q: Poly, tail_start: int) -> CheckResult:
    if P.is_zero:
        return _check_signs(_certificate(P, Q, tail_start))

    terms = _falling_terms(P)
    if P.LC() < 0:
        # H has the same leading coefficient as P, so it is eventually negative.
        while _tail_value(terms, tail_start) >= 0:
            tail_start = max(1, 2 * tail_start)
        return _negative_coefficient(tail_start)

    # Every forward difference up to degree P has positive leading
    # coefficient, so at a sufficiently large index all are nonnegative.
    size = _tail_size(P)
    while any(value < 0 for value in _forward_differences(terms, tail_start, size)):
        tail_start = max(1, 2 * tail_start)
    return _check_signs(_certificate(P, Q, tail_start))


def check_exp_series(
    P: Poly, Q: Poly, *, tail_start: int | Literal["auto"] | None = None
) -> CheckResult:
    """Check nonnegativity of P(x) exp(x) + Q(x) on the real half-line x >= 0.

    Supply two univariate SymPy Poly objects over ZZ or QQ with a common
    Symbol as generator. The tail must start strictly after the degree of Q
    (at zero or later when Q is zero). The default is the earliest such
    index. With tail_start="auto", the procedure finds a cutoff that
    certifies all Taylor coefficients or identifies a negative coefficient.
    Unsupported types, domains and invalid indices raise TypeError or ValueError.

    The method is sufficient. A negative coefficient of positive degree or
    a failed tail test leaves the claim inconclusive. A negative constant
    coefficient disproves the claim at x=0.
    """
    P, Q = _polynomials(P, Q)
    minimum = _minimum_tail_start(Q)
    automatic = isinstance(tail_start, str) and tail_start == "auto"
    if not automatic:
        if tail_start is None:
            tail_start = minimum
        _validate_tail_start(tail_start)
        if tail_start < minimum:
            raise ValueError("tail_start must exceed the degree of nonzero Q")

    if P.nth(0) + Q.nth(0) < 0:
        return CheckResult(
            status="disproved", scope=_SCOPE,
            reason="The function value at x=0 is negative.",
            witness=0,
        )
    if automatic:
        return _check_auto(P, Q, minimum)
    return _check_signs(_certificate(P, Q, tail_start))
