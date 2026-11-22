"""The public mathematical fragments and their typed dispatch routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Literal

from sympy import Poly

from .gram import GramCoefficient, GramForm, verify_gram_certificate
from .linalg import RationalMatrix, matrix_rank
from .linear import AffineForm, linear_implies
from .parameters import PositiveParameters
from .report import CheckResult
from .series import RationalSeries


@dataclass(frozen=True)
class FragmentSpec:
    """A stable fragment ID, its domain, and supported dispatcher operation IDs."""

    id: str
    domain: str
    operations: tuple[str, ...] = ()


FRAGMENTS: Mapping[str, FragmentSpec] = MappingProxyType({
    fragment.id: fragment
    for fragment in (
        FragmentSpec("rational", "Rational functions over exact coefficient domains"),
        FragmentSpec("noncommutative", "Word expressions with supplied rewrite relations"),
        FragmentSpec("gram", "Linear forms in real inner products", ("verify_gram_certificate",)),
        FragmentSpec("polynomial", "Polynomial identities and inequalities over exact coefficients"),
        FragmentSpec("linear", "Finite affine systems over the reals or rationals", (
            "linear_implies", "find_positive_state",
        )),
        FragmentSpec("series", "Formal power series and certified series sign arguments", (
            "check_kernel_domination", "check_exp_series",
        )),
        FragmentSpec("linalg", "Finite rational matrices", ("matrix_rank",)),
    )
})


@dataclass(frozen=True)
class GramCertificateClaim:
    """A Gram form and its proposed nonnegativity certificate."""

    vector_count: int
    target: GramForm
    equations: Sequence[tuple[GramCoefficient, GramForm]] = ()
    inequalities: Sequence[tuple[GramCoefficient, GramForm]] = ()
    squares: Sequence[tuple[GramCoefficient, Sequence[GramCoefficient]]] = ()
    parameters: PositiveParameters | None = None


@dataclass(frozen=True)
class LinearImplicationClaim:
    """An affine target and nonstrict or strict signed affine hypotheses."""

    variable_count: int
    target: AffineForm
    inequalities: Sequence[AffineForm] = ()
    strict_inequalities: Sequence[AffineForm] = ()


@dataclass(frozen=True)
class KernelDominationClaim:
    """An ordered pair of scalar rational series and the tail starting index."""

    upper: RationalSeries
    lower: RationalSeries
    tail_start: int | None = 0


@dataclass(frozen=True)
class ExpSeriesClaim:
    """A nonnegativity claim for P(x) exp(x) + Q(x) on x >= 0."""

    P: Poly
    Q: Poly
    tail_start: int | Literal["auto"] | None = None


@dataclass(frozen=True)
class PositiveStateClaim:
    """Existence of a positive linear functional for a fixed rate."""

    matrix: RationalMatrix
    rate: int | Fraction


@dataclass(frozen=True)
class MatrixRankClaim:
    """An explicit rational matrix whose rank is requested."""

    matrix: RationalMatrix


def check_gram_certificate(claim: GramCertificateClaim) -> CheckResult:
    """Replay a certificate for the stated Gram implication over real vectors.

    Malformed certificate data raises TypeError or ValueError. A valid
    certificate proves the implication for the stated number of vectors and,
    if present, all declared positive parameters.
    """
    verified = verify_gram_certificate(
        claim.vector_count, claim.target, equations=claim.equations,
        inequalities=claim.inequalities, squares=claim.squares,
        parameters=claim.parameters,
    )
    scope = f"All lists of {claim.vector_count} vectors in real inner-product spaces"
    if claim.parameters is not None:
        scope += " and all declared positive parameter values"
    if verified:
        return CheckResult("verified", scope, "The exact Gram certificate replays", claim)
    return CheckResult("inconclusive", scope, "The supplied Gram certificate did not replay")


def check_linear_implication(claim: LinearImplicationClaim) -> CheckResult:
    """Decide the stated affine implication over the reals and rationals."""
    verified = linear_implies(
        claim.variable_count, claim.target, inequalities=claim.inequalities,
        strict_inequalities=claim.strict_inequalities,
    )
    scope = f"All real or rational assignments to {claim.variable_count} affine variables"
    if verified:
        return CheckResult("verified", scope, "Exact elimination proves the implication")
    return CheckResult("disproved", scope, "Exact elimination shows that the strict negation is feasible")


def dispatch(fragment: str, operation: str, claim: object) -> CheckResult:
    """Run a registered operation on its declared claim type.

    Unknown routes and mismatched claim types return unsupported. Each
    operation validates the contents of its claim under its own input contract.
    """
    if (fragment, operation) == ("gram", "verify_gram_certificate"):
        if isinstance(claim, GramCertificateClaim):
            return check_gram_certificate(claim)
    elif (fragment, operation) == ("linear", "linear_implies"):
        if isinstance(claim, LinearImplicationClaim):
            return check_linear_implication(claim)
    elif (fragment, operation) == ("series", "check_kernel_domination"):
        if isinstance(claim, KernelDominationClaim):
            from .kernel_domination import check_kernel_domination

            return check_kernel_domination(claim.upper, claim.lower, tail_start=claim.tail_start)
    elif (fragment, operation) == ("series", "check_exp_series"):
        if isinstance(claim, ExpSeriesClaim):
            from .exp_series import check_exp_series

            return check_exp_series(claim.P, claim.Q, tail_start=claim.tail_start)
    elif (fragment, operation) == ("linear", "find_positive_state"):
        if isinstance(claim, PositiveStateClaim):
            from .positive_state import find_positive_state

            return find_positive_state(claim.matrix, claim.rate)
    elif (fragment, operation) == ("linalg", "matrix_rank"):
        if isinstance(claim, MatrixRankClaim):
            rank = matrix_rank(claim.matrix)
            return CheckResult("verified", "Supplied finite rational matrix", f"Exact rank is {rank}", rank)
    else:
        return CheckResult("unsupported", "Requested operation", f"Unknown route {fragment!r}/{operation!r}")
    return CheckResult(
        "unsupported", "Requested operation",
        f"Claim type {type(claim).__name__!r} does not match {fragment!r}/{operation!r}",
    )
