"""Exact checks for finite mathematical proof obligations."""

from .fragments import FRAGMENTS, dispatch
from .gram import verify_gram_certificate
from .linalg import matrix_rank
from .linear import linear_feasible, linear_implies
from .noncommutative import bilinear_identity, noncommutative_identity
from .parameters import PositiveParameters
from .polynomial import verify_polynomial_certificate
from .quadrature import quadrature_value, verify_quadrature
from .rational import rational_identity
from .report import CheckResult
from .series import rebase_series, shift_series, verify_series_certificate

__all__ = [
    "CheckResult",
    "FRAGMENTS",
    "PositiveParameters",
    "bilinear_identity",
    "dispatch",
    "linear_feasible",
    "linear_implies",
    "matrix_rank",
    "noncommutative_identity",
    "quadrature_value",
    "rational_identity",
    "rebase_series",
    "shift_series",
    "verify_gram_certificate",
    "verify_polynomial_certificate",
    "verify_quadrature",
    "verify_series_certificate",
]
