"""Exact checks for finite mathematical proof obligations."""

from .gram import verify_gram_certificate
from .rational import rational_identity

__all__ = ["rational_identity", "verify_gram_certificate"]
