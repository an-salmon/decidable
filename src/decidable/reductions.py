"""Exact Gram expansion and conditional finite-support spectral reduction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement

from sympy import Poly, Rational

from .gram import GramIndex, _exact_rational
from .rational import _validate_polynomial


def expand_gram_inner_product(
    left: Sequence[int | Fraction], right: Sequence[int | Fraction]
) -> dict[GramIndex, Fraction]:
    """Expand the inner product of two real vector combinations.

    Each input gives one rational coefficient per vector in the same
    ordered list. The output uses symmetric Gram entries and gives the
    coefficient of each inner product directly, omitting zero terms.
    """
    left = tuple(_exact_rational(value) for value in left)
    right = tuple(_exact_rational(value) for value in right)
    if len(left) != len(right):
        raise ValueError("the combinations must use the same number of vectors")

    result = {}
    for i, j in combinations_with_replacement(range(len(left)), 2):
        coefficient = left[i] * right[j]
        if i != j:
            coefficient += left[j] * right[i]
        if coefficient:
            result[i, j] = coefficient
    return result


@dataclass(frozen=True)
class SpectralReduction:
    """Exact pairs (eigenvalue, polynomial value) on the declared support.

    For a self-adjoint operator and a vector whose spectral measure is
    supported on these points, the polynomial quadratic form is the
    sum of the sampled values times the nonnegative masses at each point.
    """

    samples: tuple[tuple[Fraction, Fraction], ...]

    @property
    def nonnegative_on_support(self) -> bool:
        """Whether every listed value is nonnegative, sufficient for the form."""
        return all(value >= 0 for _, value in self.samples)


def reduce_spectral_quadratic_form(
    polynomial: Poly, support: Iterable[int | Fraction]
) -> SpectralReduction:
    """Evaluate a univariate rational polynomial over a declared finite support.

    The caller supplies self-adjointness and proves the vector's spectral
    measure is supported on this set. Repeated points are evaluated once.
    An empty support implies zero spectral mass. These values cannot
    certify a whole interval or establish the support of a given vector.
    """
    polynomial = _validate_polynomial(polynomial)
    if not polynomial.is_univariate:
        raise ValueError("the spectral polynomial must be univariate")
    points = dict.fromkeys(_exact_rational(value) for value in support)
    samples = []
    for point in points:
        value = polynomial.eval(Rational(point.numerator, point.denominator))
        samples.append((point, Fraction(int(value.p), int(value.q))))
    return SpectralReduction(tuple(samples))
