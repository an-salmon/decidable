"""Replay scale rewrites and allocated comparisons of positive monomials."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from sympy import Rational

from ..report import CheckResult
from .moments import Exact, exact


@dataclass(frozen=True)
class ScaleMonomial:
    """A positive rational coefficient, rational powers and optional generic factor.

    A generic factor is declared by its allowed deterministic dependencies.
    None denotes an exact coefficient. An empty set allows a constant
    independent of all varying quantities.
    """

    coefficient: Exact
    powers: Mapping[str, Exact]
    dependencies: frozenset[str] | None = None

    def __post_init__(self):
        coefficient = exact(self.coefficient, "scale coefficient")
        if coefficient <= 0:
            raise ValueError("a monomial coefficient must be positive")
        powers = {}
        for atom, exponent in self.powers.items():
            if type(atom) is not str or not atom:
                raise ValueError("scale atom names must be nonempty strings")
            value = exact(exponent, "scale exponent")
            if value:
                powers[atom] = value
        dependencies = self.dependencies
        if dependencies is not None and (
            type(dependencies) is not frozenset or
            any(type(name) is not str or not name for name in dependencies)
        ):
            raise TypeError("generic dependencies must be a frozenset of names")
        if dependencies is not None and dependencies.intersection(powers):
            raise ValueError("scale atoms cannot be generic constant dependencies")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "powers", MappingProxyType(powers))


@dataclass(frozen=True)
class ScaleShare:
    """Pay for one derived term from one claimed term."""

    derived: int
    claimed: int
    coefficient: Exact | None = None


def rewrite_scale(term: ScaleMonomial, atom: str, replacement: ScaleMonomial) -> ScaleMonomial:
    """Substitute a declared exact monomial equality for one scale atom."""
    if type(atom) is not str or not atom or atom in replacement.powers:
        raise ValueError("the rewrite needs a nonrecursive scale atom")
    if replacement.dependencies is not None:
        raise ValueError("an exact rewrite needs an exact replacement")
    if term.dependencies is not None and (
        atom in term.dependencies or term.dependencies.intersection(replacement.powers)
    ):
        raise ValueError("the rewrite must not change generic constant dependencies")
    powers = dict(term.powers)
    exponent = powers.pop(atom, Fraction(0))
    for name, power in replacement.powers.items():
        powers[name] = powers.get(name, Fraction(0)) + power * exponent
    factor = Rational(replacement.coefficient) ** Rational(exponent)
    if not isinstance(factor, Rational):
        raise ValueError("the rewrite produces a nonrational coefficient")
    return ScaleMonomial(term.coefficient * Fraction(factor), powers, term.dependencies)


def verify_scale_domination(derived, claimed, ranges, shares) -> CheckResult:
    """Replay a comparison of positive sums in independent atom ranges.

    Small, large and free ranges are (0,1], [1,infinity) and (0,infinity).
    The exact claimed terms have a finite budget that their shares may not
    exceed. A claimed generic factor is existential: it can pay for multiple
    assigned terms whose generic dependencies it is allowed to absorb.
    Scale atoms and generic dependency names must be disjoint.
    """
    scope = "Scale domination on the declared independent positive atom ranges"
    derived, claimed = tuple(derived), tuple(claimed)
    if not derived or not claimed:
        raise ValueError("both scale sums must contain a term")
    if any(type(term) is not ScaleMonomial for term in (*derived, *claimed)):
        raise TypeError("scales must contain ScaleMonomial terms")
    if any(type(atom) is not str or not atom for atom in ranges):
        raise ValueError("scale atom names must be nonempty strings")
    if any(kind not in ("small", "large", "free") for kind in ranges.values()):
        raise ValueError("each scale range must be small, large or free")
    for term in (*derived, *claimed):
        if set(term.powers) - ranges.keys():
            raise ValueError("each scale atom must have a declared range")
        if term.dependencies is not None and term.dependencies.intersection(ranges):
            raise ValueError("scale atoms cannot be generic constant dependencies")

    paid = set()
    totals = defaultdict(Fraction)
    for share in shares:
        if type(share) is not ScaleShare:
            raise TypeError("use ScaleShare allocations")
        i, j = share.derived, share.claimed
        if (type(i) is not int or type(j) is not int or not 0 <= i < len(derived)
                or not 0 <= j < len(claimed) or i in paid):
            raise ValueError("allocate each derived term to exactly one claimed term")
        paid.add(i)
        source, target = derived[i], claimed[j]
        for atom, kind in ranges.items():
            gap = source.powers.get(atom, 0) - target.powers.get(atom, 0)
            if (kind == "small" and gap < 0 or kind == "large" and gap > 0
                    or kind == "free" and gap != 0):
                return CheckResult("inconclusive", scope, f"An allocated share has incompatible power of {atom}.")
        if target.dependencies is None:
            if source.dependencies is not None:
                return CheckResult("inconclusive", scope, "An exact scale cannot absorb a generic factor.")
            coefficient = exact(share.coefficient, "allocated coefficient")
            if coefficient < source.coefficient:
                return CheckResult("inconclusive", scope, "An allocated coefficient is too small.")
            totals[j] += coefficient
            if totals[j] > target.coefficient:
                return CheckResult("inconclusive", scope, "Allocations exceed the claimed coefficient.")
        elif share.coefficient is not None or not (
            source.dependencies is None or source.dependencies <= target.dependencies
        ):
            return CheckResult("inconclusive", scope, "The generic allocation exceeds its declared dependencies.")
    if len(paid) != len(derived):
        return CheckResult("inconclusive", scope, "Some derived terms have no allocated share.")
    return CheckResult("verified", scope, "All allocated terms are dominated by their claimed shares.")
