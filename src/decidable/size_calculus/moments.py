"""Exact majorants and finite-law witnesses for univariate moment problems."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Union

from sympy import Poly, Rational, Symbol

from ..linalg import matrix_solve
from ..rational import _validate_polynomial
from ..report import CheckResult


Exact = Union[int, Fraction]


class _InvalidMomentInput(ValueError):
    """A declared moment domain or supplied witness is malformed."""


def exact(value, name="value") -> Fraction:
    if type(value) not in (int, Fraction):
        raise TypeError(f"{name} must be an int or Fraction")
    return Fraction(value)


def _nonnegative(poly: Poly, lower: Fraction, upper: Fraction | None) -> bool:
    if poly.is_zero:
        return True
    if lower == upper:
        return poly.eval(lower) >= 0

    # Find the first nonzero Taylor coefficient at the left endpoint.
    local = poly.shift(Rational(lower.numerator, lower.denominator))
    if local.terms()[-1][1] < 0:
        return False

    # Starting positive, a polynomial becomes negative on the interval
    # exactly when it crosses an interior root of odd multiplicity.
    return not any(
        multiplicity % 2 and root > lower and (upper is None or root < upper)
        for root, multiplicity in poly.real_roots(multiple=False)
    )


@dataclass(frozen=True)
class MomentProblem:
    """Nonnegative QQ polynomials on a closed interval or right half-line.

    Budgets bound the input expectations and the claimed output expectation.
    Points and budgets are exact rational numbers. Upper None means infinity.
    """

    inputs: tuple[Poly, ...]
    budgets: tuple[Exact, ...]
    output: Poly
    output_budget: Exact
    lower: Exact = 0
    upper: Exact | None = None

    def __post_init__(self):
        inputs = tuple(self.inputs)
        budgets = tuple(exact(value, "input budget") for value in self.budgets)
        output = _validate_polynomial(self.output)
        if len(output.gens) != 1 or type(output.gen) is not Symbol:
            raise _InvalidMomentInput("use one common polynomial coordinate")
        if len(inputs) != len(budgets):
            raise _InvalidMomentInput("supply one budget per input")
        output_budget = exact(self.output_budget, "output budget")
        lower = exact(self.lower, "lower endpoint")
        upper = None if self.upper is None else exact(self.upper, "upper endpoint")
        if upper is not None and upper < lower:
            raise _InvalidMomentInput("support endpoints must be ordered")
        if output_budget < 0 or any(budget < 0 for budget in budgets):
            raise _InvalidMomentInput("budgets must be nonnegative")
        for poly in (*inputs, output):
            _validate_polynomial(poly, output.gens)
            if not _nonnegative(poly, lower, upper):
                raise _InvalidMomentInput("moment integrands must be nonnegative on the support")
        for name, value in (
            ("inputs", tuple(poly.to_field() for poly in inputs)),
            ("output", output.to_field()), ("budgets", budgets), ("output_budget", output_budget),
            ("lower", lower), ("upper", upper),
        ):
            object.__setattr__(self, name, value)

    @property
    def scope(self) -> str:
        upper = "+infinity)" if self.upper is None else f"{self.upper}]"
        return f"All probability laws on [{self.lower}, {upper}"


@dataclass(frozen=True)
class Majorant:
    intercept: Exact
    slopes: tuple[Exact, ...]


@dataclass(frozen=True)
class SquareProof:
    polynomial: Poly
    coefficient: Exact


@dataclass(frozen=True)
class AtomicLaw:
    points: tuple[Exact, ...]
    masses: tuple[Exact, ...]


@dataclass(frozen=True)
class GridOptimum:
    points: tuple[Fraction, ...]
    law: AtomicLaw
    value: Fraction


def verify_majorant(
    problem: MomentProblem, certificate: Majorant, *, square: SquareProof | None = None
) -> CheckResult:
    """Check the cost and the pointwise inequality on the original support.

    A supplied square is replayed by polynomial equality. Otherwise use
    exact real-root isolation on the interval.
    """
    intercept = exact(certificate.intercept, "intercept")
    slopes = tuple(exact(slope, "slope") for slope in certificate.slopes)
    if len(slopes) != len(problem.inputs):
        raise _InvalidMomentInput("supply one slope per input")
    if any(slope < 0 for slope in slopes):
        return CheckResult("inconclusive", problem.scope, "Majorant slopes must be nonnegative.")
    cost = intercept + sum(slope * budget for slope, budget in zip(slopes, problem.budgets))
    if cost > problem.output_budget:
        return CheckResult("inconclusive", problem.scope, "The majorant exceeds the output budget.")
    residual = Poly(intercept, *problem.output.gens, domain="QQ") - problem.output
    for slope, integrand in zip(slopes, problem.inputs):
        residual += integrand.mul_ground(slope)
    if square is not None:
        polynomial = _validate_polynomial(square.polynomial, problem.output.gens).to_field()
        coefficient = exact(square.coefficient, "square coefficient")
        if coefficient < 0 or residual != (polynomial**2).mul_ground(coefficient):
            return CheckResult("inconclusive", problem.scope, "The square does not certify the residual.")
    elif not _nonnegative(residual, problem.lower, problem.upper):
        return CheckResult("inconclusive", problem.scope, "The proposed majorant fails on the support.")
    return CheckResult("verified", problem.scope, f"Global majorant of cost {cost}.", certificate)


def _points(problem: MomentProblem, points: Iterable[Exact]) -> tuple[Fraction, ...]:
    points = tuple(exact(point, "support point") for point in points)
    if not points or any(point < problem.lower or
                         (problem.upper is not None and point > problem.upper)
                         for point in points):
        raise _InvalidMomentInput("supply a nonempty set of points on the original support")
    return points


def _values(problem: MomentProblem, points: tuple[Fraction, ...]):
    return tuple(tuple(Fraction(poly.eval(point)) for point in points)
                 for poly in (*problem.inputs, problem.output))


def verify_atomic_law(
    problem: MomentProblem, law: AtomicLaw, *, strict_feasibility: bool = False
) -> CheckResult:
    """Verify strict input feasibility, or refute the universal moment bound."""
    if type(strict_feasibility) is not bool:
        raise TypeError("strict_feasibility must be a bool")
    points = _points(problem, law.points)
    masses = tuple(exact(mass, "atom mass") for mass in law.masses)
    if len(points) != len(masses) or any(mass <= 0 for mass in masses) or sum(masses) != 1:
        return CheckResult("inconclusive", problem.scope, "The masses must be positive and sum to one.")
    means = tuple(sum(mass * value for mass, value in zip(masses, row))
                  for row in _values(problem, points))
    for mean, budget in zip(means, problem.budgets):
        if mean > budget or (strict_feasibility and mean == budget):
            return CheckResult("inconclusive", problem.scope, "The law fails an input budget.")
    if strict_feasibility:
        return CheckResult("verified", problem.scope, "The law is strictly feasible.", law)
    if means[-1] <= problem.output_budget:
        return CheckResult("inconclusive", problem.scope, "The output does not strictly exceed its budget.")
    return CheckResult("disproved", problem.scope, f"The law has output expectation {means[-1]}.", law)


def grid_optimum(problem: MomentProblem, grid: Iterable[Exact]) -> GridOptimum | None:
    """Enumerate the exact vertices on a supplied finite rational grid.

    This is a small-grid reference search. Each candidate solves normalization
    and a subset of the tight moment constraints on at most m+1 points.
    The resulting optimum has finite scope. None means the grid is infeasible.
    """
    points = tuple(dict.fromkeys(_points(problem, grid)))
    *inputs, output = _values(problem, points)
    optimum = _grid_optimum_rows(tuple(inputs), problem.budgets, output)
    if optimum is None:
        return None
    support, masses, value = optimum
    return GridOptimum(points, AtomicLaw(tuple(points[i] for i in support), masses), value)


def _grid_optimum_rows(
    inputs: tuple[tuple[Fraction, ...], ...],
    budgets: tuple[Fraction, ...],
    output: tuple[Fraction, ...],
) -> tuple[tuple[int, ...], tuple[Fraction, ...], Fraction] | None:
    """Maximize a rational grid objective, returning indices, masses and value."""
    best = None
    for size in range(1, min(len(output), len(inputs) + 1) + 1):
        for support in combinations(range(len(output)), size):
            rows = [tuple(row[i] for i in support) for row in inputs]
            for active in combinations(range(len(inputs)), size - 1):
                solution = matrix_solve(
                    [(1,) * size, *(rows[i] for i in active)],
                    [1, *(budgets[i] for i in active)],
                )
                if solution is None or solution.nullspace or min(solution.particular) <= 0:
                    continue
                masses = solution.particular
                if any(sum(p * g for p, g in zip(masses, row)) > budget
                       for row, budget in zip(rows, budgets)):
                    continue
                value = sum(p * output[i] for p, i in zip(masses, support))
                if best is None or value > best[2]:
                    best = support, masses, value
    return best


def search_refutation(problem: MomentProblem, grid: Iterable[Exact]) -> CheckResult:
    """Search a grid and replay any counterexample on the original support."""
    optimum = grid_optimum(problem, grid)
    if optimum is not None:
        result = verify_atomic_law(problem, optimum.law)
        if result.status == "disproved":
            return result
        return CheckResult(
            "inconclusive", f"Grid {optimum.points}", f"Grid optimum is {optimum.value}.", optimum
        )
    return CheckResult("inconclusive", problem.scope, "The specified grid has no feasible law.")


def search_majorant(
    problem: MomentProblem, grid: Iterable[Exact], *, max_systems: int = 1_000
) -> CheckResult:
    """Try rational contact and tangency candidates, then replay globally."""
    if type(max_systems) is not int or max_systems <= 0:
        raise ValueError("max_systems must be a positive integer")
    points = tuple(dict.fromkeys(_points(problem, grid)))
    *inputs, output = _values(problem, points)
    differentials = tuple(
        tuple(Fraction(poly.diff().eval(point)) for point in points)
        for poly in (*problem.inputs, problem.output)
    )
    count = 0
    for size in range(len(inputs) + 1):
        for active in combinations(range(len(inputs)), size):
            contacts = [
                ((1, *(inputs[i][j] for i in active)), output[j])
                for j in range(len(points))
            ]
            tangencies = [
                ((0, *(differentials[i][j] for i in active)), differentials[-1][j])
                for j in range(len(points))
            ]
            for equations in combinations(contacts + tangencies, size + 1):
                count += 1
                if count > max_systems:
                    return CheckResult("inconclusive", problem.scope, "Majorant search budget exhausted.")
                rows, rhs = zip(*equations)
                solution = matrix_solve(rows, rhs)
                if solution is None or solution.nullspace:
                    continue
                intercept, *values = solution.particular
                if any(slope < 0 for slope in values):
                    continue
                if any(intercept + sum(slope * inputs[i][j] for i, slope in zip(active, values))
                       < target for j, target in enumerate(output)):
                    continue
                support = dict(zip(active, values))
                candidate = Majorant(intercept, tuple(support.get(i, 0) for i in range(len(inputs))))
                result = verify_majorant(problem, candidate)
                if result.status == "verified":
                    return result
    return CheckResult("inconclusive", problem.scope, "No tested candidate was a global majorant.")
