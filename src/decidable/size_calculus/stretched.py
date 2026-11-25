"""Exact replay for finite-law stretched-exponential averaging and tail fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..report import CheckResult
from .enclosures import exp_enclosure, rational_power_enclosure
from .moments import Exact, Majorant, exact


@dataclass(frozen=True)
class AverageProblem:
    """Check the proposed common-s size bound on a weighted average."""

    exponent: Exact
    input_scales: tuple[Exact, ...]
    weights: tuple[Exact, ...]
    output_scale: Exact

    def __post_init__(self):
        exponent = exact(self.exponent, "tail exponent")
        output = exact(self.output_scale, "output scale")
        scales = tuple(exact(scale, "input scale") for scale in self.input_scales)
        weights = tuple(exact(weight, "average weight") for weight in self.weights)
        if exponent <= 0 or output <= 0 or not scales or any(scale <= 0 for scale in scales):
            raise ValueError("the exponent and scales must be positive")
        if len(scales) != len(weights) or min(weights) < 0 or sum(weights) != 1:
            raise ValueError("supply nonnegative average weights summing to one")
        for name, value in (("exponent", exponent), ("output_scale", output),
                            ("input_scales", scales), ("weights", weights)):
            object.__setattr__(self, name, value)


def verify_average_refutation(
    problem: AverageProblem, points, masses, *, root_steps: int = 16, degree: int = 12
) -> CheckResult:
    """Replay the input budgets and strict output excess using rational bounds."""
    scope = "All joint laws of the nonnegative inputs to the weighted average"
    points = tuple(tuple(exact(value, "atom coordinate") for value in point) for point in points)
    masses = tuple(exact(mass, "atom mass") for mass in masses)
    count = len(problem.input_scales)
    if (not points or len(points) != len(masses) or min(masses) <= 0 or sum(masses) != 1
            or any(len(point) != count or min(point) < 0 for point in points)):
        raise ValueError("supply nonnegative atoms with positive masses summing to one")
    if type(root_steps) is not int or root_steps < 0 or type(degree) is not int or degree < 0:
        raise ValueError("enclosure budgets must be nonnegative integers")

    def enclose(value, scale):
        lower, upper = rational_power_enclosure(value / scale, problem.exponent, root_steps)
        return exp_enclosure(lower, degree)[0], exp_enclosure(upper, degree)[1]

    try:
        for index, scale in enumerate(problem.input_scales):
            upper = sum(mass * enclose(point[index], scale)[1]
                        for point, mass in zip(points, masses))
            if upper > 2:
                return CheckResult("inconclusive", scope, "An input budget was not established.")
        lower = sum(
            mass * enclose(sum(a * w for a, w in zip(point, problem.weights)), problem.output_scale)[0]
            for point, mass in zip(points, masses)
        )
    except ValueError as error:
        return CheckResult("inconclusive", scope, str(error))
    if lower <= 2:
        return CheckResult("inconclusive", scope, "Strict output excess was not established.")
    return CheckResult(
        "disproved", scope, f"The output expectation is at least {lower} > 2.", (points, masses)
    )


@dataclass(frozen=True)
class TailOptimum:
    probability: Fraction
    majorant: Majorant
    atoms: tuple[tuple[Fraction, Fraction], ...]


def sharp_closed_tail(threshold: Exact) -> TailOptimum:
    """Sharp P(Z >= threshold) for Z >= 1 and E Z <= 2 in the raw moment chart."""
    threshold = exact(threshold, "threshold")
    if threshold <= 1:
        raise ValueError("the threshold must exceed one")
    if threshold <= 2:
        return TailOptimum(Fraction(1), Majorant(1, (0,)), ((threshold, Fraction(1)),))
    # (Z-1)/(threshold-1) dominates the indicator on both sides of the threshold.
    probability = 1 / (threshold - 1)
    return TailOptimum(
        probability, Majorant(-probability, (probability,)),
        ((Fraction(1), 1 - probability), (threshold, probability)),
    )
