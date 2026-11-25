"""Exact replay for finite-law stretched-exponential averaging and tail fixtures."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from fractions import Fraction

from ..report import CheckResult
from .enclosures import exp_enclosure, rational_power_enclosure
from .moments import Exact, Majorant, _grid_optimum_rows, exact


_AVERAGE_SCOPE = "All joint laws of the nonnegative inputs to the weighted average"


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


def _average_points(
    problem: AverageProblem, points: Iterable[Iterable[Exact]],
) -> tuple[tuple[Fraction, ...], ...]:
    points = tuple(tuple(exact(value, "atom coordinate") for value in point) for point in points)
    count = len(problem.input_scales)
    if not points or any(len(point) != count or min(point) < 0 for point in points):
        raise ValueError("supply nonempty atoms with one nonnegative coordinate per input")
    return points


def _enclosure_budgets(root_steps: int, degree: int) -> None:
    if type(root_steps) is not int or root_steps < 0 or type(degree) is not int or degree < 0:
        raise ValueError("enclosure budgets must be nonnegative integers")


def _moment_bound(
    value: Fraction, scale: Fraction, exponent: Fraction, root_steps: int, degree: int,
    *, upper: bool,
) -> Fraction:
    endpoint = 1 if upper else 0
    power = rational_power_enclosure(value / scale, exponent, root_steps)[endpoint]
    return exp_enclosure(power, degree)[endpoint]


def _weighted_average(problem: AverageProblem, point: tuple[Fraction, ...]) -> Fraction:
    return sum(value * weight for value, weight in zip(point, problem.weights))


def verify_average_refutation(
    problem: AverageProblem, points: Iterable[Iterable[Exact]], masses: Iterable[Exact],
    *, root_steps: int = 16, degree: int = 12,
) -> CheckResult:
    """Replay the input budgets and strict output excess using rational bounds."""
    points = _average_points(problem, points)
    masses = tuple(exact(mass, "atom mass") for mass in masses)
    if len(points) != len(masses) or min(masses) <= 0 or sum(masses) != 1:
        raise ValueError("supply positive masses summing to one")
    _enclosure_budgets(root_steps, degree)
    try:
        for index, scale in enumerate(problem.input_scales):
            upper = sum(
                mass * _moment_bound(
                    point[index], scale, problem.exponent, root_steps, degree, upper=True,
                )
                for point, mass in zip(points, masses)
            )
            if upper > 2:
                return CheckResult("inconclusive", _AVERAGE_SCOPE, "An input budget was not established.")
        lower = sum(
            mass * _moment_bound(
                _weighted_average(problem, point), problem.output_scale,
                problem.exponent, root_steps, degree, upper=False,
            )
            for point, mass in zip(points, masses)
        )
    except ValueError as error:
        return CheckResult("inconclusive", _AVERAGE_SCOPE, str(error))
    if lower <= 2:
        return CheckResult("inconclusive", _AVERAGE_SCOPE, "Strict output excess was not established.")
    return CheckResult(
        "disproved", _AVERAGE_SCOPE, f"The output expectation is at least {lower} > 2.", (points, masses)
    )


def search_average_refutation(
    problem: AverageProblem, grid: Iterable[Iterable[Exact]],
    *, root_steps: int = 16, degree: int = 12,
) -> CheckResult:
    """Search a finite rational grid and replay a violating weighted-average law.

    The grid program uses input upper bounds and output lower bounds. Points
    outside the available enclosure precision are skipped. Search exhaustion or
    insufficient precision leaves the original bound inconclusive.
    """
    points = tuple(dict.fromkeys(_average_points(problem, grid)))
    _enclosure_budgets(root_steps, degree)
    enclosed_points = []
    rows = []
    output = []
    for point in points:
        try:
            inputs = tuple(
                _moment_bound(value, scale, problem.exponent, root_steps, degree, upper=True)
                for value, scale in zip(point, problem.input_scales)
            )
            lower = _moment_bound(
                _weighted_average(problem, point), problem.output_scale,
                problem.exponent, root_steps, degree, upper=False,
            )
        except ValueError:
            continue
        enclosed_points.append(point)
        rows.append(inputs)
        output.append(lower)
    scope = f"Enclosed rational grid ({len(enclosed_points)} of {len(points)} points)"
    input_rows = tuple(zip(*rows))
    optimum = _grid_optimum_rows(input_rows, (Fraction(2),) * len(input_rows), tuple(output))
    if optimum is None:
        return CheckResult("inconclusive", scope, "No law passed the conservative input budgets.")
    support, masses, value = optimum
    if value <= 2:
        return CheckResult("inconclusive", scope, "No optimum had strict certified output excess.")
    return verify_average_refutation(
        problem, tuple(enclosed_points[index] for index in support), masses,
        root_steps=root_steps, degree=degree,
    )


@dataclass(frozen=True)
class AverageScaleBracket:
    """A refuted lower scale and a verified upper scale for the same inputs."""

    lower: Fraction
    upper: Fraction
    lower_receipt: CheckResult
    upper_receipt: CheckResult


def bracket_average_scale(
    problem: AverageProblem, grid: Iterable[Iterable[Exact]], *, upper_scale: Exact | None = None,
    trials: int = 12, root_steps: int = 16, degree: int = 12,
) -> AverageScaleBracket | None:
    """Bracket the sharp scale, starting from the problem's proposed lower scale.

    Return None if the grid cannot refute the initial lower scale or the
    aggregation theorem cannot verify a supplied upper scale. Interior trials
    refine a bound only after exact replay. An unresolved trial leaves the
    bracket in force and makes both subintervals available for later trials.
    """
    if type(trials) is not int or trials < 0:
        raise ValueError("trials must be a nonnegative integer")
    _enclosure_budgets(root_steps, degree)
    points = _average_points(problem, grid)
    lower = problem.output_scale
    lower_receipt = search_average_refutation(problem, points, root_steps=root_steps, degree=degree)
    if lower_receipt.status != "disproved":
        return None

    exponent = problem.exponent
    scale_sum = sum(weight * scale for weight, scale in zip(problem.weights, problem.input_scales))
    # The aggregation theorem has factor 1 for s >= 1 and factor
    # (1 + (1-s)/(s log(2)))**(1/s) otherwise. The tangent to 1/t at t=3/2
    # integrated over [1, 2] proves log(2) >= 2/3.
    factor_power = 1 + max(0, 1 - exponent) / (exponent * Fraction(2, 3))
    threshold = factor_power**exponent.denominator

    def verify_upper(scale: Fraction) -> CheckResult | None:
        if (scale / scale_sum)**exponent.numerator < threshold:
            return None
        return CheckResult(
            "verified", _AVERAGE_SCOPE,
            f"The weighted-sum aggregation theorem verifies scale {scale}.",
        )

    if upper_scale is None:
        _, factor = rational_power_enclosure(factor_power, 1 / exponent, root_steps)
        upper = scale_sum * factor
    else:
        upper = exact(upper_scale, "upper scale")
        if upper <= lower:
            raise ValueError("the upper scale must exceed the lower scale")
    upper_receipt = verify_upper(upper)
    if upper_receipt is None:
        return None

    intervals = deque([(lower, upper)])
    for _ in range(trials):
        left, right = intervals.popleft()
        scale = (left + right) / 2
        candidate_upper = verify_upper(scale)
        if candidate_upper is not None:
            upper, upper_receipt = scale, candidate_upper
        else:
            candidate_lower = search_average_refutation(
                replace(problem, output_scale=scale), points, root_steps=root_steps, degree=degree,
            )
            if candidate_lower.status != "disproved":
                intervals.extend(((left, scale), (scale, right)))
                continue
            lower, lower_receipt = scale, candidate_lower
        intervals.clear()
        intervals.append((lower, upper))

    return AverageScaleBracket(lower, upper, lower_receipt, upper_receipt)


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
