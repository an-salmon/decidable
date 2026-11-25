"""Rational enclosures used when replaying elementary moment values."""

from fractions import Fraction

from fractions import Fraction

from .moments import exact


def exp_enclosure(argument, degree: int) -> tuple[Fraction, Fraction]:
    """Enclose exp(argument) by a Taylor prefix and a geometric remainder."""
    argument = exact(argument, "argument")
    if type(degree) is not int or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    magnitude = abs(argument)
    if magnitude >= degree + 2:
        raise ValueError("degree is too small for the geometric remainder")
    term = lower = Fraction(1)
    for index in range(1, degree + 1):
        term *= magnitude / index
        lower += term
    omitted = term * magnitude / (degree + 1)
    upper = lower + omitted / (1 - magnitude / (degree + 2))
    return (1 / upper, 1 / lower) if argument < 0 else (lower, upper)


def rational_power_enclosure(base, exponent, steps: int) -> tuple[Fraction, Fraction]:
    """Enclose base**exponent for a nonnegative rational base and positive exponent."""
    base = exact(base, "base")
    exponent = exact(exponent, "exponent")
    if base < 0 or exponent <= 0 or type(steps) is not int or steps < 0:
        raise ValueError("use a nonnegative base, positive exponent and nonnegative step count")
    target = base**exponent.numerator
    denominator = exponent.denominator
    if target in (0, 1) or denominator == 1:
        return target, target
    lower, upper = Fraction(0), max(Fraction(1), target)
    for _ in range(steps):
        middle = (lower + upper) / 2
        powered = middle**denominator
        if powered == target:
            return middle, middle
        if powered < target:
            lower = middle
        else:
            upper = middle
    return lower, upper
