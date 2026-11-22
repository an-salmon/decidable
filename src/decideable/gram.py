"""Check exact certificates for linear implications between Gram forms."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import combinations_with_replacement


Rational = int | Fraction
GramIndex = tuple[int, int]
GramForm = Mapping[GramIndex, Rational]


def _exact_rational(value: Rational) -> Fraction:
    if type(value) not in (int, Fraction):
        raise TypeError("Gram coefficients and multipliers must be int or Fraction")
    return Fraction(value)


def _add_gram_form(
    result: defaultdict[GramIndex, Fraction],
    vector_count: int,
    form: GramForm,
    multiplier: Fraction,
) -> None:
    if not isinstance(form, Mapping):
        raise TypeError("a Gram form must be a mapping")
    for index, coefficient in form.items():
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("a Gram index must be a pair of integers")
        i, j = index
        if type(i) is not int or type(j) is not int:
            raise TypeError("a Gram index must be a pair of integers")
        if not 0 <= i <= j < vector_count:
            raise ValueError("a Gram index must satisfy 0 <= i <= j < vector_count")
        result[i, j] += multiplier * _exact_rational(coefficient)


def verify_gram_certificate(
    vector_count: int,
    target: GramForm,
    *,
    equations: Iterable[tuple[Rational, GramForm]] = (),
    inequalities: Iterable[tuple[Rational, GramForm]] = (),
    squares: Iterable[tuple[Rational, Sequence[Rational]]] = (),
) -> bool:
    """Verify that the target Gram form is nonnegative under the hypotheses.

    An entry ``(i, j)`` gives the coefficient of ``<v_i, v_j>`` directly.
    Equation forms are assumed zero and may have signed multipliers.
    Inequality forms are assumed nonnegative and need nonnegative multipliers.
    Each square is a nonnegative weight times the squared norm of the
    vector combination given by its coefficients, in vector index order.

    Return whether the weighted terms sum exactly to the target and satisfy
    the sign conditions. Malformed inputs raise TypeError or ValueError.
    """
    if type(vector_count) is not int:
        raise TypeError("vector_count must be an integer")
    if vector_count < 0:
        raise ValueError("vector_count must be nonnegative")

    remainder: defaultdict[GramIndex, Fraction] = defaultdict(Fraction)
    _add_gram_form(remainder, vector_count, target, Fraction(1))

    for multiplier, form in equations:
        _add_gram_form(remainder, vector_count, form, -_exact_rational(multiplier))

    valid_signs = True
    for multiplier, form in inequalities:
        multiplier = _exact_rational(multiplier)
        valid_signs = valid_signs and multiplier >= 0
        _add_gram_form(remainder, vector_count, form, -multiplier)

    for weight, coefficients in squares:
        weight = _exact_rational(weight)
        valid_signs = valid_signs and weight >= 0
        if not isinstance(coefficients, Sequence) or isinstance(
            coefficients, (str, bytes, bytearray)
        ):
            raise TypeError("square coefficients must be a sequence of rationals")
        if len(coefficients) != vector_count:
            raise ValueError("a square needs one coefficient per vector")
        exact_coefficients = tuple(_exact_rational(value) for value in coefficients)
        terms = [(i, value) for i, value in enumerate(exact_coefficients) if value]
        for (i, left), (j, right) in combinations_with_replacement(terms, 2):
            multiplicity = 1 if i == j else 2
            remainder[i, j] -= weight * multiplicity * left * right

    return valid_signs and not any(remainder.values())
