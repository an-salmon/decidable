"""Check exact certificates for linear implications between Gram forms."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Union

from sympy.polys.fields import FracElement

from .parameters import PositiveParameters


GramCoefficient = Union[int, Fraction, FracElement]
GramIndex = tuple[int, int]
GramForm = Mapping[GramIndex, GramCoefficient]
_ExactCoefficient = Union[Fraction, FracElement]


def _exact_rational(value: GramCoefficient) -> Fraction:
    if type(value) not in (int, Fraction):
        raise TypeError("Gram coefficients and multipliers must be int or Fraction")
    return Fraction(value)


def _nonnegative(
    value: _ExactCoefficient, parameters: PositiveParameters | None
) -> bool:
    if parameters is None:
        return value >= 0
    return parameters.nonnegative(value)


def _add_gram_form(
    result: defaultdict[GramIndex, _ExactCoefficient],
    vector_count: int,
    form: GramForm,
    multiplier: _ExactCoefficient,
    exact_coefficient: Callable[[GramCoefficient], _ExactCoefficient],
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
        result[i, j] += multiplier * exact_coefficient(coefficient)


def verify_gram_certificate(
    vector_count: int,
    target: GramForm,
    *,
    equations: Iterable[tuple[GramCoefficient, GramForm]] = (),
    inequalities: Iterable[tuple[GramCoefficient, GramForm]] = (),
    squares: Iterable[tuple[GramCoefficient, Sequence[GramCoefficient]]] = (),
    parameters: PositiveParameters | None = None,
) -> bool:
    """Verify that the target Gram form is nonnegative under the hypotheses.

    An entry ``(i, j)`` gives the coefficient of ``<v_i, v_j>`` directly.
    Equation forms are assumed zero and may have signed multipliers.
    Inequality forms are assumed nonnegative and need nonnegative multipliers.
    Each square is a nonnegative weight times the squared norm of the
    vector combination given by its coefficients, in vector index order.

    With ``parameters``, rational functions in the declared positive
    parameters are allowed. The coefficient identity is checked in their
    rational function field, and the parameter domain must certify any
    required sign. Without parameters all coefficients are int or Fraction.

    Return whether the weighted terms sum exactly to the target and satisfy
    the sign conditions. Malformed inputs raise TypeError or ValueError.
    """
    if type(vector_count) is not int:
        raise TypeError("vector_count must be an integer")
    if vector_count < 0:
        raise ValueError("vector_count must be nonnegative")
    if parameters is not None and type(parameters) is not PositiveParameters:
        raise TypeError("parameters must be a PositiveParameters object")

    exact_coefficient = _exact_rational if parameters is None else parameters.coefficient
    remainder: defaultdict[GramIndex, _ExactCoefficient] = defaultdict(
        lambda: exact_coefficient(0)
    )
    _add_gram_form(remainder, vector_count, target, exact_coefficient(1), exact_coefficient)

    for multiplier, form in equations:
        _add_gram_form(
            remainder, vector_count, form, -exact_coefficient(multiplier), exact_coefficient
        )

    valid_signs = True
    for multiplier, form in inequalities:
        multiplier = exact_coefficient(multiplier)
        valid_signs = valid_signs and _nonnegative(multiplier, parameters)
        _add_gram_form(remainder, vector_count, form, -multiplier, exact_coefficient)

    for weight, coefficients in squares:
        weight = exact_coefficient(weight)
        valid_signs = valid_signs and _nonnegative(weight, parameters)
        if not isinstance(coefficients, Sequence) or isinstance(
            coefficients, (str, bytes, bytearray)
        ):
            raise TypeError("square coefficients must be a sequence")
        if len(coefficients) != vector_count:
            raise ValueError("a square needs one coefficient per vector")
        exact_coefficients = tuple(exact_coefficient(value) for value in coefficients)
        terms = [(i, value) for i, value in enumerate(exact_coefficients) if value]
        for (i, left), (j, right) in combinations_with_replacement(terms, 2):
            multiplicity = 1 if i == j else 2
            remainder[i, j] -= weight * multiplicity * left * right

    return valid_signs and not any(remainder.values())
