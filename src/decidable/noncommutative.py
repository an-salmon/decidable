"""Check exact identities of noncommutative polynomials under word relations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from fractions import Fraction
from typing import TypeVar, Union


Word = tuple[str, ...]
Rule = tuple[Word, Word]
BilinearTerm = tuple[str, Word, str]
Rational = Union[int, Fraction]
Polynomial = Mapping[Word, Rational]
BilinearExpression = Mapping[BilinearTerm, Rational]
Term = TypeVar("Term")


def _is_name(name: object) -> bool:
    return isinstance(name, str) and bool(name)


def _validate_word(word: Word) -> None:
    if not isinstance(word, tuple) or any(not _is_name(symbol) for symbol in word):
        raise TypeError("a word must be a tuple of nonempty strings")


def _checked_rules(rules: Iterable[Rule]) -> tuple[Rule, ...]:
    checked = []
    for source, replacement in rules:
        _validate_word(source)
        _validate_word(replacement)
        if len(replacement) >= len(source):
            raise ValueError("a rewrite rule must strictly shorten a word")
        checked.append((source, replacement))
    return tuple(checked)


def _difference_terms(
    left: Mapping[Term, Rational], right: Mapping[Term, Rational]
) -> Iterator[tuple[Term, Rational]]:
    for sign, polynomial in ((1, left), (-1, right)):
        if not isinstance(polynomial, Mapping):
            raise TypeError("an expression must map terms to rational coefficients")
        for term, coefficient in polynomial.items():
            if type(coefficient) not in (int, Fraction):
                raise TypeError("polynomial coefficients must be int or Fraction")
            yield term, sign * coefficient


def _first_reduct(word: Word, rules: tuple[Rule, ...]) -> Word | None:
    for offset in range(len(word)):
        for source, replacement in rules:
            end = offset + len(source)
            if word[offset:end] == source:
                return word[:offset] + replacement + word[end:]
    return None


def _normal_form(word: Word, rules: tuple[Rule, ...]) -> Word:
    while (reduct := _first_reduct(word, rules)) is not None:
        word = reduct
    return word


def _is_confluent(rules: tuple[Rule, ...]) -> bool:
    # Disjoint replacements commute. Checking all overlaps, including full
    # containment and overlaps of a rule with itself, gives local confluence.
    for first, first_replacement in rules:
        for second, second_replacement in rules:
            for offset in range(len(first)):
                overlap = min(len(first) - offset, len(second))
                if first[offset : offset + overlap] != second[:overlap]:
                    continue

                end = offset + len(second)
                word = first[:offset] + second + first[end:]
                first_reduct = first_replacement + word[len(first) :]
                second_reduct = word[:offset] + second_replacement + word[end:]
                if _normal_form(first_reduct, rules) != _normal_form(
                    second_reduct, rules
                ):
                    return False
    # Every replacement shortens the word, so local confluence is confluence.
    return True


def _adjoint_word(word: Word, adjoints: Mapping[str, str]) -> Word:
    try:
        return tuple(adjoints[symbol] for symbol in reversed(word))
    except KeyError as error:
        raise ValueError(f"missing adjoint declaration for {error.args[0]!r}") from None


def noncommutative_identity(
    left: Polynomial,
    right: Polynomial,
    *,
    rules: Iterable[Rule] = (),
) -> bool | None:
    """Compare rational noncommutative polynomials under word equalities.

    Each polynomial maps words to exact int or Fraction coefficients. The
    empty tuple represents the unit. Each rule is a pair of words, with the
    replacement strictly shorter than the source. Rules may be supplied in
    any order. The reducer applies the leftmost match, breaking ties by
    rule order.

    Return True if the reduced polynomials coincide. Return False if a
    confluent system gives different normal forms. This is inequality in
    the free associative algebra modulo the given word relations. A claim
    about a particular matrix representation may require more relations.
    Return None if the forms differ and the rewrite system is nonconfluent.
    """
    rewrites = _checked_rules(rules)

    difference: defaultdict[Word, Fraction] = defaultdict(Fraction)
    for word, coefficient in _difference_terms(left, right):
        _validate_word(word)
        if coefficient:
            difference[_normal_form(word, rewrites)] += coefficient

    if not any(difference.values()):
        return True
    return False if _is_confluent(rewrites) else None


def bilinear_identity(
    left: BilinearExpression,
    right: BilinearExpression,
    *,
    adjoints: Mapping[str, str],
    rules: Iterable[Rule] = (),
) -> bool | None:
    """Compare real scalar expressions built from terms ``(p, word, q)``.

    A term represents ``pᵀ word q``. The explicit ``adjoints`` mapping
    pairs each matrix symbol with its transpose, using a fixed point for
    a symmetric matrix. Supply both directions of each pair and include
    all symbols used in terms or rules. Reversing a word and transposing
    its letters, then exchanging the vector names, preserves its scalar.
    The checker also adds transposed versions of the supplied word rules.

    Matching forms establish the identity. If forms differ, return False
    for formal inequivalence when the adjoint-closed rules are confluent,
    or None otherwise. Formal inequivalence makes no claim about a
    specific matrix representation.
    """
    if not isinstance(adjoints, Mapping):
        raise TypeError("adjoints must map matrix symbols to matrix symbols")
    for symbol, adjoint in adjoints.items():
        if not _is_name(symbol) or not _is_name(adjoint):
            raise TypeError("matrix symbols must be nonempty strings")
        if adjoints.get(adjoint) != symbol:
            raise ValueError("adjoints must form an involution")

    checked = _checked_rules(rules)
    transposed_rules = tuple(
        (_adjoint_word(source, adjoints), _adjoint_word(replacement, adjoints))
        for source, replacement in checked
    )
    rewrites = tuple(dict.fromkeys(checked + transposed_rules))

    difference: defaultdict[BilinearTerm, Fraction] = defaultdict(Fraction)
    for term, coefficient in _difference_terms(left, right):
        if not isinstance(term, tuple) or len(term) != 3:
            raise TypeError("a bilinear term must be (vector name, word, vector name)")
        p, word, q = term
        if not _is_name(p) or not _is_name(q):
            raise TypeError("vector names must be nonempty strings")
        _validate_word(word)
        adjoint = _adjoint_word(word, adjoints)
        if coefficient:
            normal = (p, _normal_form(word, rewrites), q)
            transposed = (q, _normal_form(adjoint, rewrites), p)
            difference[min(normal, transposed)] += coefficient

    if not any(difference.values()):
        return True
    return False if _is_confluent(rewrites) else None
