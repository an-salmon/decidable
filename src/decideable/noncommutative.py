"""Check exact identities of noncommutative polynomials under word relations."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction


Word = tuple[str, ...]
Rule = tuple[Word, Word]
Polynomial = Mapping[Word, int | Fraction]


def _validate_word(word: Word) -> None:
    if not isinstance(word, tuple) or any(
        not isinstance(symbol, str) or not symbol for symbol in word
    ):
        raise TypeError("a word must be a tuple of nonempty strings")


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
    checked_rules = []
    for source, replacement in rules:
        _validate_word(source)
        _validate_word(replacement)
        if len(replacement) >= len(source):
            raise ValueError("a rewrite rule must strictly shorten a word")
        checked_rules.append((source, replacement))
    rewrites = tuple(checked_rules)

    difference: defaultdict[Word, Fraction] = defaultdict(Fraction)
    for sign, polynomial in ((1, left), (-1, right)):
        if not isinstance(polynomial, Mapping):
            raise TypeError("a polynomial must map words to rational coefficients")
        for word, coefficient in polynomial.items():
            _validate_word(word)
            if type(coefficient) not in (int, Fraction):
                raise TypeError("polynomial coefficients must be int or Fraction")
            if coefficient:
                difference[_normal_form(word, rewrites)] += sign * coefficient

    if not any(difference.values()):
        return True
    return False if _is_confluent(rewrites) else None
