"""Reductions for exact identities in mathematical proofs."""

from fractions import Fraction

from sympy import Add, Dummy, Integer, Mul, Poly, Pow, QQ, Rational, S, Symbol, factorint

from .rational import rational_identity


def loglinear_zero(expression, *, positive_bases=(), exponent_parameters=()) -> bool:
    """Check whether a positive monomial is identically one.

    Pass a tree of products and powers of positive exact rationals and
    declared positive symbols. Construct products and powers with
    evaluate=False when verifying their original structure. Exponents may
    be rational functions in the declared exponent parameters. The identity
    applies where every original exponent denominator is nonzero. Raise
    TypeError or ValueError for inputs outside this grammar.
    """
    bases = tuple(positive_bases)
    parameters = tuple(exponent_parameters)
    for group in (bases, parameters):
        if any(type(symbol) is not Symbol or symbol.is_commutative is not True
               for symbol in group):
            raise TypeError("declare commuting symbols")
        if len({symbol.name for symbol in group}) != len(group):
            raise ValueError("declared symbols must have distinct names")
    if any(symbol.is_positive is False for symbol in bases):
        raise ValueError("base symbols must allow positive values")

    generators = parameters or (Dummy(),)
    zero = Poly(0, *generators, domain=QQ)
    one = Poly(1, *generators, domain=QQ)
    coefficients = {}

    def check_exponent(term):
        if isinstance(term, Rational) or term in parameters:
            return
        if type(term) in (Add, Mul):
            for argument in term.args:
                check_exponent(argument)
            return
        if type(term) is Pow and isinstance(term.exp, Integer):
            check_exponent(term.base)
            if term.exp < 0:
                numerator, _ = term.base.as_numer_denom()
                if Poly(numerator, *generators, domain=QQ).is_zero:
                    raise ValueError("an exponent denominator is identically zero")
            return
        raise ValueError("exponents must be rational functions of declared parameters")

    def collect(term, exponent):
        if term in bases:
            coefficients[term] = coefficients.get(term, S.Zero) + exponent
        elif isinstance(term, Rational) and term > 0:
            for value, sign in ((term.p, 1), (term.q, -1)):
                for prime, multiplicity in factorint(value).items():
                    coefficients[prime] = (
                        coefficients.get(prime, S.Zero) + sign * multiplicity * exponent
                    )
        elif type(term) is Mul:
            for factor in term.args:
                collect(factor, exponent)
        elif type(term) is Pow:
            check_exponent(term.exp)
            collect(term.base, exponent * term.exp)
        else:
            raise ValueError("monomial bases must be products of positive exact rationals and declared symbols")

    if type(expression) in (int, Fraction):
        expression = Rational(expression.numerator, expression.denominator)
    collect(expression, S.One)
    for coefficient in coefficients.values():
        numerator, denominator = (
            Poly(part, *generators, domain=QQ) for part in coefficient.as_numer_denom()
        )
        if not rational_identity(numerator, denominator, zero, one):
            return False
    return True
