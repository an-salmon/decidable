"""Compare defined quadrature formulas in an exact formal rational field."""

from __future__ import annotations

from fractions import Fraction

from sympy import Add, Dummy, E, Expr, Mul, Poly, Pow, QQ, Rational, exp, log
from sympy.polys.polyerrors import CoercionFailed

from .parameters import PositiveParameters
from .rational import rational_identity


def same_closed_form(
    left: Expr | int | Fraction,
    right: Expr | int | Fraction,
    parameters: PositiveParameters | None,
) -> bool:
    """Check equality after replacing matching defined special terms with atoms.

    Supported terms are exact rationals, declared parameters, sums, products,
    integer powers, positive rational functions to rational powers, and exp
    or log of rational functions of the parameters. Logs of rational powers
    of declared positive symbols are rewritten using log(p**k) = k*log(p).
    Every explicit inverse is checked on its input base before cancellation.
    The atom comparison can miss equalities among special terms.
    """
    if parameters is not None and type(parameters) is not PositiveParameters:
        raise TypeError("parameters must be PositiveParameters")
    symbols = () if parameters is None else parameters.symbols
    atoms: dict[Expr, Dummy] = {}

    def rational_argument(expression: Expr, *, sign: str | None = None) -> Expr:
        if parameters is None:
            if not isinstance(expression, Rational):
                raise TypeError("expected an exact rational argument")
            canonical = expression
            nonzero, nonnegative = expression != 0, expression >= 0
        else:
            if expression.free_symbols.difference(symbols):
                raise TypeError("expected a rational function of the declared parameters")
            try:
                value = parameters.field.from_sympy(expression)
            except CoercionFailed as error:
                raise TypeError("expected a rational function of the declared parameters") from error
            canonical = value.as_expr()
            if sign is None:
                # Lowering has already checked the original inverses. A
                # reduced denominator may no longer expose that sign proof.
                return canonical
            nonzero = parameters.nonzero(value)
            nonnegative = parameters.nonnegative(value)
        if sign == "positive" and not (nonzero and nonnegative):
            raise ValueError("the argument is not certified strictly positive")
        if sign == "nonzero" and not nonzero:
            raise ValueError("the denominator base is not certified nonzero")
        return canonical

    def atom(expression: Expr) -> Dummy:
        if expression not in atoms:
            atoms[expression] = Dummy()
        return atoms[expression]

    def lower(expression: Expr | int | Fraction) -> Expr:
        if type(expression) in (int, Fraction):
            return Rational(expression.numerator, expression.denominator)
        if not isinstance(expression, Expr):
            raise TypeError("expected an exact closed-form expression")
        if isinstance(expression, Rational) or expression in symbols:
            return expression
        if expression is E:
            return atom(exp(Rational(1), evaluate=False))
        if type(expression) is Add:
            return Add(*(lower(operand) for operand in expression.args))
        if type(expression) is Mul:
            return Mul(*(lower(operand) for operand in expression.args))
        if type(expression) is Pow:
            base, exponent = expression.args
            base = lower(base)
            if not isinstance(exponent, Rational):
                raise TypeError("expected an exact rational exponent")
            if exponent.q != 1:
                base = rational_argument(base, sign="positive")
                return Rational(1) if base == 1 else atom(Pow(base, exponent, evaluate=False))
            if exponent < 0:
                base = rational_argument(base, sign="nonzero")
            return Pow(base, exponent)
        if type(expression) is exp:
            argument = rational_argument(lower(expression.args[0]))
            return Rational(1) if argument == 0 else atom(exp(argument, evaluate=False))
        if type(expression) is log and len(expression.args) == 1:
            (argument,) = expression.args
            if type(argument) is Pow:
                base, exponent = argument.args
                if base in symbols and isinstance(exponent, Rational):
                    return exponent * atom(log(base, evaluate=False))
            argument = rational_argument(lower(argument), sign="positive")
            return Rational(0) if argument == 1 else atom(log(argument, evaluate=False))
        raise TypeError("unsupported closed-form term")

    left, right = lower(left), lower(right)
    generators = (*symbols, *atoms.values()) or (Dummy(),)
    polynomials = [
        Poly(part, *generators, domain=QQ)
        for expression in (left, right)
        for part in expression.as_numer_denom()
    ]
    return rational_identity(*polynomials)
