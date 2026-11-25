"""Decide polynomial moment implications on semialgebraic supports."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm

from sympy import Add, Eq, Ge, Integer, Le, Mul, Poly, Pow, Rational, Symbol, false, true
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Not, Or

from ..rational import _validate_polynomial
from ..report import CheckResult
from ._nra import BackendUnavailable, solve
from .moments import Exact, MomentProblem, _InvalidMomentInput, exact


def _call(operation: str, arguments, identity: str | None = None) -> str:
    arguments = tuple(arguments)
    if not arguments:
        if identity is None:
            raise ValueError("missing polynomial operands")
        return identity
    if len(arguments) == 1 and identity is not None:
        return arguments[0]
    return f"({operation} {' '.join(arguments)})"


def _rational(value) -> str:
    value = Fraction(value)
    numerator = str(abs(value.numerator))
    result = numerator if value.denominator == 1 else f"(/ {numerator} {value.denominator})"
    return f"(- {result})" if value < 0 else result


def _polynomial(expression, variables) -> str:
    if type(expression) in (int, Fraction) or isinstance(expression, Rational):
        return _rational(expression)
    if type(expression) is Symbol and expression in variables:
        return variables[expression]
    if isinstance(expression, Poly):
        return _polynomial(_validate_polynomial(expression, tuple(variables)).as_expr(), variables)
    if type(expression) in (Add, Mul):
        operation, identity = ("+", "0") if type(expression) is Add else ("*", "1")
        return _call(operation, (_polynomial(term, variables) for term in expression.args), identity)
    if type(expression) is Pow and isinstance(expression.exp, Integer) and expression.exp >= 0:
        base = _polynomial(expression.base, variables)
        return "1" if expression.exp == 0 else f"(^ {base} {expression.exp})"
    raise ValueError("use exact polynomials in the declared coordinates")


def _predicate(predicate, variables) -> str:
    if predicate is true or predicate is True:
        return "true"
    if predicate is false or predicate is False:
        return "false"
    if type(predicate) is And:
        return _call("and", (_predicate(term, variables) for term in predicate.args), "true")
    if type(predicate) is Or:
        return _call("or", (_predicate(term, variables) for term in predicate.args), "false")
    if type(predicate) is Not:
        return _call("not", (_predicate(predicate.args[0], variables),))
    if isinstance(predicate, Relational):
        operator = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "=", "!=": "distinct"}
        return _call(operator[predicate.rel_op],
                     (_polynomial(side, variables) for side in (predicate.lhs, predicate.rhs)))
    raise ValueError("use Boolean combinations of polynomial comparisons")


@dataclass(frozen=True)
class SemialgebraicMomentProblem:
    """Exact rational polynomials and a Boolean polynomial support predicate.

    All input and output integrands are required to be nonnegative on the
    support. Algebraic graphs can be represented using extra coordinates
    and polynomial constraints, including an isolating interval when needed.
    """

    coordinates: tuple[Symbol, ...]
    support: object
    inputs: tuple[object, ...]
    budgets: tuple[Exact, ...]
    output: object
    output_budget: Exact

    def __post_init__(self):
        coordinates = tuple(self.coordinates)
        inputs = tuple(self.inputs)
        budgets = tuple(exact(value, "input budget") for value in self.budgets)
        budget = exact(self.output_budget, "output budget")
        if (not coordinates or any(type(symbol) is not Symbol or symbol.is_commutative is not True
                                   for symbol in coordinates)
                or len({symbol.name for symbol in coordinates}) != len(coordinates)):
            raise ValueError("declare distinct real coordinate symbols")
        if any(symbol not in (Symbol(symbol.name), Symbol(symbol.name, real=True))
               for symbol in coordinates):
            raise ValueError("put coordinate restrictions in the support predicate")
        if len(inputs) != len(budgets) or budget < 0 or any(value < 0 for value in budgets):
            raise ValueError("supply nonnegative budgets for every integrand")
        variables = {symbol: f"x{i}" for i, symbol in enumerate(coordinates)}
        _predicate(self.support, variables)
        for expression in (*inputs, self.output):
            _polynomial(expression, variables)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "output_budget", budget)

    @classmethod
    def from_interval(cls, problem: MomentProblem) -> SemialgebraicMomentProblem:
        coordinate = Symbol(problem.output.gen.name)
        support = Ge(coordinate, Rational(problem.lower), evaluate=False)
        if problem.upper is not None:
            support = And(support, Le(coordinate, Rational(problem.upper), evaluate=False))
        polynomials = tuple(Poly.from_dict(poly.as_dict(), coordinate, domain="QQ")
                            for poly in (*problem.inputs, problem.output))
        return cls((coordinate,), support, polynomials[:-1], problem.budgets,
                   polynomials[-1], problem.output_budget)


@dataclass(frozen=True)
class AlgebraicLaw:
    """An exact SMT-LIB assignment to the padded atoms and their masses."""

    atom_coordinates: tuple[tuple[str, ...], ...]
    mass_names: tuple[str, ...]
    assignment: str


def exponential_moment_problem(inputs, budgets, output, output_budget) -> SemialgebraicMomentProblem:
    """Use the common exponential chart for sums of rational-rate exponentials.

    Each integrand is a sequence of (coefficient, rate) for c*exp(a*v),
    with v >= 0. Negative rates use the reciprocal of the chart coordinate.
    The solver checks that all integrands are nonnegative on the support.
    """
    integrands = tuple(
        tuple((exact(coefficient, "exponential coefficient"), exact(rate, "exponential rate"))
              for coefficient, rate in integrand)
        for integrand in (*inputs, output)
    )
    denominator = lcm(*(rate.denominator for integrand in integrands for _, rate in integrand))
    z, inverse = Symbol("z"), Symbol("z_inverse")
    negative = any(rate < 0 for integrand in integrands for _, rate in integrand)
    support = Ge(z, 1)
    coordinates = (z,)
    if negative:
        support = And(support, Eq(z * inverse, 1))
        coordinates = (z, inverse)
    polynomials = tuple(
        Add(*(Rational(coefficient) * (inverse if rate < 0 else z)**int(abs(rate) * denominator)
              for coefficient, rate in integrand))
        for integrand in integrands
    )
    return SemialgebraicMomentProblem(coordinates, support, polynomials[:-1],
                                     tuple(budgets), polynomials[-1], output_budget)


def _script(names, assertions) -> str:
    declarations = "".join(f"(declare-const {name} Real)" for name in names)
    return declarations + "".join(f"(assert {condition})" for condition in assertions)


def decide_moment(
    problem: MomentProblem | SemialgebraicMomentProblem, *, timeout_ms: int | None = None
) -> CheckResult:
    """Decide the bound by exact existential search for a finite counterexample.

    For any violation truncate the nonnegative output, finitely represent the
    resulting integrals with the input moments and unit mass, and maximize on
    that finite support. A maximizing vertex uses at most m+1 positive masses,
    so the polynomial query is complete even at a boundary input budget.
    The optional system Z3 library is the exact decision backend.
    """
    if timeout_ms is not None and (type(timeout_ms) is not int or timeout_ms <= 0):
        raise ValueError("timeout_ms must be None or a positive integer")
    if isinstance(problem, MomentProblem):
        problem = SemialgebraicMomentProblem.from_interval(problem)
        check_nonnegative = False  # Already checked by exact real-root isolation.
    elif isinstance(problem, SemialgebraicMomentProblem):
        check_nonnegative = True
    else:
        raise TypeError("supply a polynomial moment problem")
    scope = "All probability laws on the supplied semialgebraic support"
    integrands = (*problem.inputs, problem.output)

    try:
        if check_nonnegative:
            variables = {symbol: f"x{i}" for i, symbol in enumerate(problem.coordinates)}
            negative = _call("or", (f"(< {_polynomial(item, variables)} 0)" for item in integrands),
                             "false")
            validation = _script(tuple(variables.values()),
                                 (_predicate(problem.support, variables), negative))
            verdict, _ = solve(validation, (), timeout_ms)
            if verdict == "sat":
                raise _InvalidMomentInput("an integrand is negative on the declared support")
            if verdict == "unknown":
                return CheckResult("inconclusive", scope, "The solver could not validate the integrands.")

        count = len(problem.inputs) + 1
        atoms = tuple(tuple(f"x{i}_{j}" for j in range(len(problem.coordinates)))
                      for i in range(count))
        masses = tuple(f"p{i}" for i in range(count))
        charts = tuple(dict(zip(problem.coordinates, row)) for row in atoms)
        names = (*masses, *(name for row in atoms for name in row))
        assertions = [f"(= {_call('+', masses, '0')} 1)"]
        for mass, chart in zip(masses, charts):
            assertions.extend((f"(>= {mass} 0)",
                               f"(=> (> {mass} 0) {_predicate(problem.support, chart)})"))
        means = tuple(_call("+", (f"(* {mass} {_polynomial(item, chart)})"
                                  for mass, chart in zip(masses, charts)), "0")
                      for item in integrands)
        assertions.extend(f"(<= {mean} {_rational(budget)})"
                          for mean, budget in zip(means, problem.budgets))
        assertions.append(f"(> {means[-1]} {_rational(problem.output_budget)})")
        verdict, witness = solve(_script(names, assertions), names, timeout_ms)
    except BackendUnavailable as error:
        return CheckResult("unsupported", scope, str(error))
    if verdict == "unsat":
        return CheckResult("verified", scope, f"No violating law with at most {count} atoms exists.")
    if verdict == "sat":
        return CheckResult("disproved", scope, "An exact algebraic law violates the output budget.",
                           AlgebraicLaw(atoms, masses, witness))
    return CheckResult("inconclusive", scope, "The exact solver returned unknown.")
