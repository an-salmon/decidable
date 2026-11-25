"""Exact local rules for stretched-exponential size bounds (Os)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import lcm, prod

from sympy import Add, Expr, Float, Mul, Pow, Rational, S, Symbol, exp, log

from ..report import CheckResult
from .enclosures import exp_enclosure
from .moments import Exact, exact


_LOG_TWO = log(2)
_SCOPE = "On the stated size, tail, pointwise, independence and integral hypotheses"


def _quantity(expression) -> Expr:
    if type(expression) in (int, Fraction):
        expression = Rational(expression)
    if not isinstance(expression, Expr) or expression.is_commutative is not True:
        raise TypeError("the quantity must be a commutative SymPy expression")
    if expression.has(Float):
        raise TypeError("the quantity expression must be exact")
    if expression.is_real is False or expression.has(
        S.Infinity, S.NegativeInfinity, S.ComplexInfinity, S.NaN
    ):
        raise ValueError("the quantity must be finite and real")
    return expression


def _sign(expression: Expr, nonnegative: bool) -> None:
    if type(nonnegative) is not bool:
        raise TypeError("nonnegative must be a bool")
    if nonnegative and expression.is_nonnegative is False:
        raise ValueError("the nonnegative declaration contradicts the expression")


def _reciprocal(value: Exact) -> Fraction:
    value = exact(value, "reciprocal exponent")
    if value <= 0:
        raise ValueError("reciprocal exponent must be positive")
    return value


def _positive_scale(scale) -> Expr:
    if type(scale) in (int, Fraction):
        value = exact(scale, "scale")
        scale = Rational(value.numerator, value.denominator)
    if isinstance(scale, Rational) and scale > 0:
        return scale
    if scale == _LOG_TWO or (type(scale) is Symbol and scale.is_positive is True):
        return scale
    if type(scale) in (Add, Mul):
        for term in scale.args:
            _positive_scale(term)
        return scale
    if type(scale) is Pow and isinstance(scale.exp, Rational):
        _positive_scale(scale.base)
        return scale
    raise ValueError("scale must be a positive exact expression")


@dataclass(frozen=True)
class SizeBound:
    """A given or derived judgement X <= O_s(scale), stored using 1/s."""

    expression: Expr
    reciprocal: Exact
    scale: Expr | Exact
    nonnegative: bool

    def __post_init__(self):
        expression = _quantity(self.expression)
        _sign(expression, self.nonnegative)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "reciprocal", _reciprocal(self.reciprocal))
        object.__setattr__(self, "scale", _positive_scale(self.scale))


@dataclass(frozen=True)
class UpperBound:
    """A caller-supplied pointwise hypothesis X <= a deterministic ceiling."""

    expression: Expr
    ceiling: Expr | Exact

    def __post_init__(self):
        object.__setattr__(self, "expression", _quantity(self.expression))


@dataclass(frozen=True)
class PointwiseOrder:
    """A caller-supplied pointwise hypothesis lower <= upper.

    With nonnegative=True the hypothesis also declares lower >= 0.
    """

    lower: Expr
    upper: Expr
    nonnegative: bool = False

    def __post_init__(self):
        lower = _quantity(self.lower)
        _sign(lower, self.nonnegative)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", _quantity(self.upper))


@dataclass(frozen=True)
class TailBound:
    """P(X comparison scale*h) <= prefactor*exp(-h**(1/reciprocal)).

    The bound is stated for all h above threshold, including threshold when
    include_threshold is true. The sign is a separate hypothesis about X.
    """

    expression: Expr
    reciprocal: Exact
    scale: Expr | Exact
    prefactor: Exact
    comparison: str = ">="
    threshold: Exact = 0
    include_threshold: bool = False
    nonnegative: bool = False

    def __post_init__(self):
        expression = _quantity(self.expression)
        _sign(expression, self.nonnegative)
        prefactor = exact(self.prefactor, "tail prefactor")
        threshold = exact(self.threshold, "tail threshold")
        if prefactor <= 0 or threshold < 0:
            raise ValueError("the tail prefactor must be positive and the threshold nonnegative")
        if type(self.comparison) is not str or self.comparison not in (">=", ">"):
            raise ValueError("the tail comparison must be >= or >")
        if type(self.include_threshold) is not bool:
            raise TypeError("include_threshold must be a bool")
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "reciprocal", _reciprocal(self.reciprocal))
        object.__setattr__(self, "scale", _positive_scale(self.scale))
        object.__setattr__(self, "prefactor", prefactor)
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True)
class Independence:
    """A caller-supplied joint independence declaration for an ordered tuple."""

    expressions: tuple[Expr, ...]

    def __post_init__(self):
        expressions = tuple(_quantity(expression) for expression in self.expressions)
        if not expressions:
            raise ValueError("an independence declaration needs at least one quantity")
        object.__setattr__(self, "expressions", expressions)


@dataclass(frozen=True)
class BoundedMean:
    """Assume 0 <= X <= ceiling and E[X] <= mean_ceiling, with deterministic ceilings."""

    expression: Expr
    ceiling: Expr | Exact
    mean_ceiling: Expr | Exact

    def __post_init__(self):
        expression = _quantity(self.expression)
        _sign(expression, True)
        mean_ceiling = _quantity(self.mean_ceiling)
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "ceiling", _positive_scale(self.ceiling))
        object.__setattr__(self, "mean_ceiling", _positive_scale(mean_ceiling)
                           if mean_ceiling != 0 else S.Zero)


@dataclass(frozen=True)
class IntegralAggregationPremise:
    """Declare the common measure and hypotheses for measurable aggregation.

    The expression is the integral of the nonnegative integrand against the
    named measure. Its deterministic positive measurable scale density has
    integral total_scale. Under the resulting finite weighted measure the
    integrand divided by that density is jointly measurable and has the
    stated O_s(1) bound for almost every index.
    """

    expression: Expr
    integrand: Expr
    scale_density: Expr
    measure: str
    reciprocal: Exact
    total_scale: Expr | Exact

    def __post_init__(self):
        if type(self.measure) is not str or not self.measure:
            raise ValueError("the aggregation measure needs a name")
        expression, integrand = _quantity(self.expression), _quantity(self.integrand)
        _sign(expression, True)
        _sign(integrand, True)
        density = _quantity(self.scale_density)
        if density.is_positive is False:
            raise ValueError("the scale density must be positive")
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "integrand", integrand)
        object.__setattr__(self, "scale_density", density)
        object.__setattr__(self, "reciprocal", _reciprocal(self.reciprocal))
        object.__setattr__(self, "total_scale", _positive_scale(self.total_scale))


@dataclass(frozen=True)
class BernsteinTailBound:
    """A fixed-threshold bound P(expression >= threshold) <= exp(-rate).

    A rate of None instead denotes an event of probability zero.
    """

    expression: Expr
    threshold: Exact
    rate: Exact | None

    def __post_init__(self):
        threshold = exact(self.threshold, "tail threshold")
        rate = None if self.rate is None else exact(self.rate, "tail exponent")
        if threshold <= 0 or (rate is not None and rate <= 0):
            raise ValueError("the Bernstein tail threshold and finite rate must be positive")
        object.__setattr__(self, "expression", _quantity(self.expression))
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "rate", rate)

    @property
    def probability_upper(self) -> Expr:
        return S.Zero if self.rate is None else exp(-Rational(self.rate))


class _UnprovedScale(ValueError):
    """A scale comparison is outside the supported proof templates."""


def _require_size(bound: SizeBound) -> None:
    if type(bound) is not SizeBound:
        raise TypeError("this rule requires a SizeBound")


def multiply(bound: SizeBound, coefficient: Expr | Exact) -> SizeBound:
    """Multiply a quantity and its scale by the same positive deterministic factor."""
    _require_size(bound)
    coefficient = _positive_scale(coefficient)
    return SizeBound(coefficient * bound.expression, bound.reciprocal,
                     coefficient * bound.scale, bound.nonnegative)


def monotone(bound: SizeBound, premise: PointwiseOrder) -> SizeBound:
    """Transfer a size bound to a quantity pointwise at most the input."""
    _require_size(bound)
    if type(premise) is not PointwiseOrder or premise.upper != bound.expression:
        raise ValueError("monotonicity needs a matching pointwise order")
    nonnegative = (premise.nonnegative or premise.lower.is_nonnegative is True
                   or (premise.lower == bound.expression and bound.nonnegative))
    return SizeBound(premise.lower, bound.reciprocal, bound.scale, nonnegative)


def _rational_monomial_at_most_one(expression: Expr) -> bool:
    powers = []
    for term in Mul.make_args(expression):
        if isinstance(term, Rational) and term > 0:
            powers.append((Fraction(term), Fraction(1)))
        elif (type(term) is Pow and isinstance(term.base, Rational) and term.base > 0
              and isinstance(term.exp, Rational)):
            powers.append((Fraction(term.base), Fraction(term.exp)))
        else:
            return False
    denominator = lcm(*(exponent.denominator for _, exponent in powers))
    return prod(base**int(denominator * exponent) for base, exponent in powers) <= 1


def _sum_coefficients(expression: Expr) -> dict[Expr, Fraction]:
    coefficients = {}
    for term in Add.make_args(expression):
        coefficient, atom = term.as_coeff_Mul(rational=True)
        coefficients[atom] = coefficients.get(atom, Fraction(0)) + Fraction(coefficient)
    return coefficients


def enlarge_scale(bound: SizeBound, scale: Expr | Exact) -> SizeBound:
    """Enlarge a scale by an exact ratio or a coefficientwise positive-sum check."""
    _require_size(bound)
    scale = _positive_scale(scale)
    if bound.scale != scale and not _rational_monomial_at_most_one(bound.scale / scale):
        target = _sum_coefficients(scale)
        if any(value > target.get(term, 0)
               for term, value in _sum_coefficients(bound.scale).items()):
            raise _UnprovedScale("The requested scale enlargement could not be proved.")
    return SizeBound(bound.expression, bound.reciprocal, scale, bound.nonnegative)


def tail_from_size(bound: SizeBound) -> TailBound:
    """Apply Markov's inequality at every nonnegative threshold."""
    _require_size(bound)
    return TailBound(bound.expression, bound.reciprocal, bound.scale, 2,
                     include_threshold=True, nonnegative=bound.nonnegative)


def size_from_tail(bound: TailBound) -> SizeBound:
    """Apply the sharp reverse rule for prefactor one at every positive threshold."""
    if type(bound) is not TailBound or bound.prefactor != 1 or bound.threshold != 0:
        raise ValueError("the reverse tail rule needs prefactor one at all positive thresholds")
    # Strict tail comparisons suffice for the layer-cake integral.
    scale = 2 ** Rational(bound.reciprocal) * bound.scale
    return SizeBound(bound.expression, bound.reciprocal, scale, bound.nonnegative)


def _matching_bounded_means(
    declaration: Independence, premises: Iterable[BoundedMean],
) -> tuple[tuple[Expr, ...], Expr, Expr]:
    if type(declaration) is not Independence:
        raise TypeError("the independent mean requires an independence declaration")
    premises = tuple(premises)
    if not premises or any(type(premise) is not BoundedMean for premise in premises):
        raise TypeError("supply a BoundedMean premise for each summand")
    expressions = tuple(premise.expression for premise in premises)
    if expressions != declaration.expressions:
        raise ValueError("independence and bounded means must describe the same summands")
    ceiling, mean_ceiling = premises[0].ceiling, premises[0].mean_ceiling
    if any(premise.ceiling != ceiling or premise.mean_ceiling != mean_ceiling
           for premise in premises):
        raise ValueError("all summands must have the same pointwise and mean ceilings")
    return expressions, ceiling, mean_ceiling


def independent_mean(
    declaration: Independence, premises: Iterable[BoundedMean], parameter: Exact = 1,
) -> SizeBound:
    """Bound the mean minus twice its common mean ceiling under independence.

    The matching premises declare a common positive deterministic ceiling M
    and a common nonnegative deterministic mean ceiling mu. An exact
    exponential enclosure verifies exp(a) <= 1 + 2*a for the positive
    parameter a. The resulting O_1 scale is M/(a*n).
    """
    expressions, ceiling, mean_ceiling = _matching_bounded_means(declaration, premises)

    parameter = exact(parameter, "exponential parameter")
    if not 0 < parameter <= 2 or exp_enclosure(parameter, 12)[1] > 1 + 2 * parameter:
        raise ValueError("the parameter must satisfy the exact exp(a) <= 1 + 2*a check")

    # The chord on [0,M] yields E exp(a X/M) <= exp(mu (exp(a)-1)/M).
    # Independence multiplies these bounds. Centering the mean at 2*mu then
    # makes the exponential moment at scale M/(a*n) at most one, so taking
    # the positive part gives a moment at most two.
    count = len(expressions)
    return SizeBound(Add(*expressions) / count - 2 * mean_ceiling, 1,
                     ceiling / (parameter * count), mean_ceiling == 0)


def independent_bernstein(
    declaration: Independence, premises: Iterable[BoundedMean], threshold: Exact,
) -> BernsteinTailBound:
    """Replay the Bernstein tail for the independent mean above its mean ceiling.

    The two common deterministic ceilings and the positive threshold must
    be exact rationals. At zero mean ceiling every summand vanishes almost
    surely, giving probability zero.
    """
    expressions, ceiling, mean_ceiling = _matching_bounded_means(declaration, premises)
    if not isinstance(ceiling, Rational) or not isinstance(mean_ceiling, Rational):
        raise TypeError("Bernstein replay requires rational pointwise and mean ceilings")
    maximum, mean = Fraction(ceiling), Fraction(mean_ceiling)
    threshold = exact(threshold, "tail threshold")
    if threshold <= 0:
        raise ValueError("the Bernstein tail threshold must be positive")
    count = len(expressions)
    centered = Add(*expressions) / count - mean_ceiling
    if mean == 0:
        return BernsteinTailBound(centered, threshold, None)

    eta = threshold / (maximum * (mean + threshold / 3))
    if not 0 < eta * maximum < 3:
        raise ArithmeticError("the Bernstein parameter is outside the pointwise bound's domain")
    # The Taylor bound exp(v) <= 1 + v + v²/(2*(1-v/3)) follows from
    # k! >= 2*3**(k-2) for k >= 2 and applies at v = eta*X < 3. Use
    # X² <= M*X, then independence and Markov for the centered mean.
    remainder = eta**2 * maximum * mean / (2 * (1 - eta * maximum / 3))
    markov_rate = count * (eta * threshold - remainder)
    rate = count * threshold**2 / (2 * maximum * (mean + threshold / 3))
    if markov_rate != rate:
        raise ArithmeticError("the Bernstein exponent did not match the Markov calculation")
    return BernsteinTailBound(centered, threshold, rate)


def power(bound: SizeBound, exponent: Exact) -> SizeBound:
    """Propagate a positive rational power of a nonnegative quantity."""
    _require_size(bound)
    exponent = exact(exponent, "power")
    if exponent <= 0 or not bound.nonnegative:
        raise ValueError("a power requires a positive exponent and nonnegative input")
    return SizeBound(
        bound.expression**exponent, bound.reciprocal * exponent, bound.scale**exponent, True
    )


def product(bounds: Iterable[SizeBound]) -> SizeBound:
    """Multiply nonnegative inputs with no dependence assumptions."""
    bounds = tuple(bounds)
    for bound in bounds:
        _require_size(bound)
    if not bounds or not all(bound.nonnegative for bound in bounds):
        raise ValueError("a product requires nonempty nonnegative inputs")
    return SizeBound(
        Mul(*(bound.expression for bound in bounds)),
        sum(bound.reciprocal for bound in bounds),
        Mul(*(bound.scale for bound in bounds)),
        True,
    )


def descend(bound: SizeBound, reciprocal: Exact) -> SizeBound:
    """Decrease s using the sharp uniform factor from Proposition 3.2."""
    _require_size(bound)
    reciprocal = exact(reciprocal, "reciprocal exponent")
    if reciprocal < bound.reciprocal:
        raise ValueError("descent requires an output tail exponent no larger than the input")
    scale = bound.scale * _LOG_TWO**(bound.reciprocal - reciprocal)
    return SizeBound(bound.expression, reciprocal, scale, bound.nonnegative)


def bounded_raise(bound: SizeBound, reciprocal: Exact, premise: UpperBound) -> SizeBound:
    """Increase s using a deterministic positive ceiling for a nonnegative input."""
    _require_size(bound)
    reciprocal = exact(reciprocal, "reciprocal exponent")
    if not 0 < reciprocal <= bound.reciprocal:
        raise ValueError("bounded raise requires a positive reciprocal no larger than the input")
    if type(premise) is not UpperBound or not bound.nonnegative or premise.expression != bound.expression:
        raise ValueError("bounded raise needs a matching upper bound and nonnegative input")
    ceiling = _positive_scale(premise.ceiling)
    ratio = reciprocal / bound.reciprocal
    return SizeBound(bound.expression, reciprocal, ceiling**(1 - ratio) * bound.scale**ratio, True)


def _aggregation_factor(reciprocal: Fraction, unit_factor: bool) -> Expr | int:
    if type(unit_factor) is not bool:
        raise TypeError("unit_factor must be a bool")
    if unit_factor and reciprocal > 1:
        raise ValueError("the unit-factor aggregation rule requires s >= 1")
    return (1 + (reciprocal - 1) / _LOG_TWO)**reciprocal if reciprocal > 1 else 1


def aggregate(bounds: Iterable[SizeBound], *, unit_factor: bool = False) -> SizeBound:
    """Sum bounds at a common tail exponent, retaining the aggregation constant."""
    bounds = tuple(bounds)
    for bound in bounds:
        _require_size(bound)
    if not bounds:
        raise ValueError("a sum requires at least one bound")
    reciprocal = bounds[0].reciprocal
    if any(bound.reciprocal != reciprocal for bound in bounds):
        raise ValueError("sum inputs must share their tail exponent")
    factor = _aggregation_factor(reciprocal, unit_factor)
    return SizeBound(
        Add(*(bound.expression for bound in bounds)),
        reciprocal,
        factor * Add(*(bound.scale for bound in bounds)),
        all(bound.nonnegative for bound in bounds),
    )


def weighted_aggregate(
    bounds: Iterable[SizeBound], weights: Iterable[Exact], *, unit_factor: bool = False
) -> SizeBound:
    """Sum using exact nonnegative weights, with at least one positive weight."""
    bounds = tuple(bounds)
    for bound in bounds:
        _require_size(bound)
    weights = tuple(exact(weight, "aggregation weight") for weight in weights)
    if len(bounds) != len(weights) or any(weight < 0 for weight in weights):
        raise ValueError("supply one nonnegative weight per input")
    active = tuple(multiply(bound, weight) for bound, weight in zip(bounds, weights) if weight)
    if not active:
        raise ValueError("at least one aggregation weight must be positive")
    return aggregate(active, unit_factor=unit_factor)


def integral_aggregate(premise: IntegralAggregationPremise) -> SizeBound:
    """Apply the measurable aggregation theorem to the declared integral."""
    if type(premise) is not IntegralAggregationPremise:
        raise TypeError("integral aggregation needs an IntegralAggregationPremise")
    factor = _aggregation_factor(premise.reciprocal, False)
    return SizeBound(premise.expression, premise.reciprocal, factor * premise.total_scale, True)


@dataclass(frozen=True)
class DerivationNode:
    id: str
    rule: str
    operands: tuple[str, ...] = ()
    argument: object = None


def _require_deterministic_symbols(allowed_symbols: frozenset[Symbol], *expressions: Expr) -> None:
    if any(expression.free_symbols - allowed_symbols for expression in expressions):
        raise ValueError("the rule introduces an undeclared deterministic symbol")


def _proof_scale(bound: SizeBound, scale, deterministic_symbols: frozenset[Symbol]) -> Expr:
    _require_size(bound)
    scale = _positive_scale(scale)
    _require_deterministic_symbols(deterministic_symbols.union(bound.scale.free_symbols), scale)
    return scale


def verify_derivation(
    givens: Mapping[str, SizeBound],
    nodes: Iterable[DerivationNode],
    goal: SizeBound | TailBound | BernsteinTailBound,
    *,
    upper_bounds: Mapping[str, UpperBound] | None = None,
    pointwise_orders: Mapping[str, PointwiseOrder] | None = None,
    tail_givens: Mapping[str, TailBound] | None = None,
    independence: Mapping[str, Independence] | None = None,
    bounded_means: Mapping[str, BoundedMean] | None = None,
    integral_aggregations: Mapping[str, IntegralAggregationPremise] | None = None,
    deterministic_symbols: Iterable[Symbol] = (),
) -> CheckResult:
    """Check a topologically ordered derivation whose last node is its goal.

    Given and tail_given nodes name an input. Power and descend take a
    rational argument. Bounded_raise takes (reciprocal, upper-bound id),
    and monotone takes a pointwise-order id. Multiply and enlarge_scale
    take a positive deterministic expression. Product, sum, sum_unit,
    tail_from_size and size_from_tail have no argument. Weighted_aggregate
    and weighted_aggregate_unit take a list of exact weights. Integral_aggregate
    names an integral hypothesis. Independent_mean and independent_bernstein
    take (independence id, bounded-mean ids, parameter), with the threshold as
    parameter for Bernstein. These three rules take no operands.
    New symbols in a deterministic factor, ceiling or total integral scale
    must be declared as positive deterministic_symbols. A symbol in the
    operand's scale is already declared deterministic.
    """
    proved = {}
    upper_bounds = {} if upper_bounds is None else upper_bounds
    pointwise_orders = {} if pointwise_orders is None else pointwise_orders
    tail_givens = {} if tail_givens is None else tail_givens
    independence = {} if independence is None else independence
    bounded_means = {} if bounded_means is None else bounded_means
    integral_aggregations = {} if integral_aggregations is None else integral_aggregations
    if type(goal) not in (SizeBound, TailBound, BernsteinTailBound):
        return CheckResult("inconclusive", _SCOPE, "The goal must be a typed size or tail bound.")
    try:
        deterministic_symbols = frozenset(deterministic_symbols)
        if any(type(symbol) is not Symbol or symbol.is_positive is not True
               for symbol in deterministic_symbols):
            raise ValueError("deterministic_symbols must contain only positive symbols")
    except (TypeError, ValueError) as error:
        return CheckResult("inconclusive", _SCOPE, str(error))
    for node in nodes:
        if type(node) is not DerivationNode:
            return CheckResult("inconclusive", _SCOPE, "Every proof node must be a DerivationNode.")
        try:
            if type(node.id) is not str or not node.id or node.id in proved:
                raise ValueError("node identifiers must be nonempty and unique")
            if (type(node.rule) is not str or type(node.operands) is not tuple
                    or any(type(operand) is not str for operand in node.operands)):
                raise TypeError("a node needs a rule name and a tuple of operand identifiers")
            operands = tuple(proved[operand] for operand in node.operands)
            if node.rule == "given" and not operands:
                result = givens[node.argument]
                _require_size(result)
            elif node.rule == "tail_given" and not operands:
                result = tail_givens[node.argument]
                if type(result) is not TailBound:
                    raise TypeError("a tail input must be a TailBound")
            elif node.rule == "power" and len(operands) == 1:
                result = power(operands[0], node.argument)
            elif node.rule == "descend" and len(operands) == 1:
                result = descend(operands[0], node.argument)
            elif node.rule == "multiply" and len(operands) == 1:
                factor = _proof_scale(operands[0], node.argument, deterministic_symbols)
                result = multiply(operands[0], factor)
            elif node.rule == "monotone" and len(operands) == 1:
                result = monotone(operands[0], pointwise_orders[node.argument])
            elif node.rule == "enlarge_scale" and len(operands) == 1:
                scale = _proof_scale(operands[0], node.argument, deterministic_symbols)
                result = enlarge_scale(operands[0], scale)
            elif node.rule == "bounded_raise" and len(operands) == 1:
                reciprocal, premise_id = node.argument
                premise = upper_bounds[premise_id]
                if type(premise) is not UpperBound:
                    raise TypeError("bounded raise requires an UpperBound premise")
                _proof_scale(operands[0], premise.ceiling, deterministic_symbols)
                result = bounded_raise(operands[0], reciprocal, premise)
            elif node.rule == "product" and node.argument is None:
                result = product(operands)
            elif node.rule in ("sum", "sum_unit") and node.argument is None:
                result = aggregate(operands, unit_factor=node.rule == "sum_unit")
            elif node.rule in ("weighted_aggregate", "weighted_aggregate_unit"):
                result = weighted_aggregate(
                    operands, node.argument, unit_factor=node.rule == "weighted_aggregate_unit"
                )
            elif node.rule == "integral_aggregate" and not operands:
                premise = integral_aggregations[node.argument]
                result = integral_aggregate(premise)
                _require_deterministic_symbols(deterministic_symbols, premise.total_scale)
            elif node.rule == "tail_from_size" and len(operands) == 1 and node.argument is None:
                result = tail_from_size(operands[0])
            elif node.rule == "size_from_tail" and len(operands) == 1 and node.argument is None:
                result = size_from_tail(operands[0])
            elif node.rule in ("independent_mean", "independent_bernstein") and not operands:
                declaration, premise_ids, parameter = node.argument
                if type(premise_ids) is not tuple or any(type(item) is not str for item in premise_ids):
                    raise TypeError("bounded-mean identifiers must be a tuple of names")
                premises = tuple(bounded_means[identifier] for identifier in premise_ids)
                rule = independent_mean if node.rule == "independent_mean" else independent_bernstein
                result = rule(independence[declaration], premises, parameter)
                _require_deterministic_symbols(
                    deterministic_symbols, premises[0].ceiling, premises[0].mean_ceiling,
                )
            else:
                raise ValueError("unknown rule or invalid operands")
        except _UnprovedScale as error:
            return CheckResult("unsupported", _SCOPE, f"Node {node.id}: {error}")
        except (KeyError, TypeError, ValueError) as error:
            return CheckResult("inconclusive", _SCOPE, f"Node {node.id}: {error}")
        proved[node.id] = result
    if not proved or result != goal:
        return CheckResult("inconclusive", _SCOPE, "The final judgement does not match the goal.")
    return CheckResult("verified", _SCOPE, "The derivation proves the requested bound.", result)
