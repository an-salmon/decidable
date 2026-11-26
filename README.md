# decidable

`decidable` checks exact algebraic claims that arise inside mathematical
proofs. Choose the fragment that matches the claim and supply exact inputs,
the hypotheses, and a certificate when the operation requires one. The caller
justifies the reduction from the original mathematical problem to the encoded
claim.

The table lists direct Python calls, each with its own return type.

| Fragment ID | Direct check | What it establishes |
| --- | --- | --- |
| `rational` | `rational_identity` | Equality of rational functions over the rationals. |
| `noncommutative` | `noncommutative_identity` | Equality of expressions in matrix words modulo supplied relations. |
| `gram` | `verify_gram_certificate` | Validity of a supplied certificate for a linear inequality in inner products. |
| `polynomial` | `verify_polynomial_certificate` | Validity of a supplied certificate for a real polynomial inequality. |
| `linear` | `linear_feasible`, `linear_implies` | Feasibility and implication for finite systems of affine inequalities. |
| `series` | `verify_series_certificate` | Validity of a supplied certificate for a coefficientwise inequality of formal series. |
| `linalg` | `matrix_rank` | Rank of an explicit matrix over the rationals. |

`dispatch(fragment, operation, claim)` accepts a typed claim such as
`GramCertificateClaim(1, {(0, 0): 1}, squares=((1, (1,)),))` and returns a
`CheckResult` with a status, scope, and a witness when available. The
`decidable.fragments.FRAGMENTS` registry lists these supported routes.

| Fragment ID | Dispatch operation IDs |
| --- | --- |
| `gram` | `verify_gram_certificate` |
| `linear` | `linear_implies`, `find_positive_state` |
| `series` | `check_kernel_domination`, `check_exp_series` |
| `linalg` | `matrix_rank` |

An unknown route or mismatched claim type yields `unsupported`. A certificate
that fails exact replay yields `inconclusive`. A complete decision procedure
may return `disproved` within the scope it reports.

## Run the examples

With Python 3.9 or newer, run these commands from the repository root.

```sh
python -m pip install -e .
python examples/basic.py
python examples/parameterized_series.py
python examples/harmonic_excess.py
python -m unittest discover -s tests
```

## Rational function identities

Pass the numerator and denominator of each expression as separate SymPy
`Poly` objects over the integers or rationals, with the same ordered
generators. The checker compares the cross-products in exact arithmetic and
rejects identically zero denominators. To use a verified identity at a
particular point, also establish that both original denominators are nonzero
there.

## Monomial and jet checks

`decidable.checks.loglinear_zero` checks whether a product of powers of
positive bases equals one. Declare the positive base symbols and the
symbols allowed in rational exponents. Use `evaluate=False` when building
products and powers whose original structure must be checked. The identity
holds wherever the original exponent denominators are nonzero.

For an explicit jet certificate, use `total_derivative` to compute each
derivative of the hypothesis, multiply by the corresponding certificate
polynomial, and subtract the sum from the target. Pass that residual to
`check_jet_identity` with its default right-hand side of zero. To combine
a fixed list of dimensions in a report, return a Boolean indicating
whether every per-dimension result is verified. Use the dimension list as
the scope and mark the aggregate check and its dependent assemblies `finite`.

## Noncommutative identities

Represent an expression as a dictionary from words to rational coefficients.
A word is a tuple of matrix symbol names, and `()` is the identity. Each rule
`(left_word, right_word)` replaces the longer word with the shorter one. The
caller supplies the relations that hold for the intended matrices. Omit
`rules` to check the free associative algebra. For example, the commutator
`Ab - bA` is `{("A", "b"): 1, ("b", "A"): -1}` and compares unequal to zero.
When converting an expression to words, preserve the order of every matrix
factor before any commutative simplification.

Matching reduced expressions establish the identity. If the expressions differ
and the rules are proved confluent, `False` means the identity fails in the
formal algebra with those relations. When confluence cannot be established,
different reductions give `None`.

Use `bilinear_identity` for expressions `pᵀWq`. Map the triples
`(p, W, q)` to coefficients and declare `adjoints` as an involution on
the matrix symbols. A fixed point declares a self-adjoint letter. The
checker uses scalar transposition and closes the word rules under transpose.

## Gram certificates

The certificate applies to any choice of the given number of vectors in a real
inner-product space. Represent a linear expression in their inner products as
a dictionary with keys `(i, j)`, where `0 <= i <= j < vector_count`. Each value
is the coefficient of the single term `⟨v_i, v_j⟩`. Coefficients are integers or
`Fraction` objects. Encode the goal as `target >= 0`, and normalize hypotheses
to `expression == 0` or `expression >= 0`.

An equality or inequality witness is a pair `(multiplier, expression)`.
Equality multipliers may have either sign. An inequality multiplier must be
nonnegative. A square is a pair `(weight, coefficients)` representing a
nonnegative weight times the squared norm of the indicated linear combination
of vectors. The verifier expands these terms and compares coefficients with
the target. If it rejects a certificate, the underlying inequality is still
unresolved by that check.

## Positive parameters

The rational, Gram, polynomial certificate, and formal series certificate
checks accept `parameters=PositiveParameters((p, q, ...))`.
Declare every parameter as independently strictly positive. Encode a strict
ordering such as `sigma > rho > 0` by declaring `rho` and `gap`, then use
`sigma = rho + gap` when constructing the inputs. Use `parameters.rational`
to construct a coefficient from two exact `Poly` objects in the parameters.
For polynomials in additional variables, use `domain=parameters.field` and
keep the parameter symbols out of the generator list.

Signs are certified by the coefficient signs of the numerator and denominator
of the rational function. A denominator must have a certified strict sign.
These sufficient conditions can reject forms whose sign or nonvanishing is
unresolved. The [uniform recurrence example](examples/parameterized_series.py)
checks one certificate for all positive `rho`, `gap`, `c` and `C`.

## Polynomial certificates

With real variables and rational coefficients, encode the goal as `target >= 0`
and the hypotheses as polynomials equal to zero or greater than or equal to
zero. Supply a polynomial multiplier for each equality, a nonnegative rational
multiplier for each inequality, and a list of polynomial squares with
nonnegative rational weights. The verifier checks that the resulting expression
equals the target exactly. To multiply an inequality by a sum of squares,
provide one `(weight, square_factor, hypothesis)` entry in `inequality_squares`
per square. A rejected witness leaves the claim unresolved by this check.

## Linear arithmetic

An affine form `(constant, coefficient_0, ...)` is interpreted as nonnegative
in `inequalities` and positive in `strict_inequalities`. An equality can be
entered as two opposite nonstrict inequalities. Fourier–Motzkin elimination
decides feasibility over either the real or rational numbers. Implication of
an affine target being nonnegative is checked by adding its strict negation and
testing infeasibility. The number of rows can grow doubly exponentially with
the number of variables.

## Formal series certificates

Provide the target and hypotheses as `(numerator, denominator)` pairs of
polynomials, with the first generator `z` serving as the series variable.
The other generators stand for arbitrary formal series. Numerators are affine
in those generators, and each denominator depends only on `z` and has nonzero
constant term.

A multiplier or slack is a sum of terms `(weight, shift, ratios)`. Each term
represents `weight * z**shift / product(1 - r*z for r in ratios)`. Nonnegative
weights, shifts and ratios ensure nonnegative coefficients. The verifier
checks that the target equals the sum of the hypotheses times their
multipliers, plus the slack. The certificate applies coefficient by
coefficient and requires no analytic convergence.

Use `shift_series(F, initials)` to form `(F - F(0)) / z`. Supply an exact
initial coefficient for each formal series symbol. The returned expression
uses the same symbols to denote their tails. Apply `rebase_series` with the
same initials to express other terms in those coordinates while preserving
their coefficients.

## Exact matrix rank

Pass a rectangular sequence of rows to `matrix_rank`. The entries are exact
integers or `Fraction` objects. The calculation uses rational Gaussian
elimination. See [the runnable example](examples/basic.py) for checks across
all seven fragments.

## Definite integrals over scales

`quadrature_value(k, lower, upper)` gives an exact closed form for the
integral of $t^k$ with rational $k$. With `decay=c` it integrates
$t^k e^{-c/t}$ for integer $k\le -2$ and $c>0$. Pass an `int` or
`Fraction` for the exponent and for exact constant bounds and decay.
Use `0` as the lower limit at $0^+$ and `None` as the upper limit at
$+\infty$. All other bounds are strictly positive. Two finite positive
bounds define an oriented integral. For symbolic bounds or decay, pass
exact coefficients from `PositiveParameters` and supply that declaration
as `parameters=`.

Pass the claimed closed form as the fourth argument to `verify_quadrature`
with the same bounds and options. Its return value is a `CheckResult`
that can be used in a symbolic proof obligation.

```python
from fractions import Fraction
from decidable import verify_quadrature

result = verify_quadrature(-2, 0, None, 4, decay=Fraction(1, 4))
assert result.status == "verified"
```

For powers the checker uses the power or logarithm primitive and tests
convergence separately at each improper endpoint. For exponential decay,
it sets $u=c/t$ and $n=-k-2$. The primitive is
$c^{k+1}e^{-u}P_n(u)$, where the polynomial satisfies
$P_n-P_n'=u^n$ and $P_n(0)=n!$. The primitive tends to zero at $0^+$
because exponential decay dominates polynomial growth, and to $c^{k+1}n!$
at $+\infty$. Finite endpoint values use direct substitution.

The closed-form check compares exact rational expressions after treating
supported logarithms, exponentials and nonintegral rational powers as
formal atoms. It checks the domain of the supplied expression and applies
$\log(p^r)=r\log p$ for declared positive parameters $p$ and rational $r$.
A failed comparison is `inconclusive` because identities between
different atoms can require additional rules. For example, a claim in the
form $\log(b/a)$ can require a further logarithm identity to match
$\log b-\log a$. A divergent integral disproves a claim of a finite value.
Forms outside the supported input domain yield `unsupported`.

## Size calculus

`decidable.size_calculus` replays exact moment and stretched-exponential
size bounds. For a rational polynomial moment problem on an interval,
`verify_majorant` checks a signed intercept, nonnegative slopes, the total
budget and the pointwise inequality on the entire support. The checker
accepts an explicit square or uses exact univariate root isolation.
`verify_atomic_law` replays strict feasibility or a counterexample.

```python
from fractions import Fraction
from sympy import Poly, symbols
from decidable.size_calculus import (
    Majorant, MomentProblem, grid_optimum, verify_majorant,
)

z = symbols("z")
problem = MomentProblem((Poly(z**2, z),), (4,), Poly(z, z), 2, lower=1)
assert verify_majorant(problem, Majorant(1, (Fraction(1, 4),))).status == "verified"
assert grid_optimum(problem, (1, 3)).value == Fraction(7, 4)
```

`search_refutation` enumerates exact vertices on a small supplied grid.
`search_majorant` proposes rational coefficients from grid contacts and
tangencies, then replays them on the original support. An unresolved search
returns `inconclusive`. `verify_average_refutation` uses rational
enclosures to check a finite counterexample for a stretched-exponential
weighted average. `search_average_refutation` proposes such witnesses on
a finite grid and `bracket_average_scale` retains checked lower and upper
scales. `sharp_closed_tail` computes the sharp bound in the raw moment
chart $Z\ge1$, $\mathbb E Z\le2$.

`decide_moment` uses the optional system Z3 library to decide polynomial
moment implications on semialgebraic supports, including cases needing
algebraic atoms. `exponential_moment_problem` reduces common rational-rate
exponentials to that fragment. `check_exponential_residual` replays guarded
Taylor-coefficient certificates for the E1 chart.

`verify_derivation` replays named rules for exact rational reciprocal
exponents, including powers, products, exponent changes, tail conversions,
weighted sums and the bounded independent-mean rule. It retains exact
scale expressions, including logarithm constants. `verify_scale_domination`
checks allocations of exact or generic coefficient shares on small, large
and free atom ranges. `check_scaling` checks the necessary dilation weight
condition for positive expressions under supplied rational weights.

The [portable moment format](docs/size-calculus.md) shows how to run JSON
manifests with `python -m decidable.size_calculus` and describes the exact
scope of this implementation.

## Search and reporting

`decidable.search.search_gram_certificate` tries exact rational linear algebra
first. Install with `python -m pip install -e '.[search]'` to enable optional
CVXPY proposals. Every returned certificate passes the exact Gram verifier.
`None` means the search was inconclusive.

Use `decidable.report.Obligation` to name a check and label it `symbolic`,
`finite`, `numeric`, or `control`. Supply a scope for finite checks. A composite
can depend on earlier named obligations. `Report(...).run(path)` removes the
old output before evaluating, records failures and exceptions, and writes a
fresh report with counts. Each outcome has a structured `result`. A composite
with one stated dependency scope inherits that scope. When multiple scoped
dependencies are used, the scope labels must match. The report counts verified outcomes as `PASS` and
counts numerical outcomes separately as `EVIDENCE`. If its dependencies otherwise pass, a symbolic composite backed by numerical
evidence is also recorded as `EVIDENCE`.

## Specialized checks and reductions

| Module | Public operations | Guarantee |
| --- | --- | --- |
| `decidable.kernel_domination` | `check_kernel_domination` | Exact coefficientwise comparison of rational kernels. Pass `tail_start=None` to decide the supported simple positive rational pole case with an automatically chosen cutoff. |
| `decidable.positive_state` | `check_positive_state` | Checks whether the supplied weights are strictly positive and define a linear functional that contracts at the given rational rate for a nonnegative rational state matrix. |
| `decidable.positive_state` | `find_positive_state` | Decides whether such weights exist at the given rate and returns weights or exact Farkas multipliers proving infeasibility. |
| `decidable.linear` | `verify_farkas_certificate`, `find_farkas_certificate` | A witness verifies infeasibility of a closed rational affine system. |
| `decidable.linalg` | `matrix_nullspace`, `matrix_solve` | A rational basis of the nullspace and an exact solution of a consistent rectangular system. |
| `decidable.jets` | `check_jet_identity`, `total_derivative`, `two_scale_derivative` | A differential polynomial identity follows from the supplied algebraic jet relations in the registered dimension. |
| `decidable.reductions` | `expand_gram_inner_product`, `reduce_spectral_quadratic_form` | Bilinear expansion into Gram coordinates and scalar evaluation on an explicitly assumed finite spectral support. |
| `decidable.exp_series` | `check_exp_series`, `verify_exp_series_certificate` | An exact Taylor prefix and integer-tail certificate prove $P(x)e^x+Q(x) \ge 0$ on $x \ge 0$. Use `tail_start="auto"` to decide the Taylor coefficient condition. |
| `decidable.quadrature` | `quadrature_value`, `verify_quadrature` | Exact endpoint evaluation and closed-form verification for the supported definite integrals over scales. |
| `decidable.search` | `search_series_certificate`, `search_polynomial_certificate` | Search uses exact rational arithmetic over the full supplied finite basis by default and verifies any witness it finds. Set `max_support` to bound the number of nonzero weights. An unsuccessful search is inconclusive for the claim. |
| `decidable.realization` | `recurrence_to_series` | An exact generating function for a finite scalar recurrence with rational constant coefficients. |

The [harmonic excess example](examples/harmonic_excess.py) demonstrates a
kernel comparison over all modes. Its input is the nonnegative modal expansion
supplied by the mathematical argument. The
[rewrite design](docs/rewrite-design.md) describes the check boundaries and the
remaining research directions.
