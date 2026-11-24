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
caller supplies the relations that hold for the intended matrices.

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
| `decidable.search` | `search_series_certificate`, `search_polynomial_certificate` | Search uses exact rational arithmetic over the full supplied finite basis by default and verifies any witness it finds. Set `max_support` to bound the number of nonzero weights. An unsuccessful search is inconclusive for the claim. |
| `decidable.realization` | `recurrence_to_series` | An exact generating function for a finite scalar recurrence with rational constant coefficients. |

The [harmonic excess example](examples/harmonic_excess.py) demonstrates a
kernel comparison over all modes. Its input is the nonnegative modal expansion
supplied by the mathematical argument. The
[rewrite design](docs/rewrite-design.md) describes the check boundaries and the
remaining research directions.
