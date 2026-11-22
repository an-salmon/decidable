# decideable

`decideable` checks finite algebraic statements that arise inside mathematical
proofs. To use it, write down the hypotheses and conclusion of a proof step,
then choose the check that matches their mathematical form. A successful check
establishes the encoded statement. The derivation of that statement from the
original proof remains part of the mathematical argument.

| Level | Check | What it establishes |
| --- | --- | --- |
| L1 | `rational_identity` | Equality of rational functions over the rationals. |
| L2 | `noncommutative_identity` | Equality of expressions in matrix words modulo supplied relations. |
| L3 | `verify_gram_certificate` | Validity of a supplied certificate for a linear inequality in inner products. |
| L4 | `verify_polynomial_certificate` | Validity of a supplied certificate for a real polynomial inequality. |
| L5 | `linear_feasible`, `linear_implies` | Feasibility and implication for finite systems of affine inequalities. |
| L6 | `verify_series_certificate` | Validity of a supplied certificate for a coefficientwise inequality of formal series. |

## Run the examples

With Python 3.10 or newer, run these commands from the repository root.

```sh
python -m pip install -e .
python examples/basic.py
python -m unittest discover -s tests
```

## Rational function identities

Pass the numerator and denominator of each expression as separate SymPy
`Poly` objects over the integers or rationals, with the same ordered
generators. The checker compares the cross-products in exact arithmetic and
rejects identically zero denominators. To use a verified identity at a
particular point, also establish that both original denominators are nonzero
there.

## Noncommutative identities

Represent an expression as a dictionary from words to rational coefficients.
A word is a tuple of matrix symbol names, and `()` is the identity. Each rule
`(left_word, right_word)` replaces the longer word with the shorter one. The
caller supplies the relations that hold for the intended matrices.

Matching reduced expressions establish the identity. If the expressions differ
and the rules are proved confluent, `False` means the identity fails in the
formal algebra with those relations. When confluence cannot be established,
different reductions give `None`.

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

## Polynomial certificates

With real variables and rational coefficients, encode the goal as `target >= 0`
and the hypotheses as polynomials equal to zero or greater than or equal to
zero. Supply a polynomial multiplier for each equality, a nonnegative rational
multiplier for each inequality, and a list of polynomial squares with
nonnegative rational weights. The verifier checks that the resulting expression
equals the target exactly. This certificate form covers some polynomial
inequalities. A rejected witness leaves the claim unresolved by this check.

## Linear arithmetic

An affine form `(constant, coefficient_0, ...)` is interpreted as nonnegative
in `inequalities` and positive in `strict_inequalities`. An equality can be
entered as two opposite nonstrict inequalities. Fourier–Motzkin elimination
decides feasibility over either the real or rational numbers. Implication of
an affine target being nonnegative is checked by adding its strict negation and
testing infeasibility.

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

See [the runnable example](examples/basic.py) for a check at every level.
