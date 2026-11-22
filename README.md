# decideable

`decideable` checks finite algebraic statements that arise inside mathematical
proofs. To use it, write down the hypotheses and conclusion of a proof step,
then choose the check that matches their mathematical form. A successful check
establishes the encoded statement. The derivation of that statement from the
original proof remains part of the mathematical argument.

| Check | What it establishes |
| --- | --- |
| `rational_identity` | Whether two rational functions agree in a rational function field. |
| `verify_gram_certificate` | Whether a supplied certificate proves a linear inequality in inner products. |

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

See [the runnable example](examples/basic.py) for both checks.
