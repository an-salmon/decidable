# Size calculus

The moment replay accepts a finite list of entries, each with a unique
identifier, a versioned problem and a supplied certificate.

```json
[
  {
    "id": "square",
    "problem": {
      "version": 1,
      "coordinates": ["z"],
      "support": ["le", "1", ["ref", "z"]],
      "inputs": [{"integrand": ["pow", ["ref", "z"], "2"], "budget": "4"}],
      "output": {"integrand": ["ref", "z"], "budget": "2"}
    },
    "certificate": {
      "kind": "majorant",
      "c": "1",
      "lambda": ["1/4"],
      "pointwise": {
        "method": "square",
        "coefficient": "1/4",
        "polynomial": {"variable": "z", "coeffs": ["-2", "1"]}
      }
    }
  }
]
```

Run it from an installed checkout with
`python -m decidable.size_calculus manifest.json --output receipts.json`.
Scalar literals are canonical integers or reduced fractions written as
strings. Expressions use the declared coordinate and the `add`, `mul`
and nonnegative integer `pow` tags. A closed upper bound may be included
as `["and", lower_clause, ["le", ["ref", "z"], "upper"]]`, using the bound
in place of `"upper"`.

Omit the `pointwise` field to use root isolation. For an atomic
counterexample, use `{"kind":"atoms","points":[["2"]],"masses":["1"]}`.
The same record with `"kind":"strict_feasibility"` requires strict input
inequalities. Receipts bind the problem to its SHA-256 digest. A matching
`expect` field marks an entry as a control. A passing control carries its
actual verdict and a separate `passed` flag.

To fix a run's expected identifiers, wrap the list as
`{"expected_ids":["square"],"entries":[...]}`. The runner validates
membership and uses the declared order. Checker errors yield failed
receipts even under a control expecting a rejected submission.

## Exact decisions

Use `{"kind":"decide"}` to decide a moment implication. This route
accepts multivariate rational polynomials with Boolean combinations of
polynomial support comparisons. A version-1 problem may encode an algebraic
graph with extra coordinates and isolating inequalities. The optional system
Z3 library supplies the exact quantifier-free nonlinear real arithmetic
procedure. A timed-out query gives no verdict.

With $m$ input budgets, the query searches for an output violation
supported on at most $m+1$ atoms. Given any violating law, truncate the
output integrand at a sufficiently high finite level to preserve strict
excess. Finite moment representation preserves that truncated moment and
the input moments. Optimizing the actual output on the resulting finite
support gives a vertex with at most $m+1$ positive masses. This reduction
also works at a boundary budget, where a rational grid can miss every
feasible law. A solver refutation reports exact algebraic coordinate and
mass assignments. The optional backend is part of the trusted decision
procedure for this route.

The Python API additionally exposes guarded E1 certificates, the
common exponential chart for rational rates, typed size derivations,
scale-share proofs and dilation checks. Conservative rational grids can
find average-rule refutations and refine a sharp-scale bracket. Its upper
endpoint comes from the aggregation theorem, so the bracket may remain
loose even after the grid search has settled.

## Size and scale rules

`verify_derivation` replays the fixed-exponent rules. Integral
aggregation requires a typed premise for the measurable family, its
uniform bound and its total scale. The independent mean and Bernstein
rules require an independence declaration for the same summands as the
bounded-mean premises. These are caller hypotheses.

The `parametric_*` operations use `AffineDomain` and preserve strict
guards. Aggregation returns a pair of disjoint branches when the domain
crosses reciprocal exponent one. Polynomial size functions have a
separate Python API for pointwise transfer, class comparison and convex
aggregation. These two APIs use explicit typed operations.

For scale comparison, `rewrite_scale` applies a supplied monomial
equality, and `verify_scale_domination` checks a supplied partition into
coefficient shares. Generic claims state which deterministic parameters
the constant may depend on, and exact claims spend only their fixed
coefficient budgets. Dilation checks concern weight compatibility.

The portable schema covers moment problems and receipts. Typed
derivations and scale proofs use the Python API. General nonlinear
symbolic exponent domains and transcendental combinations beyond E1 and
the common exponential chart remain outside those checkers. The draft's
retrospective counts require the original case-level transcriptions.
