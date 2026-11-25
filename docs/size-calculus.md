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

The portable checker covers univariate polynomial moments on closed
bounded intervals and right half-lines. Typed size derivations,
stretched-exponential average witnesses and dilation checks use the Python
API. The interval searches are reference algorithms for small problems.
The broader semialgebraic decision procedure, symbolic exponent domains,
arbitrary measurable aggregation, sharp-scale search and retrospective
corpus are further parts of the draft.
