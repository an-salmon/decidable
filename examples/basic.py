"""Seven exact checks with explicit hypotheses and certificates."""

from fractions import Fraction

from sympy import Poly, Symbol

from decidable import (
    linear_implies,
    matrix_rank,
    noncommutative_identity,
    rational_identity,
    verify_gram_certificate,
    verify_polynomial_certificate,
    verify_series_certificate,
)


def main() -> int:
    x = Symbol("x")
    rational_holds = rational_identity(
        Poly(x**2 - 1, x), Poly(x - 1, x), Poly(x + 1, x), Poly(1, x)
    )

    # If P is idempotent, P²QP² = PQP for any Q.
    matrix_holds = noncommutative_identity(
        {("P", "P", "Q", "P", "P"): 1},
        {("P", "Q", "P"): 1},
        rules=[(("P", "P"), ("P",))],
    )

    # If ||u||² <= ||v||², then ||u+v||² <= 4||v||².
    # The difference is 2(||v||² - ||u||²) + ||u-v||².
    gram_valid = verify_gram_certificate(
        2,
        target={(0, 0): -1, (0, 1): -2, (1, 1): 3},
        inequalities=[(2, {(0, 0): -1, (1, 1): 1})],
        squares=[(1, (1, -1))],
    )

    # For x >= 1, x² - 1 = 2(x - 1) + (x - 1)² >= 0.
    polynomial_valid = verify_polynomial_certificate(
        Poly(x**2 - 1, x),
        inequalities=[(2, Poly(x - 1, x))],
        squares=[(1, Poly(x - 1, x))],
    )

    # From x >= 0 and 2x + y <= 3, infer y <= 3.
    linear_holds = linear_implies(
        2, (3, 0, -1), inequalities=[(0, 1, 0), (3, -2, -1)]
    )

    # If 1 - (1 - z/2) A(z) is coefficientwise nonnegative,
    # multiply it by 1/(1 - z/2) to bound A(z) by a geometric series.
    z, series = Symbol("z"), Symbol("A")
    kernel_denominator = Poly(1 - z / 2, z, series)
    residual = Poly(1 - (1 - z / 2) * series, z, series)
    series_valid = verify_series_certificate(
        (residual, kernel_denominator),
        hypotheses=[([(1, 0, (Fraction(1, 2),))], (residual, Poly(1, z, series)))],
    )

    checks = {
        "rational · Rational function identity": rational_holds,
        "noncommutative · Noncommutative identity": matrix_holds,
        "gram · Gram certificate": gram_valid,
        "polynomial · Polynomial certificate": polynomial_valid,
        "linear · Linear implication": linear_holds,
        "series · Formal series certificate": series_valid,
        "linalg · Exact matrix rank": matrix_rank([[1, 2, 3], [2, 4, 6], [0, 1, 1]]) == 2,
    }
    for label, result in checks.items():
        print(f"{label}: {result}")
    return 0 if all(result is True for result in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
