"""Two exact checks with explicit inputs and a human-readable certificate."""

from sympy import Poly, Symbol

from decideable import rational_identity, verify_gram_certificate


def main() -> int:
    x = Symbol("x")
    identity_holds = rational_identity(
        Poly(x**2 - 1, x), Poly(x - 1, x), Poly(x + 1, x), Poly(1, x)
    )

    # If ||u||² <= ||v||², then ||u+v||² <= 4||v||².
    # The difference is 2(||v||² - ||u||²) + ||u-v||².
    certificate_valid = verify_gram_certificate(
        2,
        target={(0, 0): -1, (0, 1): -2, (1, 1): 3},
        inequalities=[(2, {(0, 0): -1, (1, 1): 1})],
        squares=[(1, (1, -1))],
    )

    print(f"Rational function identity holds: {identity_holds}")
    print(f"Gram certificate verified: {certificate_valid}")
    return 0 if identity_holds and certificate_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
