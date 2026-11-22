"""Certify a sequence bound uniformly over four positive parameters."""

from sympy import Poly, symbols

from decidable import PositiveParameters, verify_series_certificate


def main() -> int:
    rho, gap, c, C = symbols("rho gap c C")
    parameters = PositiveParameters((rho, gap, c, C))
    sigma = rho + gap
    z, A = symbols("z A")

    # A represents the sequence a. Encode a_0 <= c and
    # a_{n+1} <= rho*a_n + C*sigma**n as one coefficientwise hypothesis.
    numerator = c * (1 - sigma * z) + C * z - (1 - rho * z) * (1 - sigma * z) * A
    residual = Poly(numerator, z, A, domain=parameters.field)
    hypothesis_denominator = Poly(1 - sigma * z, z, A, domain=parameters.field)
    target_denominator = Poly((1 - rho * z) * (1 - sigma * z), z, A, domain=parameters.field)
    ratio = parameters.rational(Poly(rho, *parameters.symbols), Poly(1, *parameters.symbols))

    verified = verify_series_certificate(
        (residual, target_denominator),
        hypotheses=[([(1, 0, (ratio,))], (residual, hypothesis_denominator))],
        parameters=parameters,
    )
    print(f"Uniform recurrence certificate: {verified}")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
