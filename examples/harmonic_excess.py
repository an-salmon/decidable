"""Check the modal comparison for harmonic excess at half the radius."""

from sympy import Poly, Symbol

from decidable import dispatch
from decidable.fragments import KernelDominationClaim


def main() -> int:
    # Given nonnegative weights for modes j > k, normalize the energy at the
    # larger radius and write j = k + 1 + n. Halving the radius multiplies
    # the j-th squared mode by 4^(-j). After taking out 4^(-k-1), the
    # remaining modal weights are 4^(-n) versus 1, for every n >= 0.
    z = Symbol("z")
    numerator = Poly(1, z)
    upper = (numerator, Poly(1 - z, z))
    lower = (numerator, Poly(1 - z / 4, z))
    claim = KernelDominationClaim(upper, lower, tail_start=None)
    result = dispatch("series", "check_kernel_domination", claim)
    print(f"Harmonic excess comparison: {result.status}")
    print(result.reason)
    return 0 if result.status == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
