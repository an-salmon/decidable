"""Exact rank of finite rational matrices."""

from collections.abc import Sequence
from fractions import Fraction
from typing import Union


RationalMatrix = Sequence[Sequence[Union[int, Fraction]]]


def matrix_rank(matrix: RationalMatrix) -> int:
    """Return the rank of a rectangular matrix over the rationals.

    Supply a sequence of rows with equal lengths, each containing ``int``
    or ``Fraction`` entries. Empty matrices and matrices with zero columns
    have rank zero. Gaussian elimination leaves the input unchanged.
    """
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes, bytearray)):
        raise TypeError("a matrix must be a sequence of rows")

    rows: list[list[Fraction]] = []
    column_count = 0
    for row in matrix:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError("a matrix row must be a sequence")
        if not rows:
            column_count = len(row)
        elif len(row) != column_count:
            raise ValueError("matrix rows must all have the same length")
        if any(type(entry) not in (int, Fraction) for entry in row):
            raise TypeError("matrix entries must be int or Fraction")
        rows.append([Fraction(entry) for entry in row])

    rank = 0
    for column in range(column_count):
        pivot_index = next(
            (index for index in range(rank, len(rows)) if rows[index][column]), None
        )
        if pivot_index is None:
            continue
        rows[rank], rows[pivot_index] = rows[pivot_index], rows[rank]
        pivot_row = rows[rank]
        for index in range(rank + 1, len(rows)):
            row = rows[index]
            if row[column] == 0:
                continue
            factor = row[column] / pivot_row[column]
            row[column] = Fraction(0)
            for trailing_column in range(column + 1, column_count):
                row[trailing_column] -= factor * pivot_row[trailing_column]

        rank += 1
        if rank == len(rows):
            break

    return rank
