"""Exact linear algebra of finite rational matrices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Union


RationalVector = Sequence[Union[int, Fraction]]
RationalMatrix = Sequence[RationalVector]
ExactVector = tuple[Fraction, ...]


@dataclass(frozen=True)
class LinearSolution:
    """A particular solution and a basis for the homogeneous solutions."""

    particular: ExactVector
    nullspace: tuple[ExactVector, ...]


def _exact_vector(vector: RationalVector, length: int) -> ExactVector:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        raise TypeError("a vector must be a sequence")
    if len(vector) != length:
        raise ValueError("vector length must match the required dimension")
    if any(type(entry) not in (int, Fraction) for entry in vector):
        raise TypeError("vector entries must be int or Fraction")
    return tuple(Fraction(entry) for entry in vector)


def _exact_matrix(matrix: RationalMatrix) -> tuple[list[list[Fraction]], int]:
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

    return rows, column_count


def matrix_rank(matrix: RationalMatrix) -> int:
    """Return the rank of a rectangular matrix over the rationals.

    Supply a sequence of rows with equal lengths, each containing ``int``
    or ``Fraction`` entries. Empty matrices and matrices with zero columns
    have rank zero. Gaussian elimination leaves the input unchanged.
    """
    rows, column_count = _exact_matrix(matrix)
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


def _rref(rows: list[list[Fraction]], column_count: int) -> list[int]:
    """Reduce all entries in place, using pivots only in coefficient columns."""
    pivots: list[int] = []
    for column in range(column_count):
        rank = len(pivots)
        pivot_index = next(
            (index for index in range(rank, len(rows)) if rows[index][column]), None
        )
        if pivot_index is None:
            continue
        rows[rank], rows[pivot_index] = rows[pivot_index], rows[rank]
        pivot = rows[rank][column]
        pivot_row = [entry / pivot for entry in rows[rank]]
        rows[rank] = pivot_row
        for index, row in enumerate(rows):
            if index != rank and row[column]:
                factor = row[column]
                rows[index] = [
                    entry - factor * pivot_entry
                    for entry, pivot_entry in zip(row, pivot_row)
                ]
        pivots.append(column)
        if len(pivots) == len(rows):
            break
    return pivots


def _nullspace_basis(
    rows: list[list[Fraction]], pivots: list[int], column_count: int
) -> tuple[ExactVector, ...]:
    pivot_columns = set(pivots)
    basis = []
    for free_column in range(column_count):
        if free_column in pivot_columns:
            continue
        vector = [Fraction(0)] * column_count
        vector[free_column] = Fraction(1)
        for row, pivot in zip(rows, pivots):
            vector[pivot] = -row[free_column]
        basis.append(tuple(vector))
    return tuple(basis)


def matrix_nullspace(matrix: RationalMatrix) -> tuple[ExactVector, ...]:
    """Return a rational nullspace basis for a rectangular matrix.

    Basis vectors correspond to free columns in ascending order. An empty
    row sequence represents a 0-by-0 matrix. A trivial kernel gives ().
    """
    rows, column_count = _exact_matrix(matrix)
    pivots = _rref(rows, column_count)
    return _nullspace_basis(rows, pivots, column_count)


def matrix_solve(matrix: RationalMatrix, rhs: RationalVector) -> LinearSolution | None:
    """Solve A*x = rhs over the rationals for any rectangular matrix.

    Return None for an inconsistent system. Otherwise the solution set is
    the particular solution plus the span of the nullspace basis. Free
    variables in the particular solution are zero. An empty row sequence
    represents a 0-by-0 matrix, whose unique solution is the empty vector.
    """
    rows, column_count = _exact_matrix(matrix)
    right_hand_side = _exact_vector(rhs, len(rows))
    for row, value in zip(rows, right_hand_side):
        row.append(value)
    pivots = _rref(rows, column_count)
    if any(row[-1] for row in rows[len(pivots):]):
        return None
    particular = [Fraction(0)] * column_count
    for row, pivot in zip(rows, pivots):
        particular[pivot] = row[-1]
    return LinearSolution(
        particular=tuple(particular),
        nullspace=_nullspace_basis(rows, pivots, column_count),
    )
