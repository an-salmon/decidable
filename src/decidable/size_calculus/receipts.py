"""Portable replay for the version-1 polynomial moment subset."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import json
import re

from sympy import And, Eq, Le, Lt, Not, Or, Poly, QQ, Symbol

from .moments import (
    AtomicLaw, Majorant, MomentProblem, SquareProof, _InvalidMomentInput,
    verify_atomic_law, verify_majorant,
)
from .semialgebraic import AlgebraicLaw, SemialgebraicMomentProblem, decide_moment


_VERDICTS = ("CERTIFIED", "REFUTED", "UNSUPPORTED", "FAIL")
_SCALAR = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")


class _Invalid(ValueError):
    """The submission violates the portable input schema."""


class _Unsupported(ValueError):
    """The submission requests an unimplemented portable fragment."""


def _name(value) -> bool:
    return type(value) is str and bool(value.strip())


def _object(value, required, subject, *, optional=()):
    if type(value) is not dict or not set(required) <= value.keys():
        raise _Invalid(f"missing or malformed {subject} fields")
    if value.keys() - set(required) - set(optional):
        raise _Invalid(f"unexpected {subject} fields")


def _scalar(value) -> Fraction:
    if type(value) is not str or not _SCALAR.fullmatch(value):
        raise _Invalid("use a canonical integer or reduced fraction string")
    try:
        result = Fraction(value)
    except ValueError as error:
        raise _Invalid("invalid exact scalar") from error
    if str(result) != value:
        raise _Invalid("use a canonical integer or reduced fraction string")
    return result


def _scalar_list(value) -> tuple[Fraction, ...]:
    if type(value) is not list:
        raise _Invalid("expected an array of exact scalars")
    return tuple(map(_scalar, value))


def _portable_copy(value, depth=0):
    if depth > 120:
        raise _Invalid("portable JSON nesting exceeds 120")
    if type(value) is float:
        raise _Invalid("a float literal is not exact")
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is list:
        return [_portable_copy(item, depth + 1) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _portable_copy(item, depth + 1) for key, item in value.items()}
    raise _Invalid("expected a portable JSON value")


def _validate_tree(tree, coordinates, *, predicate=False, depth=0):
    if depth > 100:
        raise _Invalid("expression nesting exceeds 100")
    if not predicate and type(tree) is str:
        _scalar(tree)
        return
    if type(tree) is not list or not tree or type(tree[0]) is not str:
        raise _Invalid("invalid tagged expression")
    tag, *args = tree

    def children(*, predicates=False):
        for arg in args:
            _validate_tree(arg, coordinates, predicate=predicates, depth=depth + 1)

    if predicate:
        if (tag in ("and", "or") and args) or (tag == "not" and len(args) == 1):
            children(predicates=True)
        elif tag in ("eq", "lt", "le") and len(args) == 2:
            children()
        else:
            raise _Invalid(f"invalid predicate tag or operands for {tag!r}")
    elif tag == "ref" and len(args) == 1 and type(args[0]) is str:
        if args[0] not in coordinates:
            raise _Invalid(f"undeclared coordinate {args[0]!r}")
    elif tag == "pow" and len(args) == 2:
        _scalar(args[1])
        _validate_tree(args[0], coordinates, depth=depth + 1)
    elif ((tag in ("add", "mul") and args) or (tag == "div" and len(args) == 2)
          or (tag in ("exp", "log", "pos") and len(args) == 1)):
        children()
    elif tag == "ind" and len(args) == 1:
        children(predicates=True)
    else:
        raise _Invalid(f"invalid scalar tag or operands for {tag!r}")


def _polynomial(tree, coordinates):
    generators = tuple(coordinates.values())
    if type(tree) is str:
        return Poly(_scalar(tree), *generators, domain=QQ)
    tag, *args = tree
    if tag == "ref":
        return Poly(coordinates[args[0]], *generators, domain=QQ)
    if tag in ("add", "mul"):
        polynomial = Poly(0 if tag == "add" else 1, *generators, domain=QQ)
        for arg in args:
            part = _polynomial(arg, coordinates)
            polynomial = polynomial + part if tag == "add" else polynomial * part
        return polynomial
    if tag == "pow":
        exponent = _scalar(args[1])
        if exponent.denominator != 1 or exponent < 0:
            raise _Unsupported("polynomial powers require nonnegative integers")
        return _polynomial(args[0], coordinates)**int(exponent)
    raise _Unsupported(f"expression tag {tag!r} is outside the polynomial fragment")


def _polynomial_predicate(tree, coordinates):
    tag, *args = tree
    comparisons = {"eq": Eq, "lt": Lt, "le": Le}
    if tag in comparisons:
        left, right = (_polynomial(arg, coordinates).as_expr() for arg in args)
        return comparisons[tag](left, right, evaluate=False)
    connectives = {"and": And, "or": Or, "not": Not}
    return connectives[tag](*(_polynomial_predicate(arg, coordinates) for arg in args))


def _problem(record, *, decide=False) -> MomentProblem | SemialgebraicMomentProblem:
    if type(record) is not dict or type(record.get("version")) is not int:
        raise _Invalid("the problem requires an integer version")
    if record["version"] != 1:
        raise _Unsupported("only problem version 1 is supported")
    _object(record, ("version", "coordinates", "support", "inputs", "output"),
            "version-1 polynomial moment problem", optional=("id",))
    if "id" in record and not _name(record["id"]):
        raise _Invalid("problem id must be a nonempty string")
    coordinates = record["coordinates"]
    if (type(coordinates) is not list or not all(map(_name, coordinates))
            or len(set(coordinates)) != len(coordinates)):
        raise _Invalid("coordinate declarations must be distinct nonempty strings")

    inputs = record["inputs"]
    output = record["output"]
    if type(inputs) is not list:
        raise _Invalid("input moments must be an array")
    support = record["support"]
    _validate_tree(support, coordinates, predicate=True)
    for item in [*inputs, output]:
        _object(item, ("integrand", "budget"), "moment")
        _scalar(item["budget"])
        _validate_tree(item["integrand"], coordinates)

    if not coordinates:
        raise _Unsupported("declare at least one polynomial coordinate")
    symbols = {name: Symbol(name) for name in coordinates}
    if decide:
        budgets = tuple(_scalar(item["budget"]) for item in inputs)
        output_budget = _scalar(output["budget"])
        if output_budget < 0 or any(budget < 0 for budget in budgets):
            raise _Invalid("budgets must be nonnegative")
        return SemialgebraicMomentProblem(
            tuple(symbols.values()), _polynomial_predicate(support, symbols),
            tuple(_polynomial(item["integrand"], symbols) for item in inputs), budgets,
            _polynomial(output["integrand"], symbols), output_budget,
        )
    if len(symbols) != 1:
        raise _Unsupported("use a single polynomial coordinate")
    name, = symbols
    reference = ["ref", name]
    upper = None
    if support[0] == "and" and len(support) == 3:
        support, upper_clause = support[1:]
        if (len(upper_clause) != 3 or upper_clause[:2] != ["le", reference]
                or type(upper_clause[2]) is not str):
            raise _Unsupported("use a closed rational upper support bound")
        upper = _scalar(upper_clause[2])
    if (len(support) != 3 or support[0] != "le" or support[2] != reference
            or type(support[1]) is not str):
        raise _Unsupported("use a closed rational lower support bound")
    return MomentProblem(
        tuple(_polynomial(item["integrand"], symbols) for item in inputs),
        tuple(_scalar(item["budget"]) for item in inputs),
        _polynomial(output["integrand"], symbols), _scalar(output["budget"]),
        lower=_scalar(support[1]), upper=upper,
    )


def _majorant(certificate, problem):
    _object(certificate, ("kind", "c", "lambda"), "majorant", optional=("pointwise",))
    majorant = Majorant(_scalar(certificate["c"]), _scalar_list(certificate["lambda"]))
    square = None
    if "pointwise" in certificate:
        proof = certificate["pointwise"]
        if type(proof) is not dict or not _name(proof.get("method")):
            raise _Invalid("pointwise proof requires a method")
        if proof["method"] != "square":
            raise _Unsupported("only explicit polynomial square certificates are supported")
        _object(proof, ("method", "coefficient", "polynomial"), "square proof")
        polynomial_record = proof["polynomial"]
        _object(polynomial_record, ("variable", "coeffs"), "square polynomial")
        if polynomial_record["variable"] != problem.output.gen.name:
            raise _Invalid("square certificate must use the problem coordinate")
        coefficients = _scalar_list(polynomial_record["coeffs"])
        if not coefficients:
            raise _Invalid("square polynomial needs a coefficient array")
        polynomial = Poly.from_list(
            list(reversed(coefficients)), problem.output.gen, domain=QQ,
        )
        square = SquareProof(polynomial, _scalar(proof["coefficient"]))
    return verify_majorant(problem, majorant, square=square)


def _law(certificate, problem, *, strict_feasibility):
    _object(certificate, ("kind", "points", "masses"), "atomic law")
    raw_points = certificate["points"]
    if type(raw_points) is not list or any(
        type(point) is not list or len(point) != 1 for point in raw_points
    ):
        raise _Invalid("each atom needs one coordinate")
    points = tuple(_scalar(value) for (value,) in raw_points)
    law = AtomicLaw(points, _scalar_list(certificate["masses"]))
    return verify_atomic_law(problem, law, strict_feasibility=strict_feasibility)


def _replay(entry):
    identifier = entry["id"]
    receipt = {
        "id": identifier, "checker": "decidable.size_calculus", "checker_version": 1,
        "problem_id": identifier, "assumptions_used": [],
    }
    try:
        entry = _portable_copy(entry)
        _object(entry, ("id", "problem", "certificate"), "manifest entry", optional=("expect",))
        problem_record = entry["problem"]
        if type(problem_record) is dict:
            if type(problem_record.get("version")) is int:
                receipt["problem_version"] = problem_record["version"]
            if _name(problem_record.get("id")):
                receipt["problem_id"] = problem_record["id"]
        receipt["problem_sha256"] = sha256(json.dumps(
            problem_record, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()

        certificate = entry["certificate"]
        if type(certificate) is not dict or not _name(certificate.get("kind")):
            raise _Invalid("certificate requires a kind")
        kind = certificate["kind"]
        strict_feasibility = kind == "strict_feasibility"
        if kind in ("majorant", "atoms", "strict_feasibility", "decide"):
            receipt["claim"] = "strict_feasibility" if strict_feasibility else "moment_bound"
        if kind == "decide":
            _object(certificate, ("kind",), "decision")
            result = decide_moment(_problem(problem_record, decide=True))
            receipt.update(mode="symbolic", scope=result.scope)
            if result.status in ("unsupported", "inconclusive"):
                raise _Unsupported(result.reason)
            if result.status == "disproved":
                if not isinstance(result.witness, AlgebraicLaw):
                    raise RuntimeError("exact moment decision returned no algebraic law")
                receipt.update(
                    mode="finite", scope="Exact algebraic law on the declared support",
                    problem_scope=result.scope, witness=asdict(result.witness),
                )
        elif kind == "majorant":
            problem = _problem(problem_record)
            result = _majorant(certificate, problem)
            receipt.update(mode="symbolic", scope=result.scope)
        elif kind in ("atoms", "strict_feasibility"):
            problem = _problem(problem_record)
            result = _law(certificate, problem, strict_feasibility=strict_feasibility)
            receipt.update(
                mode="finite", scope="Supplied rational atomic law on the declared support",
                problem_scope=result.scope, witness_scope=certificate["points"],
            )
        else:
            raise _Unsupported(f"unsupported certificate kind {kind!r}")

        if result.status == "verified" and kind != "atoms":
            verdict = "CERTIFIED"
        elif result.status == "disproved" and kind in ("atoms", "decide"):
            verdict = "REFUTED"
        elif result.status == "inconclusive":
            verdict = "FAIL"
        elif result.status == "unsupported":
            raise _Unsupported(result.reason)
        else:
            raise RuntimeError(f"unexpected {kind} checker status {result.status!r}")
    except _Unsupported as error:
        return {**receipt, "verdict": "UNSUPPORTED", "mode": "unsupported", "reason": str(error)}
    except (_Invalid, _InvalidMomentInput) as error:
        return {**receipt, "verdict": "FAIL", "mode": "bookkeeping", "reason": str(error)}
    except Exception as error:
        return {
            **receipt, "verdict": "FAIL", "mode": "bookkeeping", "checker_error": True,
            "reason": f"{type(error).__name__}: {error}",
        }

    return {**receipt, "verdict": verdict, "reason": result.reason, "certificate": certificate}


def _ordered_entries(manifest):
    if type(manifest) is dict:
        if set(manifest) != {"expected_ids", "entries"}:
            raise ValueError("a wrapped manifest requires expected_ids and entries")
        expected_ids, entries = manifest["expected_ids"], manifest["entries"]
        if (type(expected_ids) is not list or not expected_ids
                or not all(map(_name, expected_ids)) or len(set(expected_ids)) != len(expected_ids)):
            raise ValueError("expected_ids must contain unique nonempty identifiers")
    else:
        expected_ids, entries = None, manifest
    if type(entries) is not list or not entries:
        raise ValueError("a receipt manifest must contain a nonempty entry list")

    by_id = {}
    for entry in entries:
        if type(entry) is not dict or not _name(entry.get("id")):
            raise ValueError("each manifest entry needs a nonempty string id")
        identifier = entry["id"]
        if identifier in by_id:
            raise ValueError("manifest identifiers must be unique")
        if "expect" in entry and entry["expect"] not in _VERDICTS:
            raise ValueError("invalid expected control verdict")
        by_id[identifier] = entry
    if expected_ids is None:
        return entries
    missing = set(expected_ids) - by_id.keys()
    extra = by_id.keys() - set(expected_ids)
    if missing or extra:
        raise ValueError(f"unexpected manifest membership (missing {sorted(missing)}, extra {sorted(extra)})")
    return [by_id[identifier] for identifier in expected_ids]


def replay_manifest(manifest):
    """Replay entries in input order or in a wrapped manifest's expected_ids order."""
    receipts = []
    for entry in _ordered_entries(manifest):
        receipt = _replay(entry)
        if "expect" in entry:
            expected = entry["expect"]
            passed = receipt["verdict"] == expected and not receipt.get("checker_error", False)
            receipt.update(mode="control", evidence_mode=receipt["mode"], expected=expected,
                           passed=passed)
        receipts.append(receipt)
    return receipts
