"""Portable replay for the version-1 polynomial moment subset."""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json

from sympy import Poly, QQ, Symbol

from .moments import AtomicLaw, Majorant, MomentProblem, SquareProof, verify_atomic_law, verify_majorant


_VERDICTS = ("CERTIFIED", "REFUTED", "UNSUPPORTED", "FAIL")


class _Unsupported(ValueError):
    pass


def _scalar(value) -> Fraction:
    if type(value) is not str:
        raise TypeError("exact scalar literals must be strings")
    result = Fraction(value)
    if str(result) != value:
        raise ValueError("use a canonical integer or reduced fraction")
    return result


def _scalar_list(value) -> tuple[Fraction, ...]:
    if type(value) is not list:
        raise TypeError("expected an array of exact scalars")
    return tuple(map(_scalar, value))


def _polynomial(tree, coordinate, depth=0):
    if depth > 100:
        raise ValueError("expression nesting exceeds 100")
    if type(tree) is str:
        return Poly(_scalar(tree), coordinate, domain=QQ)
    if type(tree) is not list or not tree or type(tree[0]) is not str:
        raise ValueError("invalid tagged expression")
    tag, *args = tree
    if tag == "ref" and args == [coordinate.name]:
        return Poly(coordinate, coordinate, domain=QQ)
    if tag in ("add", "mul") and args:
        polynomial = Poly(0 if tag == "add" else 1, coordinate, domain=QQ)
        for arg in args:
            part = _polynomial(arg, coordinate, depth + 1)
            polynomial = polynomial + part if tag == "add" else polynomial * part
        return polynomial
    if tag == "pow" and len(args) == 2:
        exponent = _scalar(args[1])
        if exponent.denominator != 1 or exponent < 0:
            raise _Unsupported("polynomial powers require nonnegative integers")
        return _polynomial(args[0], coordinate, depth + 1)**int(exponent)
    if tag == "ref" or tag in ("pow", "add", "mul"):
        raise ValueError("invalid polynomial expression operands")
    raise _Unsupported(f"expression tag {tag!r} is outside the polynomial fragment")


def _problem(record) -> MomentProblem:
    if type(record) is not dict or set(record) != {
        "version", "coordinates", "support", "inputs", "output"
    }:
        raise _Unsupported("use the version-1 polynomial moment problem fields")
    if type(record["version"]) is not int or record["version"] != 1:
        raise _Unsupported("only problem version 1 is supported")
    coordinates = record["coordinates"]
    if type(coordinates) is not list or len(coordinates) != 1:
        raise _Unsupported("use a single polynomial coordinate")
    name, = coordinates
    if type(name) is not str or not name:
        raise ValueError("coordinate name must be a nonempty string")
    coordinate = Symbol(name)
    reference = ["ref", name]
    support = record["support"]
    upper = None
    if type(support) is list and len(support) == 3 and support[0] == "and":
        support, upper_clause = support[1:]
        if (type(upper_clause) is not list or len(upper_clause) != 3
                or upper_clause[:2] != ["le", reference]):
            raise _Unsupported("use a closed upper support bound")
        upper = _scalar(upper_clause[2])
    if type(support) is not list or len(support) != 3             or support[0] != "le" or support[2] != reference:
        raise _Unsupported("use a closed lower support bound")
    inputs = record["inputs"]
    output = record["output"]
    if type(inputs) is not list or any(
        type(item) is not dict or set(item) != {"integrand", "budget"}
        for item in [*inputs, output]
    ):
        raise ValueError("each moment needs an integrand and a budget")
    return MomentProblem(
        tuple(_polynomial(item["integrand"], coordinate) for item in inputs),
        tuple(_scalar(item["budget"]) for item in inputs),
        _polynomial(output["integrand"], coordinate), _scalar(output["budget"]),
        lower=_scalar(support[1]), upper=upper,
    )


def _has_float(value) -> bool:
    if type(value) is float:
        return True
    if isinstance(value, dict):
        return any(_has_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_float(item) for item in value)
    return False


def _replay(entry):
    identifier = entry["id"]
    problem_record = entry["problem"]
    digest = sha256(json.dumps(problem_record, sort_keys=True, separators=(",", ":"))
                    .encode("utf-8")).hexdigest()
    receipt = {
        "id": identifier, "checker": "decidable.size_calculus", "checker_version": 1,
        "problem_sha256": digest, "mode": "symbolic",
    }
    try:
        if _has_float(entry):
            raise ValueError("a float literal is not exact")
        if set(entry) - {"id", "problem", "certificate", "expect"}:
            raise _Unsupported("unrecognized manifest entry field")
        problem = _problem(problem_record)
        certificate = entry["certificate"]
        kind = certificate["kind"]
        if kind == "majorant":
            if set(certificate) - {"kind", "c", "lambda", "pointwise"}:
                raise _Unsupported("unrecognized majorant field")
            majorant = Majorant(_scalar(certificate["c"]), _scalar_list(certificate["lambda"]))
            square = None
            if "pointwise" in certificate:
                proof = certificate["pointwise"]
                if proof["method"] != "square":
                    raise _Unsupported("only explicit polynomial square certificates are supported")
                polynomial_record = proof["polynomial"]
                if set(proof) != {"method", "coefficient", "polynomial"} or set(
                    polynomial_record
                ) != {"variable", "coeffs"}:
                    raise ValueError("unexpected square certificate fields")
                if polynomial_record["variable"] != problem.output.gen.name:
                    raise ValueError("square certificate must use the problem coordinate")
                polynomial = Poly.from_list(
                    list(reversed(_scalar_list(polynomial_record["coeffs"]))),
                    problem.output.gen, domain=QQ,
                )
                square = SquareProof(polynomial, _scalar(proof["coefficient"]))
            result = verify_majorant(problem, majorant, square=square)
        elif kind in ("atoms", "strict_feasibility"):
            if set(certificate) != {"kind", "points", "masses"}:
                raise _Unsupported("use only exact rational points and masses")
            receipt["mode"] = "finite"
            raw_points = certificate["points"]
            if type(raw_points) is not list or any(
                type(point) is not list or len(point) != 1 for point in raw_points
            ):
                raise ValueError("each atom needs one coordinate")
            points = tuple(_scalar(value) for (value,) in raw_points)
            law = AtomicLaw(points, _scalar_list(certificate["masses"]))
            result = verify_atomic_law(problem, law, strict_feasibility=kind == "strict_feasibility")
        else:
            raise _Unsupported(f"unsupported certificate kind {kind!r}")
    except _Unsupported as error:
        return {**receipt, "verdict": "UNSUPPORTED", "reason": str(error)}
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return {**receipt, "verdict": "FAIL", "reason": str(error)}

    verdict = {"verified": "CERTIFIED", "disproved": "REFUTED"}.get(result.status, "FAIL")
    if kind != "majorant":
        receipt["witness_scope"] = certificate["points"]
    return {**receipt, "verdict": verdict, "scope": result.scope, "reason": result.reason,
            "certificate": certificate}


def replay_manifest(manifest):
    """Replay a nonempty manifest containing individually identified entries."""
    if type(manifest) is not list or not manifest:
        raise ValueError("a receipt manifest must be a nonempty list")
    identifiers = set()
    receipts = []
    for entry in manifest:
        if type(entry) is not dict or type(entry.get("id")) is not str or not entry["id"]:
            raise ValueError("each manifest entry needs a nonempty string id")
        if entry["id"] in identifiers:
            raise ValueError("manifest identifiers must be unique")
        identifiers.add(entry["id"])
        try:
            receipt = _replay(entry)
        except Exception as error:
            receipt = {"id": entry["id"], "verdict": "FAIL",
                       "reason": f"{type(error).__name__}: {error}"}
        if "expect" in entry:
            expected = entry["expect"]
            if expected not in _VERDICTS:
                raise ValueError("invalid expected control verdict")
            receipt.update(mode="control", expected=expected,
                           passed=receipt["verdict"] == expected)
        receipts.append(receipt)
    return receipts
