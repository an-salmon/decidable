import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

from sympy import Poly, log, symbols

from decidable.exp_series import check_exp_series
from decidable.size_calculus import (
    AtomicLaw, AverageProblem, DerivationNode, Majorant, MomentProblem, SizeBound, UpperBound,
    aggregate, bounded_raise, check_scaling, descend, exp_enclosure, grid_optimum, norm_weight,
    power, product, rational_power_enclosure, search_majorant, search_refutation, sharp_closed_tail,
    verify_atomic_law, verify_average_refutation, verify_derivation, verify_majorant,
)
from decidable.size_calculus.receipts import replay_manifest


def square_record():
    return {
        "id": "square",
        "problem": {
            "version": 1,
            "coordinates": ["z"],
            "support": ["le", "1", ["ref", "z"]],
            "inputs": [{"integrand": ["pow", ["ref", "z"], "2"], "budget": "4"}],
            "output": {"integrand": ["ref", "z"], "budget": "2"},
        },
        "certificate": {
            "kind": "majorant",
            "c": "1",
            "lambda": ["1/4"],
            "pointwise": {
                "method": "square", "coefficient": "1/4",
                "polynomial": {"variable": "z", "coeffs": ["-2", "1"]},
            },
        },
    }


class MomentTests(unittest.TestCase):
    def setUp(self):
        self.z = symbols("z")
        z = self.z
        self.problem = MomentProblem((Poly(z**2, z),), (4,), Poly(z, z), 2, lower=1)

    def test_grid_optimality_and_global_majorant_have_different_scope(self):
        self.assertEqual(grid_optimum(self.problem, (1, 3)).value, F(7, 4))
        self.assertEqual(search_refutation(self.problem, (1, 3)).status, "inconclusive")
        self.assertEqual(verify_majorant(self.problem, Majorant(F(3, 4), (F(1, 4),))).status,
                         "inconclusive")
        self.assertEqual(verify_majorant(self.problem, Majorant(1, (F(1, 4),))).status,
                         "verified")
        self.assertEqual(search_majorant(self.problem, (1, 2, 3)).witness,
                         Majorant(1, (F(1, 4),)))

    def test_strict_feasibility_and_atomic_refutation(self):
        target = replace(self.problem, output_budget=F(7, 4))
        self.assertEqual(verify_atomic_law(target, AtomicLaw((1,), (1,)),
                                           strict_feasibility=True).status, "verified")
        self.assertEqual(verify_atomic_law(target, AtomicLaw((2,), (1,)),
                                           strict_feasibility=True).status, "inconclusive")
        self.assertEqual(search_refutation(target, (1, 2, 3)).status, "disproved")
        self.assertEqual(verify_atomic_law(target, AtomicLaw((2,), (1,))).status, "disproved")
        self.assertEqual(verify_atomic_law(target, AtomicLaw((2,), (F(1, 2),))).status,
                         "inconclusive")
        with self.assertRaises(ValueError):
            verify_atomic_law(target, AtomicLaw((0,), (1,)))
        with self.assertRaises(TypeError):
            verify_atomic_law(target, AtomicLaw((2.0,), (1,)))

    def test_sign_isolation_and_endpoints(self):
        z = self.z
        p = MomentProblem((), (), Poly(0, z), 0, lower=0, upper=1)
        for polynomial in (z * (1 - z), (z**2 - 2)**2):
            MomentProblem((), (), Poly(polynomial, z), 1, lower=0, upper=1)
        with self.assertRaises(ValueError):
            replace(p, output=Poly((z**2 - F(1, 2)) * (z**2 - F(3, 4)), z))
        self.assertEqual(replace(p, upper=0, output=Poly(-z, z)).upper, 0)

    def test_grid_vertex_can_need_one_more_atom_than_input_bounds(self):
        z = self.z
        inputs = Poly((z * (z - 2))**2, z), Poly((z * (z - 1) / 2)**2, z)
        problem = MomentProblem(inputs, (F(1, 3), F(1, 3)), sum(inputs), F(1, 2), upper=2)
        optimum = grid_optimum(problem, (0, 1, 2))
        self.assertEqual(optimum.law, AtomicLaw((0, 1, 2), (F(1, 3),) * 3))
        self.assertEqual(optimum.value, F(2, 3))


class LocalRuleTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = symbols("X Y")
        self.r, self.u = symbols("r u", positive=True)
        self.bound = SizeBound(self.x, F(1, 2), self.r, True)

    def test_power_bounded_raise_product_and_descent(self):
        squared = power(self.bound, 2)
        self.assertEqual(squared, SizeBound(self.x**2, 1, self.r**2, True))
        raised = bounded_raise(self.bound, F(1, 4), UpperBound(self.x, 1))
        self.assertEqual(raised.scale, self.r**F(1, 2))
        with self.assertRaises(ValueError):
            bounded_raise(self.bound, F(1, 4), UpperBound(self.y, 1))
        self.assertEqual(descend(self.bound, 1).scale, self.r / log(2)**F(1, 2))
        self.assertEqual(product((self.bound, SizeBound(self.y, F(1, 2), self.u, True))),
                         SizeBound(self.x * self.y, 1, self.r * self.u, True))

    def test_aggregation_guard_and_retained_factor(self):
        bound = SizeBound(self.x, 2, self.r, True)
        other = SizeBound(self.y, 2, self.u, True)
        with self.assertRaises(ValueError):
            aggregate((bound, other), unit_factor=True)
        self.assertEqual(aggregate((bound, other)).scale,
                         (1 + 1 / log(2))**2 * (self.r + self.u))
        self.assertEqual(aggregate((self.bound, self.bound), unit_factor=True).scale, 2 * self.r)

    def test_proof_replay(self):
        given = DerivationNode("g", "given", argument="x")
        squared = DerivationNode("p", "power", ("g",), 2)
        goal = SizeBound(self.x**2, 1, self.r**2, True)
        self.assertEqual(verify_derivation({"x": self.bound}, (given, squared), goal).status,
                         "verified")
        self.assertEqual(verify_derivation({"x": self.bound}, (squared, given), goal).status,
                         "inconclusive")
        self.assertEqual(verify_derivation({"x": self.bound}, (), goal).status, "inconclusive")

    def test_normalized_poincare_and_one_sided_dilation(self):
        r, oscillation, gradient = symbols("r oscillation gradient")
        weights = {
            r: 1, oscillation: norm_weight(0, F(1, 2), 3, normalized=True),
            gradient: norm_weight(0, F(1, 2), 3, normalized=True, derivative_order=1),
        }
        self.assertEqual(check_scaling(oscillation, r * gradient, weights).status, "verified")
        self.assertEqual(check_scaling(oscillation, gradient, weights).status, "disproved")
        self.assertEqual(check_scaling(oscillation, gradient, weights, limit="small").status,
                         "verified")
        self.assertEqual(check_scaling(oscillation, symbols("unknown"), weights).status,
                         "unsupported")
        self.assertEqual(check_scaling(1 / (1 + r), 1, weights, limit="large").status,
                         "verified")


class AnalyticFixtureTests(unittest.TestCase):
    def test_rational_enclosures_and_average_refutation(self):
        lo, hi = exp_enclosure(1, 10)
        self.assertTrue(F(27, 10) < lo <= hi < F(28, 10))
        self.assertEqual(exp_enclosure(-1, 10), (1 / hi, 1 / lo))
        lo, hi = rational_power_enclosure(F(1, 2), F(1, 2), 12)
        self.assertTrue(lo**2 < F(1, 2) < hi**2)
        problem = AverageProblem(F(1, 2), (1, 1), (F(1, 2), F(1, 2)), 1)
        args = ((0, 1), (1, 0)), (F(1, 2), F(1, 2))
        self.assertEqual(verify_average_refutation(problem, *args).status, "disproved")
        self.assertEqual(verify_average_refutation(problem, *args, root_steps=0).status,
                         "inconclusive")

    def test_tail_and_e1(self):
        tail = sharp_closed_tail(3)
        self.assertEqual(tail.majorant, Majorant(-F(1, 2), (F(1, 2),)))
        self.assertEqual(tail.atoms, ((1, F(1, 2)), (3, F(1, 2))))
        self.assertEqual(sharp_closed_tail(F(3, 2)).probability, 1)
        v = symbols("v")
        self.assertEqual(check_exp_series(Poly(2 * v - 6, v), Poly(v**2 + 4 * v + 6, v),
                                          tail_start="auto").status, "verified")
        self.assertEqual(check_exp_series(Poly((v - 1)**2, v), Poly(0, v),
                                          tail_start="auto").status, "inconclusive")


class ReceiptTests(unittest.TestCase):
    def test_manifest_replays_examples_and_keeps_controls_distinct(self):
        square = square_record()
        atoms = square_record()
        atoms["id"] = "witness"
        atoms["problem"]["output"]["budget"] = "7/4"
        atoms["certificate"] = {"kind": "atoms", "points": [["2"]], "masses": ["1"]}
        control = square_record()
        control["id"] = "unsupported"
        control["problem"]["output"]["integrand"] = ["exp", ["ref", "z"]]
        control["expect"] = "UNSUPPORTED"
        x, y, z = replay_manifest([square, atoms, control])
        self.assertEqual((x["verdict"], y["verdict"], z["verdict"]),
                         ("CERTIFIED", "REFUTED", "UNSUPPORTED"))
        self.assertNotEqual(x["problem_sha256"], y["problem_sha256"])
        self.assertTrue(z["passed"])
        self.assertEqual(z["mode"], "control")

    def test_rejects_float_wrong_square_and_invalid_manifests(self):
        floating = square_record()
        floating["certificate"]["c"] = 1.0
        self.assertEqual(replay_manifest([floating])[0]["verdict"], "FAIL")
        wrong_square = square_record()
        wrong_square["certificate"]["pointwise"]["coefficient"] = "1"
        self.assertEqual(replay_manifest([wrong_square])[0]["verdict"], "FAIL")
        with self.assertRaises(ValueError):
            replay_manifest([])
        with self.assertRaises(ValueError):
            replay_manifest([square_record(), square_record()])

    def test_cli_replaces_stale_receipt_with_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            output = Path(directory) / "result.json"
            command = [sys.executable, "-m", "decidable.size_calculus", str(manifest),
                       "--output", str(output)]
            manifest.write_text(json.dumps([square_record()]))
            self.assertEqual(subprocess.run(command, check=False).returncode, 0)
            self.assertEqual(json.loads(output.read_text())[0]["verdict"], "CERTIFIED")
            manifest.write_text("[]")
            self.assertEqual(subprocess.run(command, check=False).returncode, 1)
            self.assertEqual(json.loads(output.read_text())[0]["verdict"], "FAIL")
