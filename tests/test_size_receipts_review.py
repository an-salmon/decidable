import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decidable.size_calculus import __main__ as cli
from decidable.size_calculus._nra import BackendUnavailable
from decidable.size_calculus.receipts import replay_manifest


def _entry(identifier="run"):
    return {
        "id": identifier,
        "problem": {
            "version": 1,
            "id": "moment",
            "coordinates": ["z"],
            "support": ["le", "1", ["ref", "z"]],
            "inputs": [{"integrand": ["pow", ["ref", "z"], "2"], "budget": "4"}],
            "output": {"integrand": ["ref", "z"], "budget": "2"},
        },
        "certificate": {"kind": "majorant", "c": "1", "lambda": ["1/4"]},
    }


class ReceiptBoundaryTests(unittest.TestCase):
    def test_portable_decider_handles_an_interval_and_boolean_polynomial_support(self):
        interval = _entry("interval")
        interval["certificate"] = {"kind": "decide"}
        certification, = replay_manifest([interval])
        if certification["verdict"] == "UNSUPPORTED":
            self.skipTest(certification["reason"])
        self.assertEqual(certification["verdict"], "CERTIFIED")
        x, y = ["ref", "x"], ["ref", "y"]
        multivariate = {
            "id": "algebraic",
            "problem": {
                "version": 1,
                "coordinates": ["x", "y"],
                "support": ["and", ["eq", y, ["pow", x, "2"]],
                            ["or", ["eq", x, "1"], ["and", ["lt", "0", x],
                                                     ["not", ["lt", "1", x]]]]],
                "inputs": [],
                "output": {"integrand": y, "budget": "1/2"},
            },
            "certificate": {"kind": "decide"},
        }
        refutation, = replay_manifest([multivariate])
        self.assertEqual((refutation["verdict"], refutation["mode"]), ("REFUTED", "finite"))
        self.assertEqual(json.loads(json.dumps(refutation))["witness"]["atom_coordinates"],
                         [["x0_0", "x0_1"]])
        self.assertIn("p0", refutation["witness"]["assignment"])
        multivariate["problem"]["output"]["budget"] = "1"
        self.assertEqual(replay_manifest([multivariate])[0]["verdict"], "CERTIFIED")

    def test_portable_decider_keeps_unknown_and_unavailable_unsupported(self):
        entry = _entry()
        entry["certificate"] = {"kind": "decide"}
        entry["expect"] = "UNSUPPORTED"
        solver = "decidable.size_calculus.semialgebraic.solve"
        for replies in (BackendUnavailable("unavailable"), [("unsat", None), ("unknown", None)]):
            with self.subTest(replies=replies), patch(solver, side_effect=replies):
                receipt, = replay_manifest([entry])
                self.assertEqual(receipt["verdict"], "UNSUPPORTED")
                self.assertTrue(receipt["passed"])
                self.assertNotIn("witness", receipt)

    def test_portable_decider_rejects_invalid_syntax_before_dispatch(self):
        entry = _entry()
        entry["certificate"] = {"kind": "decide"}
        with patch("decidable.size_calculus.receipts.decide_moment") as decider:
            entry["problem"]["output"]["integrand"] = ["ref", "undeclared"]
            self.assertEqual(replay_manifest([entry])[0]["verdict"], "FAIL")
            entry["problem"]["output"]["integrand"] = ["exp", ["ref", "z"]]
            self.assertEqual(replay_manifest([entry])[0]["verdict"], "UNSUPPORTED")
            decider.assert_not_called()

    def test_strict_feasibility_certifies_a_distinct_claim(self):
        feasibility, refutation = _entry("strict"), _entry("refutation")
        for record, kind in ((feasibility, "strict_feasibility"), (refutation, "atoms")):
            record["problem"]["output"]["budget"] = "0"
            record["certificate"] = {"kind": kind, "points": [["1"]], "masses": ["1"]}
        strict, bound = replay_manifest([feasibility, refutation])
        self.assertEqual((strict["claim"], strict["verdict"]), ("strict_feasibility", "CERTIFIED"))
        self.assertEqual((bound["claim"], bound["verdict"]), ("moment_bound", "REFUTED"))
        self.assertEqual(strict["problem_id"], "moment")
        self.assertEqual(strict["problem_version"], 1)
        self.assertEqual(strict["problem_sha256"], bound["problem_sha256"])
        self.assertEqual(strict["mode"], "finite")
        self.assertEqual(strict["witness_scope"], [["1"]])
        self.assertIn("atomic law", strict["scope"])
        feasibility["certificate"]["points"][0][0] = "2"
        self.assertEqual(strict["certificate"]["points"], [["1"]])

    def test_rejected_input_can_pass_a_fail_control_but_checker_crashes_cannot(self):
        rejected = _entry()
        rejected["expect"] = "FAIL"
        rejected["certificate"]["lambda"] = []
        receipt, = replay_manifest([rejected])
        self.assertEqual(receipt["verdict"], "FAIL")
        self.assertTrue(receipt["passed"])

        crashes = _entry()
        crashes["expect"] = "FAIL"
        target = "decidable.size_calculus.receipts.verify_majorant"
        for exception in (TypeError, ValueError, RuntimeError):
            with self.subTest(exception=exception), patch(target, side_effect=exception("crash")):
                receipt, = replay_manifest([crashes])
                self.assertEqual(receipt["verdict"], "FAIL")
                self.assertFalse(receipt["passed"])
                self.assertTrue(receipt["checker_error"])
                self.assertNotIn("certificate", receipt)

    def test_malformed_is_distinct_from_unsupported(self):
        examples = (
            (["exp", ["ref", "z"]], "UNSUPPORTED"),
            (["exp"], "FAIL"),
            (["exp", ["ref", "undeclared"]], "FAIL"),
            (["add", ["exp", ["ref", "z"]], ["ref", "undeclared"]], "FAIL"),
            (["pow", ["ref", "z"], "1/2"], "UNSUPPORTED"),
            (["pow", ["ref", "undeclared"], "1/2"], "FAIL"),
        )
        for expression, verdict in examples:
            with self.subTest(expression=expression):
                entry = _entry()
                entry["problem"]["output"]["integrand"] = expression
                self.assertEqual(replay_manifest([entry])[0]["verdict"], verdict)

        entry = _entry()
        entry["problem"]["support"][0] = "lt"
        self.assertEqual(replay_manifest([entry])[0]["verdict"], "UNSUPPORTED")
        entry["problem"]["support"] = ["lt", "1"]
        self.assertEqual(replay_manifest([entry])[0]["verdict"], "FAIL")

    def test_unsupported_control_records_evidence_without_a_certificate(self):
        entry = _entry()
        entry["problem"]["output"]["integrand"] = ["exp", ["ref", "z"]]
        entry["expect"] = "UNSUPPORTED"
        receipt, = replay_manifest([entry])
        self.assertEqual((receipt["mode"], receipt["evidence_mode"]), ("control", "unsupported"))
        self.assertTrue(receipt["passed"])
        self.assertNotIn("certificate", receipt)

    def test_explicit_identity_manifest_preflights_entries_and_emits_declared_order(self):
        a, b = _entry("a"), _entry("b")
        receipts = replay_manifest({"expected_ids": ["b", "a"], "entries": [a, b]})
        self.assertEqual([receipt["id"] for receipt in receipts], ["b", "a"])
        bad_manifests = (
            {"expected_ids": ["a", "b"], "entries": [a]},
            {"expected_ids": ["a"], "entries": [a, b]},
            {"expected_ids": ["a", "a"], "entries": [a]},
            {"expected_ids": ["a"], "entries": [a, a]},
            [a, {**b, "expect": "UNRECOGNIZED"}],
        )
        with patch("decidable.size_calculus.receipts.verify_majorant") as checker:
            for manifest in bad_manifests:
                with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                    replay_manifest(manifest)
            checker.assert_not_called()

    def test_cli_crash_replaces_stale_output_with_failed_control(self):
        entry = _entry()
        entry["expect"] = "FAIL"
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            output = Path(directory) / "receipt.json"
            manifest.write_text(json.dumps([entry]), encoding="utf-8")
            output.write_text('[{"verdict":"CERTIFIED"}]', encoding="utf-8")
            command = ["size_calculus", str(manifest), "--output", str(output)]
            with patch.object(sys, "argv", command), patch(
                "decidable.size_calculus.receipts.verify_majorant", side_effect=ValueError("crash")
            ):
                self.assertEqual(cli.main(), 1)
            receipt, = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(receipt["checker_error"])
            self.assertFalse(receipt["passed"])

    def test_atomic_writer_cleans_temporary_file_on_failed_replace(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    cli._write_receipts(output, "[]\n")
            self.assertEqual(list(output.parent.iterdir()), [])
