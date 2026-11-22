import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from decidable.report import CheckResult, Obligation, Report


class ReportTests(unittest.TestCase):
    def test_run_replaces_stale_output_and_records_failures(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checks.txt"
            path.write_text("stale PASS", encoding="utf-8")

            def check_stale_output():
                return not path.exists()

            def crash():
                raise RuntimeError("broken")

            report = Report([
                Obligation("fresh", check_stale_output, kind="numeric"),
                Obligation("limited", crash, kind="finite", scope="n <= 4"),
                Obligation("composite", depends_on=("fresh", "limited")),
                Obligation("control", lambda: True, kind="control"),
                Obligation("complete", depends_on=("fresh", "control")),
            ])
            self.assertEqual([item.passed for item in report.run(path)], [True, False, False, True, True])
            contents = path.read_text(encoding="utf-8")
            self.assertNotIn("stale PASS", contents)
            self.assertIn("FAIL [finite] (n <= 4) limited | RuntimeError: broken", contents)
            self.assertIn("3 passed, 2 failed", contents)

    def test_invalid_outcomes_and_report_definition(self):
        results = Report([Obligation("unknown", lambda: None), Obligation("numeric", lambda: 1)]).run()
        self.assertFalse(any(result.passed for result in results))
        with self.assertRaises(ValueError):
            Report([Obligation("missing", depends_on=("later",))])
        with self.assertRaises(ValueError):
            Report([Obligation("finite", lambda: True, kind="finite")])

    def test_finite_scope_is_inherited_and_cannot_be_broadened(self):
        finite_scope = "Indices 0 through 4"
        outcomes = Report([
            Obligation("sample", lambda: CheckResult("verified", finite_scope, "exact coefficients"),
                       kind="finite", scope=finite_scope),
            Obligation("same", depends_on=("sample",)),
            Obligation("broader", scope="All indices", depends_on=("same",)),
            Obligation("other", lambda: True, kind="finite", scope="Indices 5 through 8"),
            Obligation("mixed", depends_on=("sample", "other")),
        ]).run()
        self.assertEqual(outcomes[1].result.scope, finite_scope)
        self.assertEqual(outcomes[1].result.status, "verified")
        self.assertEqual(outcomes[2].result.status, "inconclusive")
        self.assertEqual(outcomes[4].result.status, "inconclusive")

    def test_numerical_evidence_cannot_pass_an_exact_obligation(self):
        def exact_check():
            self.fail("the exact callback must not execute on numerical evidence")

        outcomes = Report([
            Obligation("numeric", lambda: CheckResult("numerical_evidence", "a numerical model", "approximate"),
                       kind="numeric"),
            Obligation("symbolic", exact_check, depends_on=("numeric",)),
            Obligation("later", depends_on=("symbolic",)),
        ]).run()
        self.assertEqual([outcome.passed for outcome in outcomes], [True, False, False])
        self.assertEqual(outcomes[1].result.status, "numerical_evidence")
