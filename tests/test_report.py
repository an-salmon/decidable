import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from decidable.report import Obligation, Report


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
