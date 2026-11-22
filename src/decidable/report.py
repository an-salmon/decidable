"""Run named checks and record their results with scope and dependencies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal


Kind = Literal["symbolic", "finite", "numeric", "control"]


@dataclass(frozen=True)
class Obligation:
    """A named Boolean check, or a composite that depends on earlier checks."""

    name: str
    check: Callable[[], bool | None] | None = None
    kind: Kind = "symbolic"
    scope: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    obligation: Obligation
    passed: bool
    detail: str


class Report:
    """Evaluate each check once per run, in the supplied dependency order."""

    def __init__(self, obligations: Iterable[Obligation]):
        self.obligations = tuple(obligations)
        seen = set()
        for obligation in self.obligations:
            if not obligation.name.strip() or obligation.name in seen:
                raise ValueError("obligation names must be nonempty and unique")
            if obligation.kind not in ("symbolic", "finite", "numeric", "control"):
                raise ValueError(f"unknown check kind {obligation.kind!r}")
            if obligation.kind == "finite" and not (obligation.scope or "").strip():
                raise ValueError("a finite check must declare its scope")
            if obligation.check is None and not obligation.depends_on:
                raise ValueError("an obligation needs a check or dependencies")
            if not seen.issuperset(obligation.depends_on):
                raise ValueError("dependencies must name earlier obligations")
            seen.add(obligation.name)

    def run(self, output: str | Path | None = None) -> tuple[Outcome, ...]:
        """Run the report and optionally replace a text output from an earlier run."""
        path = Path(output) if output is not None else None
        if path is not None:
            path.unlink(missing_ok=True)

        outcomes = {}
        for obligation in self.obligations:
            failed = [name for name in obligation.depends_on if not outcomes[name].passed]
            if failed:
                outcome = Outcome(obligation, False, f"failed dependencies: {', '.join(failed)}")
            elif obligation.check is None:
                outcome = Outcome(obligation, True, "all dependencies passed")
            else:
                try:
                    result = obligation.check()
                    detail = "" if result is True else "check did not return True"
                    outcome = Outcome(obligation, result is True, detail)
                except Exception as exc:
                    outcome = Outcome(obligation, False, f"{type(exc).__name__}: {exc}")
            outcomes[obligation.name] = outcome

        results = tuple(outcomes.values())
        if path is not None:
            self._write(path, results)
        return results

    @staticmethod
    def _write(path: Path, results: tuple[Outcome, ...]) -> None:
        lines = []
        for result in results:
            obligation = result.obligation
            status = "PASS" if result.passed else "FAIL"
            scope = f" ({obligation.scope})" if obligation.scope else ""
            detail = f" | {result.detail}" if result.detail else ""
            line = f"{status} [{obligation.kind}]{scope} {obligation.name}{detail}"
            lines.append(" ".join(line.splitlines()))
        passed = sum(result.passed for result in results)
        lines.append(f"{passed} passed, {len(results) - passed} failed")

        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
            temporary_path = Path(temp.name)
            try:
                temp.write("\n".join(lines) + "\n")
                temp.close()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
