"""Run named checks and record their results with scope and dependencies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal


Kind = Literal["symbolic", "finite", "numeric", "control"]
Status = Literal["verified", "disproved", "unsupported", "inconclusive", "numerical_evidence"]


@dataclass(frozen=True)
class CheckResult:
    """An operation's result, with the exact scope and optional witness."""

    status: Status
    scope: str
    reason: str
    witness: object | None = None

    def __post_init__(self) -> None:
        if self.status not in ("verified", "disproved", "unsupported", "inconclusive", "numerical_evidence"):
            raise ValueError(f"unknown check status {self.status!r}")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("a check result must declare its scope")
        if not isinstance(self.reason, str):
            raise TypeError("a check result reason must be text")


@dataclass(frozen=True)
class Obligation:
    """A named check, or a composite that depends on earlier checks."""

    name: str
    check: Callable[[], CheckResult | bool | None] | None = None
    kind: Kind = "symbolic"
    scope: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    obligation: Obligation
    passed: bool
    detail: str
    result: CheckResult | None = None


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
            if obligation.scope is not None and (
                not isinstance(obligation.scope, str) or not obligation.scope.strip()
            ):
                raise ValueError("an obligation scope must be nonempty text")
            if obligation.kind == "finite" and obligation.scope is None:
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

        outcomes: dict[str, Outcome] = {}
        scopes: dict[str, str] = {}
        legacy: set[str] = set()
        for obligation in self.obligations:
            dependencies = tuple(outcomes[name] for name in obligation.depends_on)
            failed = [item.obligation.name for item in dependencies if not item.passed]
            dependency_scopes = {scopes[name] for name in obligation.depends_on if name in scopes}
            scope = obligation.scope
            if scope is None and len(dependency_scopes) == 1:
                scope = next(iter(dependency_scopes))
            compatible_scope = not dependency_scopes or dependency_scopes == {scope}
            only_legacy = legacy.issuperset(obligation.depends_on)
            numerical = any(item.result.status == "numerical_evidence" for item in dependencies)
            passed = False
            if failed:
                status, detail = "inconclusive", f"failed dependencies: {', '.join(failed)}"
            elif not compatible_scope:
                status, detail = "inconclusive", "dependency scopes do not match the obligation scope"
            elif numerical and obligation.kind != "numeric":
                status, detail = "numerical_evidence", "numerical dependencies cannot establish an exact conclusion"
                # Keep the legacy Boolean presentation for composite reports.
                passed = only_legacy and obligation.check is None
                if passed:
                    legacy.add(obligation.name)
            elif obligation.check is None:
                status = "numerical_evidence" if obligation.kind == "numeric" else "verified"
                detail, passed = "all dependencies passed", True
                if only_legacy:
                    legacy.add(obligation.name)
            else:
                try:
                    result = obligation.check()
                    if isinstance(result, CheckResult):
                        if scope and result.scope != scope:
                            status, detail = "inconclusive", "check result scope does not match the obligation scope"
                        else:
                            scope = result.scope
                            if obligation.kind == "numeric" and result.status in ("verified", "disproved"):
                                result = CheckResult("numerical_evidence", scope, result.reason, result.witness)
                            passed = result.status == "verified" or (
                                obligation.kind == "numeric" and result.status == "numerical_evidence"
                            )
                            scopes[obligation.name] = scope
                            outcomes[obligation.name] = Outcome(obligation, passed, result.reason, result)
                            continue
                    elif dependency_scopes and obligation.scope is None:
                        status, detail = "inconclusive", "a Boolean check must declare its dependency scope"
                    else:
                        passed = result is True
                        status = "inconclusive"
                        if passed:
                            status = "numerical_evidence" if obligation.kind == "numeric" else "verified"
                        detail = "" if passed else "check did not return True"
                        if only_legacy:
                            legacy.add(obligation.name)
                except Exception as exc:
                    status, detail = "inconclusive", f"{type(exc).__name__}: {exc}"
            if scope:
                scopes[obligation.name] = scope
            result = CheckResult(status, scope or "Declared obligation", detail)
            outcomes[obligation.name] = Outcome(obligation, passed, detail, result)

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
            scope = result.result.scope if result.result.scope != "Declared obligation" else obligation.scope
            scope = f" ({scope})" if scope else ""
            detail = f" | {result.detail}" if result.detail else ""
            evidence_status = ""
            if result.result.status != "verified":
                evidence_status = f" | {result.result.status}"
            line = f"{status} [{obligation.kind}]{scope} {obligation.name}{detail}{evidence_status}"
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
