#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit(path: Path) -> dict[str, int | str]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    passed = max(tests - failures - errors - skipped, 0)
    executed = max(tests - skipped, 0)
    pass_rate = (passed / executed * 100.0) if executed else 0.0
    return {
        "name": path.name,
        "tests": tests,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "executed": executed,
        "pass_rate": round(pass_rate, 1),
    }


def render_summary(entries: list[dict[str, int | str]]) -> str:
    tests = sum(int(entry["tests"]) for entry in entries)
    passed = sum(int(entry["passed"]) for entry in entries)
    failures = sum(int(entry["failures"]) for entry in entries)
    errors = sum(int(entry["errors"]) for entry in entries)
    skipped = sum(int(entry["skipped"]) for entry in entries)
    executed = sum(int(entry["executed"]) for entry in entries)
    pass_rate = (passed / executed * 100.0) if executed else 0.0

    lines = [
        "## Test Summary",
        "",
        f"- Total cases: `{tests}`",
        f"- Passed: `{passed}`",
        f"- Failed: `{failures}`",
        f"- Errors: `{errors}`",
        f"- Skipped: `{skipped}`",
        f"- Pass rate (executed cases): `{pass_rate:.1f}%`",
        "",
        "| Report | Tests | Passed | Failed | Errors | Skipped | Pass rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for entry in sorted(entries, key=lambda value: str(value["name"])):
        lines.append(
            "| {name} | {tests} | {passed} | {failures} | {errors} | {skipped} | {pass_rate:.1f}% |".format(
                **entry
            )
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    report_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    reports = sorted(report_root.rglob("*.xml"))
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")

    if not reports:
        content = "## Test Summary\n\nNo JUnit XML reports were found.\n"
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(content)
        else:
            sys.stdout.write(content)
        return 0

    entries = [parse_junit(path) for path in reports]
    content = render_summary(entries)

    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(content)
    else:
        sys.stdout.write(content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
