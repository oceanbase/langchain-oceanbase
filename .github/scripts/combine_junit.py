#!/usr/bin/env python3
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    reports_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    suites_root = ET.Element("testsuites")
    for xml_path in sorted(reports_dir.rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        if root.tag == "testsuite":
            suites_root.append(root)
            continue
        for suite in root.findall("testsuite"):
            suites_root.append(suite)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(suites_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
