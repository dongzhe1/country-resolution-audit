#!/usr/bin/env python3
"""Claims the write-up makes about the assessment document itself.

    python source_document_facts.py /path/to/results_dir
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from facts import emit

DOC = (Path(__file__).resolve().parent.parent / "docs" / "sources"
       / "MEPC82-INF8-Add2.txt")

AGGREGATED = ["Vanuatu", "Kiribati", "Samoa", "Solomon Islands", "Comoros",
              "Djibouti", "Cabo Verde", "Maldives", "Grenada", "Palau",
              "Saint Lucia", "Seychelles", "Tonga", "Micronesia"]


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    if not DOC.exists():
        sys.exit(
            f"{DOC} not found.\n\n"
            f"This script counts things in the assessment's own text, so it needs\n"
            f"that document. It is public but large, and is not redistributed here.\n\n"
            f"  1. Download MEPC 82/INF.8/Add.2 from the IMO document repository\n"
            f"     (IMODOCS, or the UNCTAD publication page for the same study).\n"
            f"  2. Convert it:  pdftotext -layout MEPC82-INF8-Add2.pdf\n"
            f"  3. Put both files at:  {DOC.parent}/\n\n"
            f"Every other script in analysis/ runs without it.")
    text = DOC.read_text(errors="replace")

    pdf = DOC.with_suffix(".pdf")
    pages = len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes())) if pdf.exists() else 0

    hits = {c: len(re.findall(re.escape(c), text)) for c in AGGREGATED}
    named = {c: n for c, n in hits.items() if n}
    print(f"{DOC.name}: {len(text.splitlines()):,} lines, {pages} pages")
    print(f"aggregated states checked: {len(AGGREGATED)}")
    for c, n in hits.items():
        print(f"  {c:<18} {n}")
    if named:
        print(f"\nNOT ZERO: {named}\n"
              f"These names are asserted to be absent. Re-read the "
              f"context before relying on that.")
    else:
        print("\nnone of them appears anywhere in the document")

    emit(Path(sys.argv[1]), "document", {
        "pages": pages,
        "aggregated_checked": len(AGGREGATED),
        "aggregated_named": len(named),
    })


if __name__ == "__main__":
    main()
