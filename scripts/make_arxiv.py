#!/usr/bin/env python3
"""Package the paper directory into an arXiv-ready tar.gz archive.

Parses main.tex and all \\input{}'ed files to collect only referenced
figures, tables, and supporting .tex files.  Includes the .bbl so arXiv
doesn't need to run bibtex.

Usage:
    python3 scripts/make_arxiv.py            # default output: paper/arxiv_submission.tar.gz
    python3 scripts/make_arxiv.py -o foo.tar.gz
"""

import argparse
import os
import re
import sys
import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT_DIR / "paper"

# Extensions to try when \includegraphics omits the extension
FIGURE_EXTENSIONS = [".png", ".pdf", ".jpg", ".jpeg", ".eps"]


def parse_inputs(tex_path: Path, seen: set[Path] | None = None) -> list[Path]:
    """Recursively find all \\input{} references from a .tex file."""
    if seen is None:
        seen = set()
    if tex_path in seen:
        return []
    seen.add(tex_path)

    results = []
    if not tex_path.exists():
        return results

    text = tex_path.read_text(errors="replace")
    # Match \input{...} — may or may not have .tex extension
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        ref = m.group(1)
        # Skip commented-out lines
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start : m.start()]
        if "%" in line:
            continue

        path = PAPER_DIR / ref
        if not path.suffix:
            path = path.with_suffix(".tex")
        results.append(path)
        results.extend(parse_inputs(path, seen))

    return results


def parse_figures(tex_path: Path) -> list[str]:
    """Extract all \\includegraphics references from a .tex file."""
    if not tex_path.exists():
        return []
    text = tex_path.read_text(errors="replace")
    refs = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        # Skip commented-out lines
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start : m.start()]
        if "%" in line:
            continue
        refs.append(m.group(1))
    return refs


def resolve_figure(ref: str) -> Path | None:
    """Resolve a figure reference to an actual file on disk."""
    path = PAPER_DIR / ref
    if path.exists():
        return path
    # Try adding extensions
    if not path.suffix:
        for ext in FIGURE_EXTENSIONS:
            candidate = path.with_suffix(ext)
            if candidate.exists():
                return candidate
    return None


def collect_table_inputs(tex_path: Path) -> list[Path]:
    """Extract \\input{tables/...} references from a .tex file."""
    if not tex_path.exists():
        return []
    text = tex_path.read_text(errors="replace")
    results = []
    for m in re.finditer(r"\\input\{(tables/[^}]+)\}", text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start : m.start()]
        if "%" in line:
            continue
        ref = m.group(1)
        path = PAPER_DIR / ref
        if not path.suffix:
            path = path.with_suffix(".tex")
        results.append(path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Create arXiv submission archive")
    parser.add_argument(
        "-o",
        "--output",
        default=str(ROOT_DIR / "arxiv_submission.tar.gz"),
        help="Output archive path (default: arxiv_submission.tar.gz in project root)",
    )
    args = parser.parse_args()

    main_tex = PAPER_DIR / "main.tex"
    if not main_tex.exists():
        print(f"ERROR: {main_tex} not found", file=sys.stderr)
        sys.exit(1)

    # --- Collect all files ---
    files_to_include: dict[str, Path] = {}  # archive_name -> absolute_path

    # 1. main.tex
    files_to_include["main.tex"] = main_tex

    # 2. numbers.tex
    numbers = PAPER_DIR / "numbers.tex"
    if numbers.exists():
        files_to_include["numbers.tex"] = numbers
    else:
        print(f"WARNING: {numbers} not found", file=sys.stderr)

    # 3. main.bbl (compiled bibliography)
    bbl = PAPER_DIR / "main.bbl"
    if bbl.exists():
        files_to_include["main.bbl"] = bbl
    else:
        print(f"WARNING: {bbl} not found — arXiv needs this to render references", file=sys.stderr)

    # 4. Recursively find all \input{}'ed .tex files (sections)
    input_files = parse_inputs(main_tex)
    for path in input_files:
        if not path.exists():
            print(f"WARNING: referenced file not found: {path}", file=sys.stderr)
            continue
        arcname = str(path.relative_to(PAPER_DIR))
        files_to_include[arcname] = path

    # 5. Collect all \input{tables/...} from every .tex file
    all_tex_files = [main_tex] + input_files
    for tex_file in all_tex_files:
        for table_path in collect_table_inputs(tex_file):
            if not table_path.exists():
                print(f"WARNING: table not found: {table_path}", file=sys.stderr)
                continue
            arcname = str(table_path.relative_to(PAPER_DIR))
            files_to_include[arcname] = table_path

    # 6. Collect all \includegraphics references from every .tex file
    figure_refs = []
    for tex_file in all_tex_files:
        figure_refs.extend(parse_figures(tex_file))

    for ref in figure_refs:
        resolved = resolve_figure(ref)
        if resolved is None:
            print(f"WARNING: figure not found: {ref}", file=sys.stderr)
            continue
        arcname = str(resolved.relative_to(PAPER_DIR))
        files_to_include[arcname] = resolved

    # --- Validate ---
    missing = [name for name, path in files_to_include.items() if not path.exists()]
    if missing:
        print(f"ERROR: {len(missing)} files missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    # --- Create archive ---
    output_path = Path(args.output)
    with tarfile.open(output_path, "w:gz") as tar:
        for arcname in sorted(files_to_include):
            tar.add(files_to_include[arcname], arcname=arcname)

    # --- Print manifest ---
    sections = sorted(n for n in files_to_include if n.startswith("sections/"))
    tables = sorted(n for n in files_to_include if n.startswith("tables/"))
    figures = sorted(n for n in files_to_include if n.startswith("figures/"))
    root_files = sorted(n for n in files_to_include if "/" not in n)

    print(f"\n{'=' * 60}")
    print(f"arXiv submission archive: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nRoot files ({len(root_files)}):")
    for f in root_files:
        print(f"  {f}")
    print(f"\nSections ({len(sections)}):")
    for f in sections:
        print(f"  {f}")
    print(f"\nTables ({len(tables)}):")
    for f in tables:
        print(f"  {f}")
    print(f"\nFigures ({len(figures)}):")
    for f in figures:
        print(f"  {f}")

    total = len(files_to_include)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nTotal: {total} files, {size_mb:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
