r"""
data/decode_unicode.py
======================

Converts JSON files that contain \uXXXX escape sequences into properly
encoded UTF-8 files where names display as readable characters.

Usage
-----
    # Fix players.json in-place
    python data/decode_unicode.py data/players.json

    # Fix any JSON file
    python data/decode_unicode.py path/to/any_file.json

    # Write to a different output file
    python data/decode_unicode.py data/players.json data/players_decoded.json

What it does
------------
  Mat\u011bj Kov\u00e1\u0159  →  Matěj Kovář
  Tom\u00e1\u0161            →  Tomáš
  Luka Modri\u0107       →  Luka Modrić
"""

from __future__ import annotations
import json
import sys
from pathlib import Path


def decode_file(input_path: str, output_path: str | None = None) -> None:
    src = Path(input_path)
    dst = Path(output_path) if output_path else src   # in-place by default

    with open(src, encoding="utf-8") as f:
        data = json.load(f)   # json.load automatically decodes \uXXXX sequences

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Quick sanity check
    total = len(data) if isinstance(data, list) else "?"
    print(f"✓ Decoded {total} records → {dst}")

    # Show a few sample names so you can verify
    if isinstance(data, list):
        names = [r.get("Player") or r.get("name") or "" for r in data[:5] if isinstance(r, dict)]
        if names:
            print("  Sample names:", " · ".join(n for n in names if n))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data/decode_unicode.py <input.json> [output.json]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    decode_file(inp, out)