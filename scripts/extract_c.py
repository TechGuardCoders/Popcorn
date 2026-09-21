"""Extract the C source embedded in popcorn.py to popcorn_c.c for building."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import popcorn

out = Path(__file__).resolve().parent.parent / "popcorn_c.c"
out.write_text(popcorn.C_SOURCE)
print(f"extracted {len(popcorn.C_SOURCE)} chars -> {out.name}")
