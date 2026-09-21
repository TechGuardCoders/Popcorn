"""Build popcorn_c extension with MSYS2 gcc against CPython 3.11 headers."""
import subprocess
import sysconfig
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GCC = r"C:\msys64\mingw64\bin\gcc.exe"

inc = sysconfig.get_paths()["include"]
# python311.dll lives next to the interpreter executable
import sys
prefix = Path(sys.base_prefix)
libs = prefix / "libs"
# find import lib
cand = list(libs.glob("python3*.lib")) or list(prefix.glob("python3*.lib")) or list((prefix / "libs").glob("*.lib"))
if not cand:
    # uv-managed python: DLL-based import lib may not exist; use python3.dll directly via def? Simplest:
    # link against python311.dll in the prefix root (gcc can link DLLs directly)
    dlls = list(prefix.glob("python3*.dll"))
    link = str(dlls[0]) if dlls else None
else:
    link = str(cand[0])

src = HERE / "popcorn_c.c"
out = HERE / "popcorn_c.pyd"
print(f"building: gcc -O2 -shared -I{inc} {src} -> {out} (link: {link})")
cmd = [GCC, "-O2", "-shared", f"-I{inc}", str(src)]
if link:
    cmd += [str(link)]
cmd += ["-o", str(out)]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if r.returncode != 0:
    print("BUILD FAILED:\n", r.stderr[:2000])
    sys.exit(1)
print("BUILT:", out.exists())
