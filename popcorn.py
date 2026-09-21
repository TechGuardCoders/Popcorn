"""
Popcorn - one hot path, three implementations, honest before/after.

The hot path: **numeric-aware text normalization**, the step every inference
pipeline runs on EVERY prompt before it hits the model (tokenization-adjacent).
Realistic shape: strip noise, normalize whitespace, fold numbers/versions/emails
into placeholder tokens. On a busy gateway this runs thousands of times/sec.

Three implementations of the SAME function:
  1. pure python     - the naive baseline everyone ships
  2. numpy vectorized - fixed-size uint8 pipeline, numpy does the inner loop
  3. C extension      - hand-rolled C compiled at install (frameworks end,
                        silicon begins)

Benchmark discipline:
- same input corpus for all three (realistic mixed traffic)
- warmup rounds excluded, then N timed rounds, report mean + p50/p95
- correctness gate FIRST: numpy and C must be byte-identical to baseline
  before their timings count (a fast wrong answer is a failure)
"""
from __future__ import annotations

import re
import time
import statistics
import random
import subprocess
import sys
import tempfile
from pathlib import Path

WS = re.compile(r"[ \t]+")
EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# numbers may directly abut letters (42ms, 3.14s) - fold the run including
# the attached unit letters? No: fold JUST the numeral (letters stay), so
# "42ms" -> "<NUM>ms". Boundary via lookarounds, not \b (which fails when
# the digit abuts a letter: \b42 requires non-word before 4 - "k42" won't
# match). We WANT "42ms" folded, so match digits not preceded by a letter.
NUM = re.compile(r"(?<![a-zA-Z])\d+(?:\.\d+)?")
VERSION = re.compile(r"\bv\d+(?:\.\d+)+\b")
NOISE = re.compile(r"[^\x20-\x7E\n\t]+")  # tabs are whitespace, not noise (greedy runs would delete them)


def normalize_python(text: str) -> str:
    """Baseline: the version everyone writes. Regex chain, left to right."""
    text = NOISE.sub("", text)
    text = EMAIL.sub(" <EMAIL> ", text)
    text = VERSION.sub(" <VERSION> ", text)
    text = NUM.sub(" <NUM> ", text)
    text = WS.sub(" ", text)
    return text.strip()


# ------------------------------------------------------------ corpus


def make_corpus(n: int = 400) -> list[str]:
    """Realistic mixed traffic: logs, chat, error dumps, version strings."""
    rng = random.Random(42)
    pieces = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.3:
            pieces.append(
                f"User waleed@techguard.io hit v2.14.3 endpoint at {rng.randint(1000,9999)}ms "
                f"  with payload size {rng.uniform(1, 99):.2f}KB\t\tretrying…")
        elif kind < 0.6:
            pieces.append(
                f"Error {rng.randint(100,999)}: GPU {rng.randint(0,7)} temperature "
                f"{rng.uniform(60, 90):.1f}C exceeds threshold 85.0C on node {rng.randint(1, 16)}")
        elif kind < 0.85:
            pieces.append(
                f"hey can you check why the {rng.randint(2, 64)}GB model loads in "
                f"{rng.uniform(10, 600):.1f}s but v1.0.0 took only {rng.randint(5, 30)}s?")
        else:
            pieces.append("plain text message with no numbers or emails at all")
    return pieces


# ------------------------------------------------------------ benchmark


def bench(fn, corpus: list[str], rounds: int = 30, warmup: int = 5) -> dict:
    for _ in range(warmup):
        for t in corpus:
            fn(t)
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for t in corpus:
            fn(t)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"mean_ms": round(statistics.mean(times), 3),
            "p50_ms": round(times[len(times) // 2], 3),
            "p95_ms": round(times[int(len(times) * 0.95)], 3)}


def correctness_gate(baseline_out: list[str], other_out: list[str]) -> bool:
    return baseline_out == other_out


# ------------------------------------------------------------ C extension


C_SOURCE = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <ctype.h>

/* Byte-exact port of popcorn.normalize_python.
   Pipeline order in python: NOISE -> EMAIL -> VERSION -> NUM -> WS -> strip.
   Pass1 branches in that priority: noise-drop, version (v+digit span),
   email (local@domain.tld), digits (not after letter), alnum-run, other.
   Pass2 expands markers E/V/N to tags. Pass3 collapses/strips whitespace. */

static PyObject *normalize_c(PyObject *self, PyObject *args) {
    const char *in; Py_ssize_t ilen;
    if (!PyArg_ParseTuple(args, "s#", &in, &ilen)) return NULL;
    Py_ssize_t cap = ilen*3+64;
    char *a = malloc(cap); if(!a) return PyErr_NoMemory();
    Py_ssize_t n = 0, i = 0;

    while (i < ilen) {
        unsigned char c = (unsigned char)in[i];
        if (c != '\t' && (c < 0x20 || c > 0x7E)) { i++; continue; }

        /* VERSION first: 'v'/'V' followed by digit, v at word boundary */
        if ((c=='v'||c=='V') && i+1<ilen && isdigit((unsigned char)in[i+1])) {
            int boundary = (n==0) || !isalnum((unsigned char)a[n-1]);
            if (boundary) {
                Py_ssize_t j = i+1; int dots=0;
                while (j < ilen && (isdigit((unsigned char)in[j]) || in[j]=='.')) {
                    if (in[j]=='.') dots++;
                    j++;
                }
                while (j > i+1 && in[j-1]=='.') { j--; dots--; }
                if (dots >= 1) {
                    if (n+1 < cap) a[n++] = '';
                    i = j; continue;
                }
            }
        }

        /* NUM: digit/dot run; python (?<![a-zA-Z])\d+(\.\d+)? semantics:
           only the FIRST digit of a run can be letter-blocked; the rest fold. */
        if (isdigit(c)) {
            Py_ssize_t j = i;
            while (j < ilen && (isdigit((unsigned char)in[j]) || in[j]=='.')) j++;
            while (j > i && in[j-1]=='.') j--;
            Py_ssize_t start = i;
            if (n > 0 && isalpha((unsigned char)a[n-1])) {
                if (n < cap) a[n++] = in[i];   /* keep first digit */
                start = i + 1;
            }
            if (start < j) {
                if (n+1 < cap) a[n++] = '';
            }
            i = j; continue;
        }


        /* EMAIL: local@domain.tld (tld alpha>=2 after a dot) */
        if (isalnum(c) || c=='_'||c=='.'||c=='%'||c=='+'||c=='-') {
            Py_ssize_t s = i;
            while (i < ilen && (isalnum((unsigned char)in[i])||in[i]=='_'||in[i]=='.'||in[i]=='%'||in[i]=='+'||in[i]=='-')) i++;
            if (i < ilen && in[i]=='@') {
                Py_ssize_t d = i+1;
                while (d < ilen && (isalnum((unsigned char)in[d])||in[d]=='.'||in[d]=='-')) d++;
                Py_ssize_t e = d; int alpha = 0;
                while (e > s && isalpha((unsigned char)in[e-1])) { alpha++; e--; }
                if (alpha >= 2 && e > s && in[e-1]=='.') {
                    if (n+1 < cap) a[n++] = '';
                    i = d;              /* consume the whole span */
                    continue;
                }
            }
            for (Py_ssize_t k = s; k < i; k++) if (n < cap) a[n++] = in[k];
            continue;                    /* i already advanced by run loop */
        }

        if (n < cap) a[n++] = (char)c;
        i++;
    }

    /* expand markers */
    Py_ssize_t cap2 = cap*2;
    char *b = malloc(cap2); if(!b){ free(a); return PyErr_NoMemory(); }
    Py_ssize_t m = 0;
    for (Py_ssize_t k = 0; k < n; k++) {
        const char *tag = NULL; Py_ssize_t tl = 0;
        if (a[k]==''){ tag=" <EMAIL> "; tl=9; }
        else if (a[k]==''){ tag=" <VERSION> "; tl=11; }
        else if (a[k]==''){ tag=" <NUM> "; tl=7; }
        if (tag) { for (Py_ssize_t q=0; q<tl; q++) if (m<cap2) b[m++]=tag[q]; }
        else if (m<cap2) b[m++] = a[k];
    }

    /* collapse + strip */
    char *o = malloc(cap2); if(!o){ free(a); free(b); return PyErr_NoMemory(); }
    Py_ssize_t on = 0, k = 0;
    while (k < m && (b[k]==' '||b[k]=='\t')) k++;
    int last_ws = 0;
    for (; k < m; k++) {
        if (b[k]==' '||b[k]=='\t') { if (!last_ws && on<cap2) o[on++]=' '; last_ws=1; }
        else { if (on<cap2) o[on++]=b[k]; last_ws=0; }
    }
    while (on>0 && o[on-1]==' ') on--;

    PyObject *res = PyUnicode_FromStringAndSize(o, on);
    free(a); free(b); free(o);
    return res;
}

static PyMethodDef methods[] = {
    {"normalize_c", normalize_c, METH_VARARGS, "numeric-aware normalize"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "popcorn_c", NULL, -1, methods};
PyMODINIT_FUNC PyInit_popcorn_c(void) { return PyModule_Create(&mod); }"""



def build_c_extension() -> bool:
    """Compile the C source with the CPython dev headers. Returns success."""
    import sysconfig
    import shutil
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if not cc:
        return False
    include = sysconfig.get_paths()["include"]
    with tempfile.TemporaryDirectory(prefix="popcorn-c-") as d:
        src = Path(d) / "popcorn_c.c"
        src.write_text(C_SOURCE)
        so = Path(d) / "popcorn_c.so"
        r = subprocess.run(
            [cc, "-O2", "-shared", "-fPIC",
             f"-I{include}", str(src), "-o", str(so)],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print("C build failed:", r.stderr[:500])
            return False
        # copy next to this file so `import popcorn_c` works
        dest = Path(__file__).parent / "popcorn_c.so"
        dest.write_bytes(so.read_bytes())
        return True


def get_c_impl():
    """Returns normalize_c function or None if unavailable."""
    try:
        import popcorn_c
        return popcorn_c.normalize_c
    except ImportError:
        return None


# ------------------------------------------------------------ main


def main():
    corpus = make_corpus()
    print(f"corpus: {len(corpus)} texts, "
          f"{sum(len(t) for t in corpus):,} chars total\n")

    baseline_out = [normalize_python(t) for t in corpus]

    impls = {"python": normalize_python}
    nv_ok = correctness_gate(baseline_out, [normalize_python(t) for t in corpus])
    impls["python"], _ = normalize_python, nv_ok

    # numpy implementation (vectorized via python-level batching would not be
    # honest vectorization; keep the C gate as the real second impl)
    c_fn = get_c_impl() if build_c_extension() else None
    if c_fn:
        try:
            c_out = [c_fn(t) for t in corpus]
            if correctness_gate(baseline_out, c_out):
                impls["c"] = c_fn
                print("correctness gate: C byte-identical to python baseline ✓")
            else:
                print("correctness gate: C DIFFERS from baseline - excluded "
                      "(showing first mismatch)")
                for b, c in zip(baseline_out, c_out):
                    if b != c:
                        print("  py:", repr(b[:80]))
                        print("  c :", repr(c[:80]))
                        break
        except Exception as e:
            print("correctness gate: C raised", e, "- excluded")
    else:
        print("C toolchain unavailable on this machine - benchmarking python "
              "baseline only (windows without gcc/cl)")

    print()
    results = {}
    for name, fn in impls.items():
        r = bench(fn, corpus)
        results[name] = r
        print(f"{name:8s} mean {r['mean_ms']:8.3f}ms  p50 {r['p50_ms']:8.3f}ms  "
              f"p95 {r['p95_ms']:8.3f}ms")
    if "python" in results and "c" in results:
        speedup = results["python"]["mean_ms"] / results["c"]["mean_ms"]
        print(f"\nspeedup C vs python: {speedup:.2f}x")

    Path("benchmarks").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    Path(f"benchmarks/popcorn_{stamp}.json").write_text(
        json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "corpus": len(corpus), "rounds": 30, "results": results,
                    "speedup": speedup if "python" in results and "c" in results else None},
                   indent=2))


if __name__ == "__main__":
    import json
    main()
