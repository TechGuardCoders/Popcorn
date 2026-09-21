"""Offline tests: correctness of the baseline + gate discipline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from popcorn import normalize_python, make_corpus, correctness_gate  # noqa: E402


def test_numbers_folded():
    out = normalize_python("took 42ms and 3.14s")
    assert "<NUM>" in out and " 42" not in f" {out} " and " 3." not in out
    assert "ms" in out and "s" in out  # units survive; numerals folded


def test_versions_folded():
    out = normalize_python("running v2.14.3 now")
    assert "<VERSION>" in out and "2.14.3" not in out


def test_emails_folded():
    out = normalize_python("mail waleed@techguard.io please")
    assert "<EMAIL>" in out and "waleed@techguard.io" not in out


def test_whitespace_collapsed_and_noise_dropped():
    # NBSP (\u00a0) is >0x7E -> dropped by NOISE (not treated as ws);
    # zero-width (\u200b) also dropped. "a b c" with the nbsp gone.
    out = normalize_python("a\t\tb   c\u00a0d\u200b")
    assert out == "a b cd"  # nbsp DROPPED (noise), tabs/spaces collapsed


def test_corpus_nonempty_and_stable():
    c1, c2 = make_corpus(), make_corpus()
    assert c1 == c2 and len(c1) == 400  # seeded -> deterministic


def test_gate_is_exact():
    assert correctness_gate(["a", "b"], ["a", "b"])
    assert not correctness_gate(["a", "b"], ["a ", "b"])  # byte-identical or fail
