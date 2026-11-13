"""requirements.txt pins must match the library versions recorded by the 2026-09-08 runs
(CPU-only; UNTESTED ON GPU). Guards against the pins and the recorded versions drifting."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "results" / "run_20260908_093810" / "run_metadata.json"


def _pins():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=)\s*([^\s#]+)", line)
        if m:
            pins[m.group(1).lower()] = (m.group(2), m.group(3))
    return pins


def test_core_libraries_are_pinned_to_the_recorded_versions():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    pins = _pins()
    assert pins["transformers"] == ("==", rec["transformers_version"]) == ("==", "5.16.1")
    assert pins["datasets"] == ("==", rec["datasets_version"]) == ("==", "5.0.1")
    assert pins["torch"] == ("==", rec["torch_version"].split("+")[0]) == ("==", "2.11.0")


def test_all_four_metadata_files_agree_with_each_other():
    vers = {tuple(json.loads(p.read_text(encoding="utf-8"))[k] for k in
                  ("transformers_version", "torch_version", "datasets_version"))
            for p in (ROOT / "results").glob("run_20260908_*/run_metadata.json")}
    assert len(vers) == 1


def test_old_transformers_floor_is_gone_and_other_deps_untouched():
    active = [l for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.lstrip().startswith("#")]
    assert not any(l.replace(" ", "").startswith("transformers>=") for l in active)  # comments may mention it
    pins = _pins()
    assert pins["accelerate"] == (">=", "0.26.0") and pins["numpy"] == (">=", "1.26.4")
    assert pins["scipy"] == (">=", "1.14.1") and pins["matplotlib"] == (">=", "3.8")
    assert pins["tqdm"] == (">=", "4.66.5") and pins["pytest"] == (">=", "9.0.1")


def test_cu128_build_is_documented():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "cu128" in text
