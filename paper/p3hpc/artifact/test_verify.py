"""CPU-only regression checks for the paper evidence audit.

Run from the repository root:
    python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
"""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify


PAPER = Path(__file__).resolve().parents[2]
RESULTS = PAPER / "results"
CELL = RESULTS / "nvidia/paper-v1-strict/SmolLM2-135M_summary.json"


class EvidenceTests(unittest.TestCase):
    def test_frozen_inventory_and_gates(self):
        with patch.object(verify, "ROOT", PAPER), patch.object(verify, "RESULTS", RESULTS):
            self.assertEqual(verify.audit_results()["summaries"], 50)

    def test_summary_cannot_diverge_from_raw_record(self):
        original = verify.load_json

        def changed(path):
            data = original(path)
            if path == CELL:
                data[0]["timings"]["inference_ms"] += 1
            return data

        with patch.object(verify, "ROOT", PAPER), patch.object(verify, "RESULTS", RESULTS), \
             patch.object(verify, "load_json", changed):
            with self.assertRaisesRegex(RuntimeError, "summary disagrees"):
                verify.audit_results()

    def test_timing_cannot_diverge_from_samples(self):
        original = verify.load_json

        def changed(path):
            data = original(path)
            if path.name == "SmolLM2-135M_meganeura.json":
                data["timings"]["training_ms"] += 1
            return data

        with patch.object(verify, "ROOT", PAPER), patch.object(verify, "RESULTS", RESULTS), \
             patch.object(verify, "load_json", changed):
            with self.assertRaisesRegex(RuntimeError, "table timing disagrees"):
                verify.audit_results()

    def test_claimed_validity_is_recomputed(self):
        records = json.loads(CELL.read_text())
        mg = next(item for item in records if item["framework"] == "meganeura")
        pt = next(item for item in records if item["framework"] == "pytorch")
        mg["validation"]["training_valid"] = False
        with patch.object(verify, "RESULTS", RESULTS):
            with self.assertRaisesRegex(RuntimeError, "validity gates disagree"):
                verify.validate_pair(mg, pt, CELL)

    def test_invalid_forward_gets_zero_portability(self):
        spec = importlib.util.spec_from_file_location("mktables", PAPER / "mktables.py")
        tables = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tables)
        original = tables.load

        def changed(platform, mode, model):
            data = copy.deepcopy(original(platform, mode, model))
            if platform == "nvidia":
                data["meganeura"]["validation"]["forward_valid"] = False
            return data

        with patch.object(tables, "load", changed):
            self.assertEqual(tables.pennycook("strict", "inference_ms", "SmolLM2-135M")["meganeura"], 0)

    def test_manifest_rejects_path_outside_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MANIFEST.sha256").write_text("0" * 64 + "  ../outside\n")
            with patch.object(verify, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "escapes bundle"):
                    verify.verify_manifest()


if __name__ == "__main__":
    unittest.main()
