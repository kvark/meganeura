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
import cohort
import search_ablation


PAPER = Path(__file__).resolve().parents[2]
RESULTS = PAPER / "results"
CELL = RESULTS / "nvidia/paper-v1-strict/SmolLM2-135M_summary.json"


class EvidenceTests(unittest.TestCase):
    def test_current_cohort_keeps_partial_and_cpu_runs_out_of_aggregates(self):
        campaigns = {device: {"args": {"backend": backend, "eager": backend == "cpu"}}
                     for device, backend in (("full", "cuda"), ("partial", "rocm"), ("oracle", "cpu"))}
        rows = {device: {("strict", model, cohort.primary_condition(c)):
                        {"replicates_" + phase: 3 for phase in cohort.PHASES}
                        for model in cohort.MODELS} for device, c in campaigns.items()}
        rows["partial"]["strict", "SmolLM2-135M", "default-graph1"]["replicates_training"] = 0
        self.assertEqual(cohort.complete_devices(campaigns, rows, "strict"), ["full"])
        self.assertEqual(cohort.complete_devices(campaigns, rows, "accelerated"), [])
        pair = {engine: {"timing_samples_ms": {phase: [1.0] for phase in cohort.PHASES},
                         "timings": {"compile_s": 0.0}, "execution": {"graph_replay": {"phases": {}}}}
                for engine in cohort.ENGINES}
        key = ("strict", "Whisper-tiny", "default-graph1")
        runs = [{"pair": pair, "phases": ("inference",), "diagnostic_errors": None} for _ in range(3)]
        row = cohort.aggregate({key: runs})[key]
        self.assertEqual(row["replicates_inference"], 3)
        self.assertEqual(row["replicates_training"], 0)
        self.assertEqual(row["ratio_inference"], 1)
        self.assertIsNone(row["ratio_training"])
        self.assertIsNone(row["meganeura_training"])
        for run in runs:
            run["diagnostic_errors"] = {"output_relative_l2_error": 0.0}
        row = cohort.aggregate({key: runs})[key]
        self.assertEqual(row["meganeura_training"], 1.0)
        self.assertEqual(row["meganeura_training_replicates"], 3)
        self.assertEqual(row["pytorch_training_replicates"], 0)
        self.assertIsNone(row["pytorch_training"])
        self.assertIsNone(row["ratio_training"])
        self.assertEqual(cohort.portability_score([row], "meganeura", "training"), 1.0)
        self.assertEqual(cohort.portability_score([row], "pytorch", "training"), 0.0)

    def test_ablation_outputs_must_match_the_reference(self):
        outputs = {"output_shape": [1, 4], "logits_sample": [1.0, 2.0, 3.0], "loss": 2.0,
                   "grad_norm": 5.0, "gradient_norms": {"a": 3.0, "b": 4.0}}
        reference = {"outputs": copy.deepcopy(outputs)}
        errors = search_ablation.validate({"outputs": copy.deepcopy(outputs)}, reference)
        self.assertEqual(max(errors.values()), 0.0)
        outputs["gradient_norms"]["b"] = 5.0
        with self.assertRaisesRegex(ValueError, "gradient gate"):
            search_ablation.validate({"outputs": outputs}, reference)
        self.assertAlmostEqual(search_ablation.geomean([1.0, 4.0]), 2.0)

    def test_streamed_records_preserve_raw_summary_identity(self):
        record = {"framework": "pytorch", "status": "ok", "timings": {"inference_ms": 1.0}}
        records = cohort.read_records(iter((("raw.json", copy.deepcopy(record)),
                                           ("summary.json", [copy.deepcopy(record)]))))
        self.assertEqual(records["raw.json"], records["summary.json"][0])
        record["timings"]["inference_ms"] = 2.0
        changed = cohort.read_records(iter((("raw.json", record),)))
        self.assertNotEqual(records["raw.json"]["source_digest"], changed["raw.json"]["source_digest"])

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
