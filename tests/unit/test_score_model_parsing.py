"""Unit tests for scripts/evaluation/score_model.py's argument parsing and case-discovery
plumbing -- deliberately does NOT invoke real scoring (that needs topometrics installed in
the vesuvius_eval environment, see environment-eval.yml), only the pure logic around it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "score_model", REPO_ROOT / "scripts" / "evaluation" / "score_model.py"
)
score_model = importlib.util.module_from_spec(_spec)
sys.modules["score_model"] = score_model
_spec.loader.exec_module(score_model)


class TestParseArgs:
    def test_single_condition(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "score_model.py", "--gt-dir", "gt", "--pred-dir", "baseline=preds/baseline",
        ])
        args = score_model.parse_args()
        assert args.pred_dir == ["baseline=preds/baseline"]
        assert args.postprocess == []
        assert args.workers == 8  # documented default -- proven-safe ceiling, see docstring

    def test_multiple_conditions_and_postprocess_flags(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "score_model.py", "--gt-dir", "gt",
            "--pred-dir", "zero_shot=preds/a",
            "--pred-dir", "finetuned=preds/b",
            "--postprocess", "finetuned",
        ])
        args = score_model.parse_args()
        assert len(args.pred_dir) == 2
        assert args.postprocess == ["finetuned"]


class TestMainValidation:
    def test_rejects_malformed_pred_dir_spec(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "score_model.py", "--gt-dir", "gt", "--pred-dir", "not-a-key-value-pair",
        ])
        rc = score_model.main()
        assert rc == 1
        assert "NAME=PATH" in capsys.readouterr().err

    def test_rejects_postprocess_name_not_in_pred_dirs(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "score_model.py", "--gt-dir", "gt",
            "--pred-dir", "baseline=preds/baseline",
            "--postprocess", "typo_name",
        ])
        rc = score_model.main()
        assert rc == 1
        assert "typo_name" in capsys.readouterr().err


class TestDiscoverCaseIds:
    def test_discovers_both_tif_and_npz_stems(self, tmp_path):
        (tmp_path / "case_a.tif").touch()
        (tmp_path / "case_b.npz").touch()
        (tmp_path / "case_a.json").touch()  # sidecar, must be ignored

        ids = score_model.discover_case_ids(tmp_path)

        assert ids == ["case_a", "case_b"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
