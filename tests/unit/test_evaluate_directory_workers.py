"""Regression test for evaluation.harness.evaluate_directory's workers>1 path: it must
produce the exact same CaseScore records as the workers=1 (original, sequential) path.

Mocks score_pair rather than calling the real official metric -- that package
(topometrics) lives only in the eval conda env, not this repo's CI env, so tests/unit/
can't depend on it. The workers>1 code path was additionally verified against the real metric on real
data manually (bit-identical results, sequential vs. 4-way parallel, before it replaced
the sequential run on a live scoring job) -- this test locks in that the *dispatch* logic
(splitting work across a Pool, writing results back in the main process) doesn't silently
drop or reorder cases, independent of which metric implementation is behind it.
"""
from __future__ import annotations

import json

import numpy as np
import tifffile

from vesuvius_surface.evaluation import harness


def _write_case(directory, case_id: str, value: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(directory / f"{case_id}.tif"), np.full((4, 4, 4), value, dtype=np.uint8))


def _fake_score_pair(prediction, label, **kwargs):
    # Deterministic, cheap stand-in: encodes the case via the volume's fill value so each
    # case gets a distinguishable, checkable "score".
    val = int(prediction.flat[0])
    return {
        "score": float(val) / 10,
        "topo_score": float(val),
        "surface_dice": float(val),
        "voi_score": float(val),
        "voi_split": float(val),
        "voi_merge": float(val),
        "topo_f1_dim0": float(val),
        "topo_f1_dim1": float(val),
        "topo_f1_dim2": float(val),
        "n_foreground": float(val),
    }


def _run(tmp_path, workers: int, subdir: str):
    pred_dir = tmp_path / "pred"
    label_dir = tmp_path / "label"
    case_ids = [f"case{i}" for i in range(6)]
    for i, cid in enumerate(case_ids):
        _write_case(pred_dir, cid, i + 1)
        _write_case(label_dir, cid, i + 1)

    results_path = tmp_path / subdir / "results.jsonl"
    scores = harness.evaluate_directory(
        pred_dir, label_dir, results_path,
        case_ids=case_ids, resume=False, progress=False, workers=workers,
    )
    return {s.case_id: s for s in scores}, results_path


class TestEvaluateDirectoryWorkers:
    def test_parallel_matches_sequential(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness, "score_pair", _fake_score_pair)

        seq, seq_path = _run(tmp_path, workers=1, subdir="seq")
        par, par_path = _run(tmp_path, workers=3, subdir="par")

        assert set(seq) == set(par) == {f"case{i}" for i in range(6)}
        for cid in seq:
            s, p = seq[cid], par[cid]
            for field in harness._MEAN_FIELDS + ("n_foreground",):
                assert getattr(s, field) == getattr(p, field), f"{cid}.{field} differs"

        # Every case actually got written to disk (no dropped work under the Pool).
        assert len(seq_path.read_text().splitlines()) == 6
        assert len(par_path.read_text().splitlines()) == 6

    def test_parallel_resume_skips_done_cases(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harness, "score_pair", _fake_score_pair)

        pred_dir = tmp_path / "pred"
        label_dir = tmp_path / "label"
        case_ids = [f"case{i}" for i in range(4)]
        for i, cid in enumerate(case_ids):
            _write_case(pred_dir, cid, i + 1)
            _write_case(label_dir, cid, i + 1)

        results_path = tmp_path / "resume" / "results.jsonl"
        results_path.parent.mkdir(parents=True)
        # Pre-seed 2 of 4 cases as already done, workers=1 path (matches production usage).
        done_score = harness.CaseScore(
            case_id="case0", scroll_id=None, score=0.0, topo_score=0.0, surface_dice=0.0,
            voi_score=0.0, voi_split=0.0, voi_merge=0.0, topo_f1_dim0=0.0, topo_f1_dim1=0.0,
            topo_f1_dim2=0.0, n_foreground=0.0, seconds=0.0,
        )
        with results_path.open("w") as f:
            f.write(json.dumps(harness.asdict(done_score)) + "\n")
            done2 = harness.asdict(done_score)
            done2["case_id"] = "case1"
            f.write(json.dumps(done2) + "\n")

        scores = harness.evaluate_directory(
            pred_dir, label_dir, results_path,
            case_ids=case_ids, resume=True, progress=False, workers=2,
        )
        assert {s.case_id for s in scores} == {"case0", "case1", "case2", "case3"}
        assert len(results_path.read_text().splitlines()) == 4
