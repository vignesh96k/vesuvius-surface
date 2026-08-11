"""Unit tests asserting the split-generation functions in scripts/make_scroll_split.py never
leak a scroll or a case across folds -- directly on-theme for this project, whose whole
validation-protocol design exists because of a real, hard-won lesson about split leakage.
This is the single test most likely to catch a regression that actually matters: every
reported LOSO/local number in this project depends on these functions producing a genuine,
scroll-disjoint split.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "make_scroll_split", REPO_ROOT / "scripts" / "make_scroll_split.py"
)
make_scroll_split = importlib.util.module_from_spec(_spec)
sys.modules["make_scroll_split"] = make_scroll_split
_spec.loader.exec_module(make_scroll_split)


def _synthetic_dataset(n_scrolls: int = 6, cases_per_scroll: int = 20) -> tuple[list[str], dict[str, str]]:
    """Mimics this project's real, very uneven scroll sizes (one scroll has 376 cases,
    another has 13) by giving each scroll a different count."""
    case_ids: list[str] = []
    scroll_map: dict[str, str] = {}
    for s in range(n_scrolls):
        count = cases_per_scroll * (s + 1)  # deliberately uneven, like the real dataset
        for c in range(count):
            cid = f"scroll{s}_case{c}"
            case_ids.append(cid)
            scroll_map[cid] = f"scroll{s}"
    return case_ids, scroll_map


class TestScrollHoldoutFolds:
    def test_every_fold_is_a_single_whole_scroll(self):
        case_ids, scroll_map = _synthetic_dataset()
        folds, scrolls = make_scroll_split.scroll_holdout_folds(case_ids, scroll_map)

        assert len(folds) == len(scrolls) == 6
        for fold, scroll in zip(folds, scrolls):
            assert all(scroll_map[cid] == scroll for cid in fold), (
                f"fold for {scroll} contains cases from another scroll"
            )

    def test_folds_partition_every_case_exactly_once(self):
        case_ids, scroll_map = _synthetic_dataset()
        folds, _ = make_scroll_split.scroll_holdout_folds(case_ids, scroll_map)

        all_in_folds = [cid for fold in folds for cid in fold]
        assert sorted(all_in_folds) == sorted(case_ids)
        assert len(all_in_folds) == len(set(all_in_folds)), "a case appears in more than one fold"


class TestNamedScrollHoldout:
    def test_holdout_set_contains_only_the_named_scroll(self):
        case_ids, scroll_map = _synthetic_dataset()
        val = make_scroll_split.named_scroll_holdout(case_ids, scroll_map, ["scroll2"])

        assert len(val) > 0
        assert all(scroll_map[cid] == "scroll2" for cid in val)

    def test_train_and_val_are_disjoint(self):
        """The actual leakage guarantee this project's LOSO protocol depends on: every case
        held out for validation must be absent from what a caller would use as the train set
        (all case_ids minus val)."""
        case_ids, scroll_map = _synthetic_dataset()
        val = set(make_scroll_split.named_scroll_holdout(case_ids, scroll_map, ["scroll0", "scroll1"]))
        train = set(case_ids) - val

        assert val.isdisjoint(train)
        assert all(scroll_map[cid] not in {"scroll0", "scroll1"} for cid in train)

    def test_unknown_scroll_id_raises(self):
        case_ids, scroll_map = _synthetic_dataset()
        with pytest.raises(ValueError, match="not in the dataset"):
            make_scroll_split.named_scroll_holdout(case_ids, scroll_map, ["scroll_that_does_not_exist"])


class TestStratifiedFolds:
    def test_every_case_assigned_to_exactly_one_fold(self):
        case_ids, scroll_map = _synthetic_dataset()
        folds = make_scroll_split.stratified_folds(case_ids, scroll_map, n_splits=5, seed=42)

        all_in_folds = [cid for fold in folds for cid in fold]
        assert sorted(all_in_folds) == sorted(case_ids)
        assert len(all_in_folds) == len(set(all_in_folds))

    def test_every_scroll_has_a_share_in_every_fold(self):
        """The whole point of stratified folds vs. scroll-holdout: unlike LOSO, every fold
        should get proportional representation from every scroll, including small ones --
        this fixes the "blind spot" problem (a scroll with zero cases in the holdout hides
        that scroll's failure mode entirely)."""
        case_ids, scroll_map = _synthetic_dataset(n_scrolls=6, cases_per_scroll=20)
        folds = make_scroll_split.stratified_folds(case_ids, scroll_map, n_splits=5, seed=42)

        scrolls = set(scroll_map.values())
        for fold in folds:
            scrolls_present = {scroll_map[cid] for cid in fold}
            assert scrolls_present == scrolls, "a fold is missing at least one scroll entirely"

    def test_deterministic_given_same_seed(self):
        case_ids, scroll_map = _synthetic_dataset()
        folds_a = make_scroll_split.stratified_folds(case_ids, scroll_map, n_splits=3, seed=7)
        folds_b = make_scroll_split.stratified_folds(case_ids, scroll_map, n_splits=3, seed=7)
        assert folds_a == folds_b


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
