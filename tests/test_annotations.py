"""Tests for annotation storage."""
from __future__ import annotations

from pathlib import Path

import pytest

from deal_memory import annotations as ann_mod


def test_upsert_creates_file_and_returns(tmp_path: Path):
    ann = ann_mod.upsert("p1", "d1", "budget", "correct", base=tmp_path)
    assert ann.verdict == "correct"
    assert (tmp_path / "p1.jsonl").exists()


def test_upsert_overwrites_same_cell(tmp_path: Path):
    ann_mod.upsert("p1", "d1", "budget", "correct", base=tmp_path)
    ann_mod.upsert("p1", "d1", "budget", "wrong", base=tmp_path)

    loaded = ann_mod.load_pool("p1", base=tmp_path)
    assert loaded[("d1", "budget")].verdict == "wrong"
    assert len(loaded) == 1  # not appended


def test_clear_removes_cell_and_file_when_empty(tmp_path: Path):
    ann_mod.upsert("p1", "d1", "budget", "correct", base=tmp_path)
    ann_mod.clear("p1", "d1", "budget", base=tmp_path)
    assert not (tmp_path / "p1.jsonl").exists()


def test_unknown_field_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown field"):
        ann_mod.upsert("p1", "d1", "not_a_field", "correct", base=tmp_path)
