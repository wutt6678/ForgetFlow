"""Tests for dataset loading."""

from pathlib import Path

from experiments.trustparadox_u.dataset import load_split

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
SPLITS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "splits"


class TestDataset:
    def test_load_split(self) -> None:
        ids = load_split(SPLITS_DIR / "development.jsonl")
        assert len(ids) == 3
        assert "credential_001_high_direct" in ids
