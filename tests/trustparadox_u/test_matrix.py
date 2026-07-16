"""Tests for experiment matrix generator."""

from experiments.trustparadox_u.generate_matrix import generate_matrix


class TestMatrix:
    def test_deterministic(self) -> None:
        m1 = generate_matrix(["s1"], ["high"], ["direct"], ["full_mvp"], [42])
        m2 = generate_matrix(["s1"], ["high"], ["direct"], ["full_mvp"], [42])
        assert m1 == m2

    def test_no_duplicate_ids(self) -> None:
        entries = generate_matrix(
            ["s1", "s2"],
            ["low", "high"],
            ["direct", "alias"],
            ["full_mvp", "no_firewall"],
            [42, 43],
        )
        ids = [e.run_id for e in entries]
        assert len(ids) == len(set(ids))

    def test_correct_size(self) -> None:
        entries = generate_matrix(
            ["s1", "s2"],
            ["low", "high"],
            ["direct"],
            ["full_mvp"],
            [42],
        )
        assert len(entries) == 2 * 2 * 1 * 1 * 1

    def test_config_path(self) -> None:
        entries = generate_matrix(["s1"], ["high"], ["direct"], ["full_mvp"], [42])
        assert "full_mvp.yaml" in entries[0].config_path
