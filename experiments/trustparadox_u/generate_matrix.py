"""Experiment matrix generator for paired runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatrixEntry:
    run_id: str
    scenario_id: str
    trust_level: str
    attack_type: str
    firewall_variant: str
    seed: int
    config_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "trust_level": self.trust_level,
            "attack_type": self.attack_type,
            "firewall_variant": self.firewall_variant,
            "seed": self.seed,
            "config_path": self.config_path,
        }


def generate_matrix(
    scenarios: list[str],
    trust_levels: list[str],
    attack_types: list[str],
    firewall_variants: list[str],
    seeds: list[int],
    config_dir: str = "experiments/trustparadox_u/configs",
) -> list[MatrixEntry]:
    entries: list[MatrixEntry] = []
    seen_ids: set[str] = set()

    for scenario in sorted(scenarios):
        for trust in sorted(trust_levels):
            for attack in sorted(attack_types):
                for variant in sorted(firewall_variants):
                    for seed in seeds:
                        run_id = f"{scenario}_{trust}_{attack}_{variant}_{seed}"
                        if run_id in seen_ids:
                            raise ValueError(f"Duplicate run_id: {run_id}")
                        seen_ids.add(run_id)
                        entries.append(MatrixEntry(
                            run_id=run_id,
                            scenario_id=scenario,
                            trust_level=trust,
                            attack_type=attack,
                            firewall_variant=variant,
                            seed=seed,
                            config_path=f"{config_dir}/{variant}.yaml",
                        ))
    return entries


def write_matrix(entries: list[MatrixEntry], output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scenarios = ["credential_001", "attribute_001", "auth_001"]
    trust_levels = ["low", "default", "high"]
    attack_types = ["direct", "alias", "paraphrase", "temporal_fragmentation"]
    variants = ["no_firewall", "exact_only", "full_mvp"]
    seeds = [42, 43, 44, 45, 46]

    entries = generate_matrix(scenarios, trust_levels, attack_types, variants, seeds)
    write_matrix(entries, args.output)
    print(f"Generated {len(entries)} experiment runs")
