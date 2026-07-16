"""TrustParadox-U dataset loader and episode types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role: str
    public_profile: str
    private_context: tuple[str, ...]


@dataclass(frozen=True)
class RelationshipSpec:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class TaskSpec:
    description: str
    success_label: str


@dataclass(frozen=True)
class SensitiveItemSpec:
    forget_id: str
    target_type: str
    canonical_target: str
    aliases: tuple[str, ...]
    semantic_variants: tuple[str, ...]
    permitted_residuals: tuple[str, ...]
    active_from_turn: int


@dataclass(frozen=True)
class PreForgetPhase:
    sender: str
    recipient: str
    instruction: str
    message_source: str = "generated"


@dataclass(frozen=True)
class ForgetPhase:
    turn: int
    clean_agents: tuple[str, ...]


@dataclass(frozen=True)
class PostForgetAttack:
    attack_type: str
    attacker: str
    target_agent: str
    instruction: str


@dataclass(frozen=True)
class PhasesSpec:
    pre_forget: tuple[PreForgetPhase, ...]
    forget: ForgetPhase
    post_forget: tuple[PostForgetAttack, ...]


@dataclass(frozen=True)
class ExpectedSpec:
    forbidden_strings: tuple[str, ...]
    permitted_strings: tuple[str, ...]


@dataclass(frozen=True)
class TrustParadoxEpisode:
    episode_id: str
    scenario_id: str
    macro_scene: str
    trust_level: str
    agents: tuple[AgentSpec, ...]
    relationships: tuple[RelationshipSpec, ...]
    task: TaskSpec
    sensitive_items: tuple[SensitiveItemSpec, ...]
    phases: PhasesSpec
    expected: ExpectedSpec
    fragment_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    fact_chains: tuple[tuple[tuple[str, str, str], ...], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def agent_ids(self) -> set[str]:
        return {a.agent_id for a in self.agents}

    def get_agent(self, agent_id: str) -> AgentSpec:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        raise ValueError(f"Unknown agent: {agent_id}")


_SCHEMA_PATH = (
    Path(__file__).parents[2] / "data" / "trustparadox_u" / "schema" / "episode.schema.json"
)


def _load_schema() -> dict[str, Any]:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _build_episode(raw: dict[str, Any]) -> TrustParadoxEpisode:
    agents = tuple(
        AgentSpec(
            agent_id=a["agent_id"],
            role=a["role"],
            public_profile=a["public_profile"],
            private_context=tuple(a.get("private_context", [])),
        )
        for a in raw["agents"]
    )
    agent_id_set = {a.agent_id for a in agents}

    relationships = tuple(
        RelationshipSpec(source=r["source"], target=r["target"], relation=r["relation"])
        for r in raw["relationships"]
    )

    task = TaskSpec(
        description=raw["task"]["description"],
        success_label=raw["task"]["success_label"],
    )

    sensitive_items = tuple(
        SensitiveItemSpec(
            forget_id=s["forget_id"],
            target_type=s["target_type"],
            canonical_target=s["canonical_target"],
            aliases=tuple(s.get("aliases", [])),
            semantic_variants=tuple(s.get("semantic_variants", [])),
            permitted_residuals=tuple(s.get("permitted_residuals", [])),
            active_from_turn=s["active_from_turn"],
        )
        for s in raw["sensitive_items"]
    )

    # Check duplicate forget IDs
    forget_ids = [s.forget_id for s in sensitive_items]
    if len(forget_ids) != len(set(forget_ids)):
        raise ValueError(f"Duplicate forget_ids in episode {raw['episode_id']}")

    pre_forget = tuple(
        PreForgetPhase(
            sender=p["sender"],
            recipient=p["recipient"],
            instruction=p["instruction"],
            message_source=p.get("message_source", "generated"),
        )
        for p in raw["phases"]["pre_forget"]
    )
    forget = ForgetPhase(
        turn=raw["phases"]["forget"]["turn"],
        clean_agents=tuple(raw["phases"]["forget"]["clean_agents"]),
    )
    post_forget = tuple(
        PostForgetAttack(
            attack_type=p["attack_type"],
            attacker=p["attacker"],
            target_agent=p["target_agent"],
            instruction=p["instruction"],
        )
        for p in raw["phases"]["post_forget"]
    )
    phases = PhasesSpec(pre_forget=pre_forget, forget=forget, post_forget=post_forget)

    expected = ExpectedSpec(
        forbidden_strings=tuple(raw["expected"]["forbidden_strings"]),
        permitted_strings=tuple(raw["expected"]["permitted_strings"]),
    )

    # Validate agent references
    for pf in pre_forget:
        if pf.sender not in agent_id_set:
            raise ValueError(f"Unknown sender '{pf.sender}' in pre_forget")
        if pf.recipient not in agent_id_set:
            raise ValueError(f"Unknown recipient '{pf.recipient}' in pre_forget")
    for ca in forget.clean_agents:
        if ca not in agent_id_set:
            raise ValueError(f"Unknown clean_agent '{ca}'")
    for atk in post_forget:
        if atk.attacker not in agent_id_set:
            raise ValueError(f"Unknown attacker '{atk.attacker}'")
        if atk.target_agent not in agent_id_set:
            raise ValueError(f"Unknown target_agent '{atk.target_agent}'")

    fragment_map = raw.get("fragment_map", {})
    raw_facts = raw.get("fact_chains", [])
    fact_chains = tuple(tuple(tuple(triple) for triple in chain) for chain in raw_facts)

    return TrustParadoxEpisode(
        episode_id=raw["episode_id"],
        scenario_id=raw["scenario_id"],
        macro_scene=raw["macro_scene"],
        trust_level=raw["trust_level"],
        agents=agents,
        relationships=relationships,
        task=task,
        sensitive_items=sensitive_items,
        phases=phases,
        expected=expected,
        fragment_map=fragment_map,
        fact_chains=fact_chains,
        metadata={
            k: v
            for k, v in raw.items()
            if k
            not in {
                "episode_id",
                "scenario_id",
                "macro_scene",
                "trust_level",
                "agents",
                "relationships",
                "task",
                "sensitive_items",
                "phases",
                "expected",
                "fragment_map",
                "fact_chains",
            }
        },
    )


def load_episode(path: str | Path) -> TrustParadoxEpisode:
    """Load and validate a single episode from YAML."""
    p = Path(path)
    with open(p) as f:
        raw = yaml.safe_load(f)
    schema = _load_schema()
    jsonschema.validate(instance=raw, schema=schema)
    return _build_episode(raw)


def load_episodes_from_dir(directory: str | Path) -> list[TrustParadoxEpisode]:
    """Load all YAML episodes from a directory."""
    d = Path(directory)
    episodes = []
    for fp in sorted(d.glob("*.yaml")):
        episodes.append(load_episode(fp))
    return episodes


def load_split(path: str | Path) -> list[str]:
    """Load episode IDs from a JSONL split file."""
    p = Path(path)
    ids = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                ids.append(entry["episode_id"])
    return ids
