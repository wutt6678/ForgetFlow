"""TrustParadox-U dataset loader and episode types."""

from __future__ import annotations

import hashlib
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
    success_type: str = "exact_label"  # "required_release" or "exact_label"
    success_value: str = ""


@dataclass(frozen=True)
class SensitiveItemSpec:
    forget_id: str
    target_type: str
    canonical_target: str
    aliases: tuple[str, ...]
    semantic_variants: tuple[str, ...]
    permitted_residuals: tuple[str, ...]
    active_from_turn: int
    reconstruction: dict[str, Any] = field(default_factory=dict)
    secret_variant_id: str = ""

    def __post_init__(self) -> None:
        if not self.secret_variant_id:
            # Generate stable ID from scenario-relevant fields
            # Note: caller should set this after construction with scenario context
            pass


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
class MessageLabel:
    is_attack_attempt: bool = False
    is_legitimate_message: bool = False
    is_reconstruction_attempt: bool = False
    is_recontamination_attempt: bool = False
    task_relevant: bool = False
    target_forget_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PostForgetAttack:
    attack_type: str
    attacker: str
    target_agent: str
    instruction: str
    label: MessageLabel = field(default_factory=MessageLabel)


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
    fact_chain_map: dict[str, list[list[tuple[str, str, str]]]] = field(default_factory=dict)
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


_RECONSTRUCTION_ATTACK_TYPES = {
    "temporal_fragmentation",
    "cross_agent_fragmentation",
    "compositional_inference",
}


def _parse_message_label(raw_label: dict[str, Any], attack_type: str) -> MessageLabel:
    """Parse message label from YAML, inferring defaults from attack type."""
    if raw_label:
        # Parse and normalize target_forget_ids
        target_forget_ids: tuple[str, ...] = ()
        if "target_forget_ids" in raw_label:
            raw_ids = raw_label["target_forget_ids"]
            if not isinstance(raw_ids, list):
                raise ValueError("target_forget_ids must be a list")
            # Normalize: convert to strings, remove duplicates, sort for determinism
            normalized_ids = tuple(
                sorted(set(str(id_val) for id_val in raw_ids if str(id_val).strip()))
            )
            target_forget_ids = normalized_ids

        is_recontamination = raw_label.get(
            "is_recontamination_attempt", attack_type == "recontamination"
        )

        # Validate: recontamination attempts must have target_forget_ids
        if is_recontamination and not target_forget_ids:
            raise ValueError("Recontamination attempts must specify target_forget_ids")

        return MessageLabel(
            is_attack_attempt=raw_label.get("is_attack_attempt", True),
            is_legitimate_message=raw_label.get("is_legitimate_message", False),
            is_reconstruction_attempt=raw_label.get(
                "is_reconstruction_attempt", attack_type in _RECONSTRUCTION_ATTACK_TYPES
            ),
            is_recontamination_attempt=is_recontamination,
            task_relevant=raw_label.get("task_relevant", False),
            target_forget_ids=target_forget_ids,
        )
    # Default: all post_forget attacks are attack attempts
    is_recontamination = attack_type == "recontamination"
    if is_recontamination:
        raise ValueError("Recontamination attempts must specify target_forget_ids")
    return MessageLabel(
        is_attack_attempt=True,
        is_legitimate_message=False,
        is_reconstruction_attempt=attack_type in _RECONSTRUCTION_ATTACK_TYPES,
        is_recontamination_attempt=False,
        task_relevant=False,
        target_forget_ids=(),
    )


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
        success_type=raw["task"].get("success_type", "exact_label"),
        success_value=raw["task"].get("success_value", ""),
    )

    scenario_id = raw["scenario_id"]
    sensitive_items_list = []
    for s in raw["sensitive_items"]:
        variant_id = s.get("secret_variant_id", "")
        if not variant_id:
            # Generate stable ID from scenario-relevant fields
            payload = f"{scenario_id}|{s['forget_id']}|{s['canonical_target']}"
            variant_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
        sensitive_items_list.append(
            SensitiveItemSpec(
                forget_id=s["forget_id"],
                target_type=s["target_type"],
                canonical_target=s["canonical_target"],
                aliases=tuple(s.get("aliases", [])),
                semantic_variants=tuple(s.get("semantic_variants", [])),
                permitted_residuals=tuple(s.get("permitted_residuals", [])),
                active_from_turn=s["active_from_turn"],
                reconstruction=s.get("reconstruction", {}),
                secret_variant_id=variant_id,
            )
        )
    sensitive_items = tuple(sensitive_items_list)

    # Check duplicate forget IDs
    forget_ids = [s.forget_id for s in sensitive_items]
    if len(forget_ids) != len(set(forget_ids)):
        raise ValueError(f"Duplicate forget_ids in episode {raw['episode_id']}")

    # s3 (20th): Reject cross-record representation collisions
    validate_representation_ownership(sensitive_items)

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
            label=_parse_message_label(p.get("label", {}), p["attack_type"]),
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

    # Validate attack target references against sensitive items
    valid_forget_ids = {item.forget_id for item in sensitive_items}
    for atk in post_forget:
        if atk.label.is_recontamination_attempt and not atk.label.target_forget_ids:
            raise ValueError(
                f"Recontamination attempt in episode {raw['episode_id']} "
                "requires non-empty target_forget_ids"
            )
        unknown = set(atk.label.target_forget_ids) - valid_forget_ids
        if unknown:
            raise ValueError(
                f"Unknown target_forget_ids in episode {raw['episode_id']}: " f"{sorted(unknown)}"
            )

    fragment_map = raw.get("fragment_map", {})
    raw_facts = raw.get("fact_chains", [])
    fact_chains = tuple(tuple(tuple(triple) for triple in chain) for chain in raw_facts)

    # Parse fact_chain_map (keyed by forget_id) for record-specific fact-chain reconstruction
    raw_fcm = raw.get("fact_chain_map", {})
    fact_chain_map: dict[str, list[list[tuple[str, str, str]]]] = {}
    for fid, chains in raw_fcm.items():
        if not isinstance(fid, str) or not fid:
            raise ValueError(f"fact_chain_map key must be a non-empty string, got {fid!r}")
        parsed_chains = []
        for chain in chains:
            parsed_chain = [tuple(triple) for triple in chain]
            parsed_chains.append(parsed_chain)
        fact_chain_map[fid] = parsed_chains

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
        fact_chain_map=fact_chain_map,
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
                "fact_chain_map",
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


def validate_single_target_episode(episode: TrustParadoxEpisode) -> None:
    """Validate that an episode has exactly one sensitive item.

    Raises ValueError if the episode does not meet single-target requirements.
    """
    count = len(episode.sensitive_items)
    if count != 1:
        raise ValueError(
            f"Single-target episodes require exactly one sensitive item; found {count}"
        )


def load_single_target_episode(path: str | Path) -> TrustParadoxEpisode:
    """Load an episode and validate it has exactly one sensitive item."""
    episode = load_episode(path)
    validate_single_target_episode(episode)
    return episode


def validate_attack_target_references(episode: TrustParadoxEpisode) -> None:
    """Validate that recontamination attacks reference valid episode forget_ids.

    Checks:
    - Recontamination attempts must have non-empty target_forget_ids
    - All target_forget_ids must exist in the episode's sensitive items

    Raises ValueError if validation fails.
    """
    valid_ids = {item.forget_id for item in episode.sensitive_items}

    for attack in episode.phases.post_forget:
        targets = attack.label.target_forget_ids

        if attack.label.is_recontamination_attempt and not targets:
            raise ValueError("Recontamination attempts require non-empty target_forget_ids")

        unknown = set(targets) - valid_ids
        if unknown:
            raise ValueError(f"Unknown target_forget_ids: {sorted(unknown)}")


def validate_representation_ownership(sensitive_items: tuple[SensitiveItemSpec, ...]) -> None:
    """Reject episodes where normalized representations map to multiple forget IDs.

    Every normalized sensitive representation must have exactly one owning
    forget_id.  Collisions (shared aliases, case-only duplicates,
    canonical-to-alias collisions, etc.) make targeted forgetting ambiguous.

    Raises ValueError listing all ambiguous representations.
    """
    owners: dict[str, set[str]] = {}
    for item in sensitive_items:
        values = {
            item.canonical_target,
            *item.aliases,
            *item.semantic_variants,
        }
        for value in values:
            normalized = value.strip().casefold()
            if not normalized:
                continue
            owners.setdefault(normalized, set()).add(item.forget_id)

    ambiguous = {
        representation: sorted(forget_ids)
        for representation, forget_ids in owners.items()
        if len(forget_ids) > 1
    }
    if ambiguous:
        raise ValueError(
            "Sensitive representations must map to exactly one forget record: " f"{ambiguous}"
        )


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
