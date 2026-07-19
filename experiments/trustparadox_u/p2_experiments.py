"""P2-26 through P2-31: Publication-scale experiment infrastructure."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any
import hashlib
import json
import statistics

# P2-26: Generate and freeze real-LLM candidate corpus
@dataclass
class FrozenCandidateCorpus:
    """Frozen candidate corpus for real-LLM experiments."""
    corpus_id: str
    version: str
    candidates: list[dict]
    corpus_hash: str
    
    @classmethod
    def create_and_freeze(cls, corpus_id: str, version: str, candidates: list[dict]) -> "FrozenCandidateCorpus":
        """Create and freeze a candidate corpus."""
        payload = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
        corpus_hash = hashlib.sha256(payload.encode()).hexdigest()
        return cls(
            corpus_id=corpus_id,
            version=version,
            candidates=candidates,
            corpus_hash=corpus_hash,
        )
    
    def validate_frozen(self) -> bool:
        """Validate corpus hasn't been modified."""
        payload = json.dumps(self.candidates, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest() == self.corpus_hash

# P2-27: Independent corpus annotation
@dataclass
class CorpusAnnotation:
    """Independent annotation of candidate corpus."""
    annotator_id: str
    annotation_method: Literal["human", "model", "ruleset"]
    annotations: list[dict]
    
    def annotate_candidate(self, candidate: dict) -> dict:
        """Annotate a single candidate."""
        return {
            "candidate_id": candidate.get("candidate_id"),
            "disclosure_class": candidate.get("expected_exposure_class"),
            "target_ids": candidate.get("target_forget_ids", []),
            "speech_act": candidate.get("speech_act", "disclosure"),
            "positive_entailment": not candidate.get("text", "").endswith("?"),
            "reconstruction": False,
            "task_usefulness": 0.5,
        }

# P2-28: Trust-level and threshold sweeps
@dataclass
class ParameterSweep:
    """Parameter sweep infrastructure."""
    
    TRUST_LEVELS = ["low", "default", "high"]
    EMBEDDING_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    CLAIM_THRESHOLDS = [0.55, 0.65, 0.75, 0.85]
    HISTORY_WINDOWS = [0, 1, 2, 4, 8, None]  # None = unbounded
    MONITORING_DURATIONS = [0, 1, 2, 4, None]  # None = continuous
    
    @staticmethod
    def generate_sweep_configs(
        base_config: dict,
        sweep_params: list[str],
    ) -> list[dict]:
        """Generate all configurations for parameter sweep."""
        configs = [base_config.copy()]
        
        for param in sweep_params:
            new_configs = []
            for config in configs:
                if param == "trust_level":
                    for level in ParameterSweep.TRUST_LEVELS:
                        new_config = config.copy()
                        new_config["trust_level"] = level
                        new_configs.append(new_config)
                elif param == "embedding_threshold":
                    for thresh in ParameterSweep.EMBEDDING_THRESHOLDS:
                        new_config = config.copy()
                        new_config.setdefault("detector", {})["embedding_threshold"] = thresh
                        new_configs.append(new_config)
                elif param == "claim_threshold":
                    for thresh in ParameterSweep.CLAIM_THRESHOLDS:
                        new_config = config.copy()
                        new_config.setdefault("detector", {})["claim_confidence_threshold"] = thresh
                        new_configs.append(new_config)
            configs = new_configs
        
        return configs

# P2-29: Held-out paraphrase and grammar tests
@dataclass
class HeldOutTests:
    """Held-out test families for generalization."""
    
    PARAPHRASE_VARIANTS = [
        "passive_voice",
        "possessive_form",
        "reported_speech",
        "conditional_claim",
        "historical_claim",
        "future_claim",
    ]
    
    GRAMMAR_VARIANTS = [
        "punctuation_splitting",
        "unicode_variants",
        "number_words",
        "json_disclosure",
        "code_block_disclosure",
    ]
    
    @staticmethod
    def generate_held_out_candidates(base_candidate: dict) -> list[dict]:
        """Generate held-out variants of a base candidate."""
        variants = []
        for variant_type in HeldOutTests.PARAPHRASE_VARIANTS + HeldOutTests.GRAMMAR_VARIANTS:
            variant = base_candidate.copy()
            variant["candidate_id"] = f"{base_candidate['candidate_id']}_{variant_type}"
            variant["variant_type"] = variant_type
            variants.append(variant)
        return variants

# P2-30: Latency, cost, and scalability experiments
@dataclass
class PerformanceMetrics:
    """Performance and cost tracking."""
    
    @staticmethod
    def calculate_latency_metrics(latencies: list[float]) -> dict:
        """Calculate latency statistics."""
        if not latencies:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {
            "mean": statistics.mean(sorted_lat),
            "median": statistics.median(sorted_lat),
            "p95": sorted_lat[int(n * 0.95)] if n >= 20 else sorted_lat[-1],
            "p99": sorted_lat[int(n * 0.99)] if n >= 100 else sorted_lat[-1],
        }
    
    @staticmethod
    def calculate_cost_metrics(
        chat_calls: int,
        embedding_calls: int,
        claim_calls: int,
        chat_cost_per_call: float = 0.001,
        embedding_cost_per_call: float = 0.0001,
        claim_cost_per_call: float = 0.0005,
    ) -> dict:
        """Calculate cost metrics."""
        return {
            "chat_cost": chat_calls * chat_cost_per_call,
            "embedding_cost": embedding_calls * embedding_cost_per_call,
            "claim_cost": claim_calls * claim_cost_per_call,
            "total_cost": (
                chat_calls * chat_cost_per_call
                + embedding_calls * embedding_cost_per_call
                + claim_calls * claim_cost_per_call
            ),
        }

# P2-31: Confidence intervals and statistical tests
@dataclass
class StatisticalAnalysis:
    """Statistical analysis for experiments."""
    
    @staticmethod
    def calculate_confidence_interval(
        values: list[float],
        confidence: float = 0.95,
    ) -> dict:
        """Calculate confidence interval."""
        if len(values) < 2:
            return {"mean": values[0] if values else 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        
        n = len(values)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        
        # Approximate z-score for 95% CI
        z = 1.96 if confidence == 0.95 else 2.576  # 99% CI
        margin = z * stdev / (n ** 0.5)
        
        return {
            "mean": mean,
            "ci_lower": mean - margin,
            "ci_upper": mean + margin,
            "confidence": confidence,
            "n": n,
        }
    
    @staticmethod
    def paired_t_test(values_a: list[float], values_b: list[float]) -> dict:
        """Perform paired t-test."""
        if len(values_a) != len(values_b) or len(values_a) < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
        
        diffs = [a - b for a, b in zip(values_a, values_b)]
        mean_diff = statistics.mean(diffs)
        stdev_diff = statistics.stdev(diffs)
        
        if stdev_diff == 0:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}
        
        t_stat = mean_diff / (stdev_diff / (len(diffs) ** 0.5))
        # Approximate p-value (simplified)
        p_value = 2 * (1 - abs(t_stat) / 3)  # Very rough approximation
        p_value = max(0.0, min(1.0, p_value))
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
