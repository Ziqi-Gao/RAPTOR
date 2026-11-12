"""Frozen data contracts for the two-stage, blinded six-metric judge."""
from __future__ import annotations

import json

METHODS = ("original", "raptor", "gcs", "fixed_reg", "dim_gcav")
METRICS_A = ("target_behavior", "output_quality", "naturalness", "overall_success")
METRICS_B = ("content_preservation", "semantic_drift")
METRICS = METRICS_A + METRICS_B
TARGETS = {
    ("STSA", "neutral_to_negative"): "negative sentiment",
    ("counterfact", "lie_to_truth"): "truthful statement",
    ("counterfact", "truth_to_lie"): "false statement",
}
PAIR_FIELDS = ("model_id", "task", "direction", "prompt_id", "generation_seed")
SHUFFLE_SEED = 20261104


class ProtocolError(ValueError):
    """Invalid input, identity, or judge response; never a scientific score."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def bad_constant(value):
        raise ProtocolError("non-finite JSON constant: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def response_schema(stage):
    metric = obj({
        "score": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
        "applicability": {"type": "string", "enum": ["applicable", "not_applicable", "unverifiable"]},
        "null_reason": {"type": ["string", "null"]},
    })
    metrics = METRICS_A if stage == "A" else METRICS_B
    fields = {
        "record_id": {"type": "string"},
        "stage": {"type": "string", "enum": [stage]},
        "scores": obj({name: metric for name in metrics}),
    }
    if stage == "A":
        fields.update({
            "factual_status": {"type": "string", "enum": [
                "true", "false", "mixed", "unverifiable", "nonfactual", "not_applicable"]},
            "factual_reason": {"type": "string"},
        })
    else:
        fields.update({
            "reference_comparability": {"type": "string", "enum": ["high", "limited", "none"]},
            "reference_comparability_reason": {"type": "string"},
        })
    return obj(fields)


def validate_response(value, unit):
    """Enforce both schema and frozen applicability/target rules locally."""
    stage = unit["stage"]
    schema = response_schema(stage)
    require(isinstance(value, dict) and set(value) == set(schema["properties"]), "response fields")
    require(value["record_id"] == unit["record_id"] and value["stage"] == stage, "response identity")
    names = METRICS_A if stage == "A" else METRICS_B
    require(isinstance(value["scores"], dict) and set(value["scores"]) == set(names), "metric fields")
    for name in names:
        item = value["scores"][name]
        require(isinstance(item, dict) and set(item) == {"score", "reason", "applicability", "null_reason"}, "metric contract")
        require(isinstance(item["reason"], str) and item["reason"].strip(), "missing metric reason")
        score = item["score"]
        if score is None:
            require(item["applicability"] in ("not_applicable", "unverifiable"), "null applicability")
            require(isinstance(item["null_reason"], str) and item["null_reason"].strip(), "missing null reason")
        else:
            require(type(score) is int and 1 <= score <= 5, "integer score range")
            require(item["applicability"] == "applicable" and item["null_reason"] is None, "numeric applicability")
    if stage == "A":
        factual = value["factual_status"]
        require(factual in schema["properties"]["factual_status"]["enum"], "factual status")
        require(isinstance(value["factual_reason"], str) and value["factual_reason"].strip(), "factual reason")
        cf = unit["task"] == "counterfact"
        require((factual != "not_applicable") == cf, "factual task applicability")
        for name in names:
            missing = cf and factual == "unverifiable" and name in ("target_behavior", "overall_success")
            item = value["scores"][name]
            require((item["score"] is None) == missing, "invalid independent metric null")
            if missing:
                require(item["applicability"] == "unverifiable", "unverifiable applicability")
        target = value["scores"]["target_behavior"]["score"]
        overall = value["scores"]["overall_success"]["score"]
        if target is not None and target <= 2:
            require(overall <= 3, "fluent target failure cannot have high overall success")
        if cf and factual == "nonfactual":
            require(target == 1 and overall <= 2, "nonfactual output fails factual target")
        opposite = cf and ((factual == "true" and unit["direction"] == "truth_to_lie") or
                           (factual == "false" and unit["direction"] == "lie_to_truth"))
        if opposite:
            require(target <= 2, "opposite factual status fails target")
    else:
        comparability = value["reference_comparability"]
        require(comparability in ("high", "limited", "none"), "reference comparability")
        require(isinstance(value["reference_comparability_reason"], str) and value["reference_comparability_reason"].strip(), "comparability reason")
        for name in names:
            item = value["scores"][name]
            require((item["score"] is None) == (comparability == "none"), "reference null rule")
            if comparability == "none":
                require(item["null_reason"] == "no_comparable_reference_semantics" and
                        item["applicability"] == "not_applicable", "reference null reason")
    return value
