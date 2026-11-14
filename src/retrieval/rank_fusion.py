"""Deterministic reciprocal-rank fusion for controlled retrieval comparisons."""

import math


def reciprocal_rank_scores(rankings, *, constant=60, weights=None):
    if isinstance(constant, bool) or not math.isfinite(constant) or constant <= 0:
        raise ValueError("RRF constant must be finite and positive")
    weights = weights or {}
    if set(weights) - set(rankings):
        raise ValueError("Weight supplied for an unknown ranking")
    scores = {}
    for source, ranking in rankings.items():
        weight = weights.get(source, 1.0)
        if isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Ranking weights must be finite and positive")
        for index, rank in ranking.items():
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("Ranks must be positive integers")
            scores[index] = scores.get(index, 0.0) + weight / (constant + rank)
    return scores


def fused_order(rankings, *, constant=60, weights=None):
    scores = reciprocal_rank_scores(rankings, constant=constant, weights=weights)
    return sorted(scores, key=lambda index: (-scores[index], index))
