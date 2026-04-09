import pandas as pd
import numpy as np
from typing import Dict
from collections import defaultdict

from data_models import CompositeAlphaFrame, CandidateUniverse, ConstraintSet

def build_topk_equal_weight(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    k: int,
) -> pd.Series:
    """Basic fallback: equal weight for Top K stocks."""
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    topk = scores.nlargest(k).index
    weight = pd.Series(1.0 / k, index=topk)
    return weight

def build_topk_score_weighted(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    k: int,
    softmax_temp: float = 1.0,
) -> pd.Series:
    """Fallback: softmax score weighted for Top K stocks."""
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    topk_scores = scores.nlargest(k)
    exp_s = np.exp(topk_scores / softmax_temp)
    weight = exp_s / exp_s.sum()
    return weight

def build_topk_with_buffer(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    prev_holdings: pd.Index,
    k: int,
    buffer_ratio: float = 0.2,
) -> pd.Series:
    """Fallback: Top K with buffer to reduce turnover."""
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    sorted_scores = scores.sort_values(ascending=False)

    hard_k = int(k * (1 - buffer_ratio))
    buffer_k = k - hard_k
    extra_pool_size = int(k * buffer_ratio * 2)

    hard_picks = sorted_scores.iloc[:hard_k].index
    buffer_zone = sorted_scores.iloc[hard_k: hard_k + extra_pool_size].index

    retained = prev_holdings.intersection(buffer_zone)
    new_picks = buffer_zone.difference(prev_holdings)

    buffer_picks = retained.append(new_picks)[:buffer_k]
    final_picks = hard_picks.append(buffer_picks)
    
    return pd.Series(1.0 / len(final_picks), index=final_picks)

def build_topk_with_constraints(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    constraint_set: ConstraintSet,
    industry_map: pd.Series,
    benchmark_industry_weights: pd.Series,
    k: int,
) -> pd.Series:
    """Greedy selection respecting industry constraints."""
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    sorted_scores = scores.sort_values(ascending=False)

    selected = []
    industry_current_weight: Dict[str, float] = defaultdict(float)
    per_stock_weight = 1.0 / k

    for stock in sorted_scores.index:
        if len(selected) >= k:
            break
        ind = industry_map.get(stock, "unknown")
        bm_ind_w = benchmark_industry_weights.get(ind, 0.0)
        new_ind_w = industry_current_weight[ind] + per_stock_weight
        if new_ind_w > bm_ind_w + constraint_set.industry_deviation_ub:
            continue
        selected.append(stock)
        industry_current_weight[ind] += per_stock_weight

    if not selected:
        raise RuntimeError("Constraints too strict, no stocks selected.")

    selected_scores = sorted_scores.loc[selected]
    exp_s = np.exp(selected_scores - selected_scores.max())
    weight = exp_s / exp_s.sum()
    weight = weight.clip(upper=constraint_set.max_single_stock_weight)
    weight = weight / weight.sum()
    return weight
