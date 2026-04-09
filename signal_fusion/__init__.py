from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from data_models import AlphaFrame, CompositeAlphaFrame

def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if pd.isna(std) or std == 0:
        return s - s.mean()
    return (s - s.mean()) / std

def winsorize(s: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
    return s.clip(lower=s.quantile(limits[0]), upper=s.quantile(1 - limits[1]))

class AlphaCombiner(ABC):
    @abstractmethod
    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        pass

    def validate_frames(self, frames: List[AlphaFrame]) -> None:
        dates = [f.date for f in frames if f.available]
        if dates:
            assert len(set(dates)) <= 1, f"AlphaFrame 日期不一致: {set(dates)}"

class SingleModelFusion(AlphaCombiner):
    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        self.validate_frames(frames)
        available_frames = [f for f in frames if f.available]
        if not available_frames:
            raise RuntimeError("无可用 AlphaFrame")
        
        frame = available_frames[0]
        score_norm = zscore(frame.scores)
        
        return CompositeAlphaFrame(
            date=frame.date,
            composite_score=score_norm,
            source_domains=[frame.domain],
            fusion_method="single_model",
            domain_weights={frame.domain: 1.0},
            is_degraded=len(available_frames) < len(frames),
            degraded_domains=[f.domain for f in frames if not f.available]
        )

class WeightedAverageFusion(AlphaCombiner):
    def __init__(self, weights: Dict[str, float], normalize: bool = True):
        self.weights = weights
        self.normalize = normalize

    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        self.validate_frames(frames)
        available_frames = [f for f in frames if f.available]
        if not available_frames:
            raise RuntimeError("所有 AlphaFrame 均不可用，无法融合")

        scores_df = pd.DataFrame({f.model_id: f.scores for f in available_frames})
        w = {mid: self.weights.get(mid, 0.0) for mid in scores_df.columns}

        if self.normalize:
            total = sum(w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}

        composite = sum(scores_df[mid] * wt for mid, wt in w.items())
        composite = zscore(composite)

        degraded_domains = list({f.domain for f in frames if not f.available})
        return CompositeAlphaFrame(
            date=available_frames[0].date,
            composite_score=composite,
            source_domains=list({f.domain for f in available_frames}),
            fusion_method="weighted_average",
            domain_weights=w,
            is_degraded=len(available_frames) < len(frames),
            degraded_domains=degraded_domains,
        )

class MultiDomainFusion(AlphaCombiner):
    def __init__(self, domain_weights: Dict[str, float], min_available_domains: int = 1):
        self.domain_weights = domain_weights
        self.min_available_domains = min_available_domains

    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        self.validate_frames(frames)
        domain_scores: Dict[str, pd.Series] = {}
        
        for f in frames:
            if f.available:
                if f.domain in domain_scores:
                    domain_scores[f.domain] = (domain_scores[f.domain] + f.scores) / 2
                else:
                    domain_scores[f.domain] = f.scores

        available_domains = list(domain_scores.keys())
        if len(available_domains) < self.min_available_domains:
            raise RuntimeError(f"可用域数 {len(available_domains)} 低于最低要求 {self.min_available_domains}")

        total_w = sum(self.domain_weights.get(d, 0.0) for d in available_domains)
        if total_w <= 0:
            raise ValueError("所有可用域的权重总和为 0")

        # Get union of all indices
        all_indices = pd.Index([])
        for s in domain_scores.values():
            all_indices = all_indices.union(s.index)

        composite = pd.Series(0.0, index=all_indices)
        used_weights = {}
        for d in available_domains:
            w = self.domain_weights.get(d, 0.0) / total_w
            score_norm = zscore(domain_scores[d])
            score_norm = score_norm.reindex(all_indices).fillna(0) # handle missing
            composite += w * score_norm
            used_weights[d] = w

        # Need to mask out stocks that have no scores in any domain
        valid_mask = pd.concat([domain_scores[d].notna() for d in available_domains], axis=1).any(axis=1)
        composite = composite[valid_mask]
        composite = zscore(composite)

        degraded_domains = list({f.domain for f in frames if not f.available})
        return CompositeAlphaFrame(
            date=frames[0].date,
            composite_score=composite,
            source_domains=available_domains,
            fusion_method="multi_domain",
            domain_weights=used_weights,
            is_degraded=len(degraded_domains) > 0,
            degraded_domains=degraded_domains,
        )

class HierarchicalFusion(AlphaCombiner):
    def __init__(
        self,
        intra_domain_weights: Dict[str, Dict[str, float]],
        inter_domain_weights: Dict[str, float],
    ):
        self.intra = intra_domain_weights
        self.inter = inter_domain_weights

    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        self.validate_frames(frames)
        domain_scores: Dict[str, pd.Series] = {}
        
        for domain in set(f.domain for f in frames):
            domain_frames = [f for f in frames if f.domain == domain and f.available]
            if not domain_frames:
                continue
            
            intra_w = self.intra.get(domain, {})
            total_intra = sum(intra_w.get(f.model_id, 1.0) for f in domain_frames)
            
            merged = None
            for f in domain_frames:
                w = intra_w.get(f.model_id, 1.0) / total_intra
                if merged is None:
                    merged = w * f.scores
                else:
                    # Align indices
                    merged = merged.add(w * f.scores, fill_value=0)
            
            domain_scores[domain] = zscore(merged)

        available_domains = list(domain_scores.keys())
        total_inter = sum(self.inter.get(d, 0.0) for d in available_domains)
        
        all_indices = pd.Index([])
        for s in domain_scores.values():
            all_indices = all_indices.union(s.index)
            
        composite = pd.Series(0.0, index=all_indices)
        for d in available_domains:
            w = self.inter.get(d, 0.0) / total_inter
            s = domain_scores[d].reindex(all_indices).fillna(0)
            composite += w * s
            
        valid_mask = pd.concat([domain_scores[d].notna() for d in available_domains], axis=1).any(axis=1)
        composite = composite[valid_mask]
        composite = zscore(composite)

        degraded_domains = [d for d in self.inter if d not in available_domains]
        return CompositeAlphaFrame(
            date=frames[0].date,
            composite_score=composite,
            source_domains=available_domains,
            fusion_method="hierarchical",
            domain_weights={d: self.inter.get(d, 0.0) / total_inter for d in available_domains},
            is_degraded=len(degraded_domains) > 0,
            degraded_domains=degraded_domains,
        )
