from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import logging

from data_models import AlphaFrame, TargetPortfolio, CandidateUniverse
from signal_fusion import AlphaCombiner, SingleModelFusion, MultiDomainFusion, HierarchicalFusion

logger = logging.getLogger(__name__)

class FatalDegradationError(Exception):
    pass

@dataclass
class DegradationConfig:
    min_domains_for_hierarchical: int = 3
    fallback_domain_weights: Dict[str, float] = field(default_factory=lambda: {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})
    intra_domain_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    inter_domain_weights: Dict[str, float] = field(default_factory=lambda: {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})

class DegradationManager:
    def __init__(self, config: DegradationConfig):
        self.config = config

    def select_fusion_method(self, frames: List[AlphaFrame]) -> AlphaCombiner:
        available_domains = list(set(f.domain for f in frames if f.available))
        n = len(available_domains)

        if n == 0:
            raise FatalDegradationError("无可用 Alpha 域，系统无法运行")
        elif n == 1:
            logger.critical(f"仅有 1 个可用域: {available_domains}，切换为 SingleModelFusion")
            return SingleModelFusion()
        elif n < self.config.min_domains_for_hierarchical:
            logger.error(f"可用域数 {n} 不足，切换为 MultiDomainFusion")
            return MultiDomainFusion(
                domain_weights=self.config.fallback_domain_weights,
                min_available_domains=1,
            )
        else:
            return HierarchicalFusion(
                intra_domain_weights=self.config.intra_domain_weights,
                inter_domain_weights=self.config.inter_domain_weights,
            )

    def handle_empty_candidate_pool(
        self,
        date: str,
        candidate_universe: CandidateUniverse,
        prev_portfolio: Optional[TargetPortfolio],
    ) -> Optional[TargetPortfolio]:
        if len(candidate_universe.primary) > 0:
            return None

        if len(candidate_universe.reserve) > 0:
            logger.warning(f"[{date}] 主候选池为空，扩展至替补池")
            candidate_universe.primary = candidate_universe.reserve
            return None

        if prev_portfolio is not None:
            logger.error(f"[{date}] 候选池完全为空，维持上期持仓")
            return TargetPortfolio.hold_previous(date, prev_portfolio)

        raise FatalDegradationError(f"[{date}] 候选池为空且无历史持仓，无法生成目标组合")
