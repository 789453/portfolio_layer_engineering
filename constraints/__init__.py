from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import numpy as np

from data_models import CandidateUniverse, RiskExposureFrame, ConstraintSet

@dataclass
class ConstraintBuilderConfig:
    total_weight_lb: float = 0.95
    total_weight_ub: float = 1.05
    industry_deviation_ub: float = 0.05
    industry_abs_ub: float = 0.30
    style_deviation_ub: Dict[str, float] = field(default_factory=dict)
    turnover_ub: float = 0.30
    liquidity_adv_fraction_ub: float = 0.10
    max_single_stock_weight: float = 0.05
    min_stock_count: int = 30
    max_stock_count: int = 150
    tracking_error_ub: Optional[float] = None
    small_cap_threshold: float = 200000.0 # 20e4 万元 = 20 亿

class ConstraintBuilder:
    def __init__(self, config: ConstraintBuilderConfig):
        self.config = config

    def build(
        self,
        date: str,
        candidate_universe: CandidateUniverse,
        risk_exposure: RiskExposureFrame,
        daily_basic_df: pd.DataFrame,
        prev_weights: pd.Series,
    ) -> ConstraintSet:
        stocks = candidate_universe.primary
        if stocks.empty:
            return ConstraintSet(weight_lb=pd.Series(), weight_ub=pd.Series())

        weight_ub = self._build_weight_ub(date, stocks, daily_basic_df)
        weight_lb = pd.Series(0.0, index=stocks)

        # Asymmetric bounds for limit up/down stocks
        for stock in candidate_universe.limit_up_stocks:
            if stock in weight_ub.index:
                # Cannot buy limit up, max weight is what we currently hold
                weight_ub[stock] = prev_weights.get(stock, 0.0)

        for stock in candidate_universe.limit_down_stocks:
            if stock in weight_lb.index:
                # Cannot sell limit down, min weight is what we currently hold
                weight_lb[stock] = prev_weights.get(stock, 0.0)
                # Ensure ub is not less than lb
                weight_ub[stock] = max(weight_ub.get(stock, 0.0), weight_lb[stock])

        return ConstraintSet(
            weight_lb=weight_lb,
            weight_ub=weight_ub,
            total_weight_lb=self.config.total_weight_lb,
            total_weight_ub=self.config.total_weight_ub,
            industry_deviation_ub=self.config.industry_deviation_ub,
            industry_abs_ub=self.config.industry_abs_ub,
            style_deviation_ub=self.config.style_deviation_ub,
            turnover_ub=self.config.turnover_ub,
            liquidity_adv_fraction_ub=self.config.liquidity_adv_fraction_ub,
            max_single_stock_weight=self.config.max_single_stock_weight,
            min_stock_count=self.config.min_stock_count,
            max_stock_count=self.config.max_stock_count,
            tracking_error_ub=self.config.tracking_error_ub,
        )

    def _build_weight_ub(self, date: str, stocks: pd.Index, daily_basic_df: pd.DataFrame) -> pd.Series:
        base_ub = self.config.max_single_stock_weight
        ub = pd.Series(base_ub, index=stocks)
        
        if not daily_basic_df.empty:
            df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
            circ_mv = df["circ_mv"].reindex(stocks, fill_value=np.nan)
            small_cap_mask = circ_mv < self.config.small_cap_threshold
            ub[small_cap_mask] = base_ub * 0.5
            
        return ub
