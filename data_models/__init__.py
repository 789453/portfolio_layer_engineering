from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class AlphaFrame:
    date: str                          # YYYYMMDD
    domain: str                        # "A"/"B"/"C"/"D"/"E"
    model_id: str                      # "lgb_v3_5d"
    horizon: int                       # 1/5/10/20
    scores: pd.Series                  # index=ts_code, value=float alpha_score
    score_version: str                 # "20250701"
    available: bool = True
    meta: dict = field(default_factory=dict)

@dataclass
class CompositeAlphaFrame:
    date: str
    composite_score: pd.Series
    source_domains: List[str]
    fusion_method: str
    domain_weights: Dict[str, float]
    is_degraded: bool = False
    degraded_domains: List[str] = field(default_factory=list)

@dataclass
class CandidateUniverse:
    date: str
    primary: pd.Index
    reserve: pd.Index
    excluded: pd.Index
    exclusion_reason: Dict[str, str]
    limit_up_stocks: pd.Index = field(default_factory=lambda: pd.Index([]))
    limit_down_stocks: pd.Index = field(default_factory=lambda: pd.Index([]))

@dataclass
class RiskExposureFrame:
    date: str
    industry_exposure: pd.DataFrame
    style_exposure: pd.DataFrame
    benchmark_weights: pd.Series

    @classmethod
    def empty(cls, date: str) -> "RiskExposureFrame":
        return cls(
            date=date,
            industry_exposure=pd.DataFrame(),
            style_exposure=pd.DataFrame(),
            benchmark_weights=pd.Series(dtype=float)
        )

@dataclass
class ConstraintSet:
    weight_lb: pd.Series
    weight_ub: pd.Series
    total_weight_lb: float = 0.95
    total_weight_ub: float = 1.05
    industry_deviation_ub: float = 0.05
    industry_abs_ub: float = 0.30
    style_deviation_ub: Dict[str, float] = field(default_factory=dict)
    turnover_ub: float = 0.30
    liquidity_adv_fraction_ub: float = 0.10
    min_liquidity_score: float = 0.0
    max_single_stock_weight: float = 0.05
    min_stock_count: int = 30
    max_stock_count: int = 150
    tracking_error_ub: Optional[float] = None
    active_risk_budget: Optional[float] = None

@dataclass
class TargetPortfolio:
    date: str
    target_weight: pd.Series
    target_position: pd.Series
    rebalance_list: pd.DataFrame
    gross_exposure: float
    cash_ratio: float
    optimizer_status: str
    fusion_method: str
    is_degraded: bool = False
    meta: dict = field(default_factory=dict)

    @classmethod
    def hold_previous(cls, date: str, prev_portfolio: "TargetPortfolio" = None) -> "TargetPortfolio":
        if prev_portfolio is None:
            return cls(
                date=date,
                target_weight=pd.Series(dtype=float),
                target_position=pd.Series(dtype=float),
                rebalance_list=pd.DataFrame(),
                gross_exposure=1.0,
                cash_ratio=0.0,
                optimizer_status="hold_previous_empty",
                fusion_method="none",
                is_degraded=True
            )
        return cls(
            date=date,
            target_weight=prev_portfolio.target_weight,
            target_position=prev_portfolio.target_position,
            rebalance_list=pd.DataFrame(),
            gross_exposure=prev_portfolio.gross_exposure,
            cash_ratio=prev_portfolio.cash_ratio,
            optimizer_status="hold_previous",
            fusion_method="none",
            is_degraded=True
        )

@dataclass
class PortfolioRiskReport:
    date: str
    industry_exposure_active: pd.Series
    style_exposure_active: pd.Series
    top10_weight: float
    stock_count: int
    estimated_tracking_error: float
    estimated_active_risk: float
    turnover_rate: float
    constraint_violations: List[str]
    herfindahl_index: float

@dataclass
class MarketStateSignal:
    date: str
    gross_exposure_scale: float
    cash_ratio_signal: float
    risk_on_off_signal: str
    signal_source: str
    confidence: float = 1.0
    available: bool = True

@dataclass
class MarketDataBundle:
    stk_limit: pd.DataFrame
    suspend: pd.DataFrame
    daily_basic: pd.DataFrame
    index_member: pd.DataFrame
    price: pd.DataFrame
    fundamental: pd.DataFrame
    benchmark_weights: pd.Series
    cov_matrix: Optional[pd.DataFrame] = None
    total_asset: float = 1e8
    price_today: Optional[pd.Series] = None
