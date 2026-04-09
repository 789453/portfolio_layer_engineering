from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import cvxpy as cp
import logging

from data_models import CompositeAlphaFrame, RiskExposureFrame, ConstraintSet

logger = logging.getLogger(__name__)

from .penalty_calibrator import PenaltyCalibrator

@dataclass
class OptimizerConfig:
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.5
    use_dynamic_penalty: bool = True
    eta_industry: float = 5.0
    eta_style: float = 3.0
    eta_turnover: float = 4.0
    industry_soft_penalty: float = 10.0 # Used if dynamic is False
    style_soft_penalty: float = 5.0     # Used if dynamic is False
    turnover_soft_penalty: float = 10.0 # Used if dynamic is False
    solver: str = "SCS" # Using SCS or OSQP since CLARABEL might not be installed or as robust in all envs without specific versions
    fallback_solver: str = "OSQP"
    fallback_topk: int = 50
    max_solve_time: float = 60.0
    warm_start: bool = True

class PortfolioOptimizer:
    def __init__(self, config: OptimizerConfig):
        self.config = config

    def optimize(
        self,
        composite_alpha: CompositeAlphaFrame,
        risk_exposure: RiskExposureFrame,
        constraint_set: ConstraintSet,
        prev_weights: pd.Series,
        cov_matrix: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, str]:
        
        stocks = composite_alpha.composite_score.dropna().index
        if stocks.empty:
            return pd.Series(dtype=float), "empty_universe"
            
        n = len(stocks)
        mu = composite_alpha.composite_score.loc[stocks].values

        if self.config.use_dynamic_penalty:
            calibrator = PenaltyCalibrator(
                eta_industry=self.config.eta_industry,
                eta_style=self.config.eta_style,
                eta_turnover=self.config.eta_turnover
            )
            penalties = calibrator.calibrate(mu, constraint_set)
        else:
            from .penalty_calibrator import PenaltyMultipliers
            penalties = PenaltyMultipliers(
                industry_deviation=self.config.industry_soft_penalty,
                style_deviation=self.config.style_soft_penalty,
                turnover_excess=self.config.turnover_soft_penalty
            )

        w = cp.Variable(n, nonneg=True)
        objective_terms = [mu @ w]

        if cov_matrix is not None:
            # align
            Sigma = cov_matrix.reindex(index=stocks, columns=stocks).fillna(0).values
            objective_terms.append(-self.config.risk_aversion * cp.quad_form(w, Sigma))

        w_prev = prev_weights.reindex(stocks, fill_value=0.0).values
        objective_terms.append(-self.config.turnover_penalty * cp.norm1(w - w_prev))

        objective = cp.Maximize(sum(objective_terms))
        constraints = []

        constraints.append(cp.sum(w) >= constraint_set.total_weight_lb)
        constraints.append(cp.sum(w) <= constraint_set.total_weight_ub)

        w_ub = constraint_set.weight_ub.reindex(stocks, fill_value=constraint_set.max_single_stock_weight).values
        w_lb = constraint_set.weight_lb.reindex(stocks, fill_value=0.0).values
        constraints.append(w <= w_ub)
        constraints.append(w >= w_lb)

        if not risk_exposure.industry_exposure.empty:
            industry_dummies = risk_exposure.industry_exposure.reindex(stocks).fillna(0)
            bm_industry_w = (risk_exposure.benchmark_weights.reindex(stocks, fill_value=0.0).values
                             @ industry_dummies.values)
            for k, ind in enumerate(industry_dummies.columns):
                ind_vec = industry_dummies[ind].values
                bm_w = bm_industry_w[k] if hasattr(bm_industry_w, '__len__') else 0.0
                # Use soft constraints for industry
                deviation = cp.Variable(nonneg=True)
                constraints.append(ind_vec @ w - bm_w <= constraint_set.industry_deviation_ub + deviation)
                constraints.append(bm_w - ind_vec @ w <= constraint_set.industry_deviation_ub + deviation)
                objective_terms.append(-penalties.industry_deviation * deviation) # Penalty for violating industry constraint

        # Style constraints with soft penalty
        if constraint_set.style_deviation_ub and not risk_exposure.style_exposure.empty:
            style_mat = risk_exposure.style_exposure.reindex(stocks).fillna(0)
            bm_style = (risk_exposure.benchmark_weights.reindex(stocks, fill_value=0.0).values
                        @ style_mat.values)
            for j, style in enumerate(style_mat.columns):
                if style in constraint_set.style_deviation_ub:
                    eps = constraint_set.style_deviation_ub[style]
                    f_vec = style_mat[style].values
                    bm_f = bm_style[j] if hasattr(bm_style, '__len__') else bm_style
                    
                    style_dev = cp.Variable(nonneg=True)
                    constraints.append(f_vec @ w - bm_f <= eps + style_dev)
                    constraints.append(bm_f - f_vec @ w <= eps + style_dev)
                    objective_terms.append(-penalties.style_deviation * style_dev) # Penalty for style deviation

        # Turnover soft constraint
        # cp.norm1(w - w_prev) computes double-sided turnover. We divide by 2 to compare with single-sided constraint_set.turnover_ub
        turnover_dev = cp.Variable(nonneg=True)
        constraints.append(cp.norm1(w - w_prev) / 2.0 <= constraint_set.turnover_ub + turnover_dev)
        objective_terms.append(-penalties.turnover_excess * turnover_dev) # Penalty for violating turnover constraint

        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=self.config.solver, warm_start=self.config.warm_start)
        except Exception as e:
            logger.error(f"优化器求解失败：{e}，尝试降级求解")
            try:
                prob.solve(solver=self.config.fallback_solver)
            except Exception as e2:
                logger.error(f"降级求解也失败：{e2}")

        if prob.status in ["optimal", "optimal_inaccurate"]:
            weight_series = pd.Series(w.value, index=stocks)
            weight_series = weight_series.clip(lower=0)
            if weight_series.sum() > 0:
                weight_series = weight_series / weight_series.sum()
            return weight_series, prob.status
        else:
            logger.warning(f"优化器状态 {prob.status}，降级为等权 Top-K")
            k = min(self.config.fallback_topk, len(stocks))
            topk = composite_alpha.composite_score.loc[stocks].nlargest(k).index
            weight_series = pd.Series(1.0 / k, index=topk)
            return weight_series, "degraded"
