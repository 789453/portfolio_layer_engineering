import pandas as pd
from typing import Optional, List
import logging

from data_models import TargetPortfolio, PortfolioRiskReport, RiskExposureFrame, AlphaFrame, CandidateUniverse
from signal_fusion import SingleModelFusion, MultiDomainFusion, HierarchicalFusion, AlphaCombiner

logger = logging.getLogger(__name__)

class RiskReporter:
    def report(
        self,
        portfolio: TargetPortfolio,
        risk_exposure: RiskExposureFrame,
        prev_portfolio: Optional[TargetPortfolio] = None,
    ) -> PortfolioRiskReport:
        w = portfolio.target_weight
        bm_w = risk_exposure.benchmark_weights

        if not risk_exposure.industry_exposure.empty:
            ind_exp = risk_exposure.industry_exposure
            portfolio_industry = ind_exp.T @ w.reindex(ind_exp.index, fill_value=0)
            bm_industry = ind_exp.T @ bm_w.reindex(ind_exp.index, fill_value=0)
            industry_active = portfolio_industry - bm_industry
        else:
            industry_active = pd.Series(dtype=float)

        if not risk_exposure.style_exposure.empty:
            sty_exp = risk_exposure.style_exposure
            portfolio_style = sty_exp.T @ w.reindex(sty_exp.index, fill_value=0)
            bm_style = sty_exp.T @ bm_w.reindex(sty_exp.index, fill_value=0)
            style_active = portfolio_style - bm_style
        else:
            style_active = pd.Series(dtype=float)

        turnover = 0.0
        if prev_portfolio is not None and not prev_portfolio.target_weight.empty:
            prev_w = prev_portfolio.target_weight
            all_stocks = w.index.union(prev_w.index)
            delta = w.reindex(all_stocks, fill_value=0) - prev_w.reindex(all_stocks, fill_value=0)
            turnover = delta.abs().sum() / 2 # Single-sided turnover

        top10_weight = w.nlargest(10).sum() if not w.empty else 0.0
        hhi = (w ** 2).sum() if not w.empty else 0.0
        
        # Estimate Active Risk (Tracking Error) if cov_matrix is available (optional improvement for later)
        # We leave it as 0.0 here, but ideally we'd pass cov_matrix to reporter

        return PortfolioRiskReport(
            date=portfolio.date,
            industry_exposure_active=industry_active,
            style_exposure_active=style_active,
            top10_weight=top10_weight,
            stock_count=(w > 0).sum(),
            estimated_tracking_error=0.0,
            estimated_active_risk=0.0,
            turnover_rate=turnover,
            constraint_violations=[],
            herfindahl_index=hhi,
        )

class PortfolioExporter:
    def export(self, portfolio: TargetPortfolio, output_path: str, format: str = "parquet") -> None:
        data = {
            "date": portfolio.date,
            "ts_code": portfolio.target_weight.index.tolist(),
            "target_weight": portfolio.target_weight.values.tolist(),
            "is_degraded": portfolio.is_degraded,
            "optimizer_status": portfolio.optimizer_status,
            "fusion_method": portfolio.fusion_method,
        }
        df = pd.DataFrame(data)
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
