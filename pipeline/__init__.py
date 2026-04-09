from typing import Tuple, List, Optional
import logging
import pandas as pd

from data_models import AlphaFrame, MarketDataBundle, TargetPortfolio, PortfolioRiskReport, MarketStateSignal, RiskExposureFrame
from signal_fusion import AlphaCombiner
from candidate_selection import CandidateSelector
from risk_model import RiskExposureBuilder
from risk_model.covariance.factor_cov_estimator import FactorCovEstimator
from constraints import ConstraintBuilder
from optimizer import PortfolioOptimizer
from postprocess import WeightPostProcessor, PositionScaler
from reporting import RiskReporter, PortfolioExporter
from degradation import DegradationManager

logger = logging.getLogger(__name__)

class PortfolioPipeline:
    def __init__(
        self,
        candidate_selector: CandidateSelector,
        risk_exposure_builder: RiskExposureBuilder,
        cov_estimator: FactorCovEstimator,
        constraint_builder: ConstraintBuilder,
        optimizer: PortfolioOptimizer,
        post_processor: WeightPostProcessor,
        position_scaler: PositionScaler,
        exporter: PortfolioExporter,
        reporter: RiskReporter,
        degradation_manager: DegradationManager,
    ):
        self.candidate_selector = candidate_selector
        self.risk_builder = risk_exposure_builder
        self.cov_estimator = cov_estimator
        self.constraint_builder = constraint_builder
        self.optimizer = optimizer
        self.post_processor = post_processor
        self.position_scaler = position_scaler
        self.exporter = exporter
        self.reporter = reporter
        self.degradation = degradation_manager

    def run(
        self,
        date: str,
        alpha_frames: List[AlphaFrame],
        market_data: MarketDataBundle,
        prev_portfolio: Optional[TargetPortfolio] = None,
        market_signal: Optional[MarketStateSignal] = None,
    ) -> Tuple[TargetPortfolio, PortfolioRiskReport]:
        
        logger.info(f"[{date}] 开始组合层流水线运行")

        # Step 1: Alpha 融合
        combiner = self.degradation.select_fusion_method(alpha_frames)
        composite_alpha = combiner.fuse(alpha_frames)
        logger.info(f"[{date}] Alpha 融合完成，使用方法: {composite_alpha.fusion_method}")

        # Step 2: 候选池构建
        candidate_universe = self.candidate_selector.build(
            date, composite_alpha,
            market_data.stk_limit, market_data.suspend,
            market_data.daily_basic, market_data.index_member,
        )
        
        fallback = self.degradation.handle_empty_candidate_pool(
            date, candidate_universe, prev_portfolio
        )
        if fallback is not None:
            return fallback, self.reporter.report(fallback, RiskExposureFrame.empty(date), prev_portfolio)

        # Step 3: 风险暴露构建
        risk_exposure = self.risk_builder.build(
            date, candidate_universe,
            market_data.index_member, market_data.daily_basic,
            market_data.price, market_data.fundamental,
            benchmark_weights=market_data.benchmark_weights,
        )

        # Step 3.5: 协方差矩阵估计
        if market_data.cov_matrix is None:
            cov_matrix = self.cov_estimator.estimate(
                date,
                candidate_universe.primary,
                risk_exposure.industry_exposure,
                risk_exposure.style_exposure,
                market_data.price
            )
        else:
            cov_matrix = market_data.cov_matrix

        # Step 4: 约束构建
        prev_weights = prev_portfolio.target_weight if prev_portfolio else pd.Series(dtype=float)
        constraint_set = self.constraint_builder.build(
            date, candidate_universe, risk_exposure,
            market_data.daily_basic, prev_weights,
        )

        # Step 5: 组合优化
        raw_weight, optimizer_status = self.optimizer.optimize(
            composite_alpha, risk_exposure, constraint_set,
            prev_weights, cov_matrix=cov_matrix,
        )

        # Step 6: 权重后处理
        final_weight, target_position, rebalance_list = self.post_processor.process(
            raw_weight, prev_weights,
            total_asset=market_data.total_asset,
            price_series=market_data.price_today,
        )

        # Step 7: 仓位缩放
        prev_gross = prev_portfolio.gross_exposure if prev_portfolio else 1.0
        scaled_weight, gross_exposure, cash_ratio = self.position_scaler.scale(
            final_weight, market_signal, prev_gross
        )

        # Step 8: 构建 TargetPortfolio
        portfolio = TargetPortfolio(
            date=date,
            target_weight=scaled_weight,
            target_position=target_position,
            rebalance_list=rebalance_list,
            gross_exposure=gross_exposure,
            cash_ratio=cash_ratio,
            optimizer_status=optimizer_status,
            fusion_method=composite_alpha.fusion_method,
            is_degraded=composite_alpha.is_degraded or optimizer_status == "degraded",
        )

        # Step 9: 风险报告
        risk_report = self.reporter.report(portfolio, risk_exposure, prev_portfolio)

        # Step 10: 导出
        self.exporter.export(portfolio, output_path=f"outputs/{date}/portfolio.parquet")

        logger.info(f"[{date}] 组合层流水线完成，持仓数: {risk_report.stock_count}，"
                    f"换手率: {risk_report.turnover_rate:.2%}，状态: {optimizer_status}")
        return portfolio, risk_report
