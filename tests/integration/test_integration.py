import pytest
import pandas as pd
import numpy as np

from tests.mock_data_generator import generate_mock_alpha_frames, generate_mock_market_data
from pipeline import PortfolioPipeline
from candidate_selection import CandidateSelector, CandidateSelectorConfig
from risk_model import RiskExposureBuilder
from risk_model.covariance.factor_cov_estimator import FactorCovEstimator
from constraints import ConstraintBuilder, ConstraintBuilderConfig
from optimizer import PortfolioOptimizer, OptimizerConfig
from postprocess import WeightPostProcessor, PostProcessConfig, PositionScaler, PositionScalerConfig
from reporting import RiskReporter, PortfolioExporter
from degradation import DegradationManager, DegradationConfig
from data_models import MarketStateSignal

def test_full_pipeline():
    date = "20240701"
    stocks = [f"{i:06d}.SZ" for i in range(1, 101)] # 100 stocks for test
    
    alpha_frames = generate_mock_alpha_frames(date, stocks)
    # Ensure some variation
    
    schema_path = r"D:\Trading\portfolio_layer_engineering\raw_data_stats.json"
    market_data = generate_mock_market_data(date, stocks, schema_path)
    
    pipeline = PortfolioPipeline(
        candidate_selector=CandidateSelector(CandidateSelectorConfig(min_turnover_rate=0.0)), # Disable to not filter out everything mock
        risk_exposure_builder=RiskExposureBuilder(),
        cov_estimator=FactorCovEstimator(window=20),
        constraint_builder=ConstraintBuilder(ConstraintBuilderConfig(max_single_stock_weight=0.1, min_stock_count=5, max_stock_count=50)),
        optimizer=PortfolioOptimizer(OptimizerConfig(fallback_topk=10)),
        post_processor=WeightPostProcessor(PostProcessConfig()),
        position_scaler=PositionScaler(PositionScalerConfig()),
        exporter=PortfolioExporter(),
        reporter=RiskReporter(),
        degradation_manager=DegradationManager(DegradationConfig())
    )
    
    market_signal = MarketStateSignal(
        date=date,
        gross_exposure_scale=0.9,
        cash_ratio_signal=0.1,
        risk_on_off_signal="neutral",
        signal_source="mock",
        available=True
    )
    
    portfolio, report = pipeline.run(
        date=date,
        alpha_frames=alpha_frames,
        market_data=market_data,
        market_signal=market_signal
    )
    
    assert portfolio.date == date
    assert abs(portfolio.gross_exposure - 0.93) < 1e-5
    assert abs(portfolio.target_weight.sum() - 0.93) < 0.05
    assert not portfolio.target_weight.empty
    assert len(portfolio.target_weight) <= 50
    assert report.stock_count > 0
    
if __name__ == "__main__":
    pytest.main(["-v", "test_integration.py"])
