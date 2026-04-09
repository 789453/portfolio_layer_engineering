import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta

# Ensure portfolio_layer is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_models import AlphaFrame, MarketDataBundle, MarketStateSignal
from pipeline import PortfolioPipeline
from candidate_selection import CandidateSelector, CandidateSelectorConfig
from risk_model import RiskExposureBuilder
from risk_model.covariance.factor_cov_estimator import FactorCovEstimator
from constraints import ConstraintBuilder, ConstraintBuilderConfig
from optimizer import PortfolioOptimizer, OptimizerConfig
from postprocess import WeightPostProcessor, PostProcessConfig, PositionScaler, PositionScalerConfig
from reporting import RiskReporter, PortfolioExporter
from degradation import DegradationManager, DegradationConfig
from data_loaders import ParquetDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_lookback_date(target_date: str, days: int) -> str:
    dt = datetime.strptime(target_date, "%Y%m%d")
    lookback = dt - timedelta(days=days*1.5) # Approximate trading days
    return lookback.strftime("%Y%m%d")

def load_real_data(target_date: str, raw_dir: str, feature_dir: str):
    logging.info(f"Loading data for {target_date} using efficient ParquetDataLoader...")
    loader = ParquetDataLoader(raw_dir, feature_dir)
    
    # 1. Load basic daily data
    lookback_date = get_lookback_date(target_date, 120) # Need up to 120 days for long_momentum
    daily_hist = loader.load_time_series('daily.parquet', lookback_date, target_date, is_feature=False)
    
    # 2. Load daily_basic for today
    daily_basic_today = loader.load_cross_section('daily_basic.parquet', target_date, is_feature=False)
    
    # 3. Load limit and suspend
    limit_today = loader.load_cross_section('stk_limit.parquet', target_date, is_feature=False)
    suspend_df = loader.load_full('suspend_d.parquet', is_feature=False) # Suspend is usually small, or we can cross-section
    
    # 4. Load index member
    index_member_df = loader.load_full('index_member_all.parquet', is_feature=False)
    
    # 5. Load features for Alpha and Risk
    fundamental_today = loader.load_cross_section('feature_D_fundamental.parquet', target_date, is_feature=True)
    chip_today = loader.load_cross_section('feature_C_chip.parquet', target_date, is_feature=True)
    
    # Define universe based on daily_basic_today
    if daily_basic_today.empty:
        raise ValueError(f"No daily_basic data found for {target_date}")
    stocks = daily_basic_today['ts_code'].unique().tolist()
    
    # Create MarketDataBundle
    daily_today = daily_hist[daily_hist['trade_date'] == target_date].copy() if not daily_hist.empty else pd.DataFrame()
    price_today = daily_today.set_index('ts_code')['close'] if not daily_today.empty else pd.Series(dtype=float)
    
    # Cap-weighted benchmark using daily_basic circ_mv
    if not daily_basic_today.empty and 'circ_mv' in daily_basic_today.columns:
        benchmark_weights = daily_basic_today.set_index('ts_code')['circ_mv'].reindex(stocks).fillna(0)
        benchmark_weights = benchmark_weights / benchmark_weights.sum()
    else:
        benchmark_weights = pd.Series(1.0 / len(stocks), index=stocks) if len(stocks) > 0 else pd.Series(dtype=float)
    
    # We set cov_matrix to None to let the pipeline estimate it using FactorCovEstimator
    cov_matrix = None
    
    bundle = MarketDataBundle(
        stk_limit=limit_today,
        suspend=suspend_df,
        daily_basic=daily_basic_today,
        index_member=index_member_df,
        price=daily_hist,
        fundamental=fundamental_today,
        benchmark_weights=benchmark_weights,
        cov_matrix=cov_matrix,
        total_asset=1e8,
        price_today=price_today
    )
    
    # Generate mock Alpha frames using real feature data
    alpha_frames = []
    
    if not fundamental_today.empty and 'val_rank' in fundamental_today.columns:
        scores = fundamental_today.set_index('ts_code')['val_rank'].reindex(stocks).fillna(0.5)
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        alpha_frames.append(AlphaFrame(
            date=target_date, domain="B", model_id="fundamental_v1", horizon=5,
            scores=scores, score_version="v1"
        ))
        
    # We create some additional dummy domains to test multi-domain fusion
    scores_d = pd.Series(np.random.randn(len(stocks)), index=stocks)
    alpha_frames.append(AlphaFrame(
        date=target_date, domain="D", model_id="dummy_v1", horizon=5,
        scores=scores_d, score_version="v1"
    ))
    
    scores_e = pd.Series(np.random.randn(len(stocks)), index=stocks)
    alpha_frames.append(AlphaFrame(
        date=target_date, domain="E", model_id="dummy_v2", horizon=5,
        scores=scores_e, score_version="v1"
    ))
        
    if not alpha_frames:
        # Fallback random alpha
        scores = pd.Series(np.random.randn(len(stocks)), index=stocks)
        alpha_frames.append(AlphaFrame(
            date=target_date, domain="A", model_id="random_v1", horizon=5,
            scores=scores, score_version="v1"
        ))
        
    return bundle, alpha_frames

def main():
    raw_dir = r"D:\Trading\data_ever_26_3_14\data\Raw_data"
    feature_dir = r"D:\Trading\data_ever_26_3_14\data\Feature_data\factor_ready"
    target_date = "20260313"
    
    bundle, alpha_frames = load_real_data(target_date, raw_dir, feature_dir)
    
    # To speed up optimization in test, select top 300 stocks by circ_mv
    if not bundle.daily_basic.empty:
        top_stocks = bundle.daily_basic.nlargest(300, 'circ_mv')['ts_code'].tolist()
        
        # Filter all data to these top 300 stocks to avoid slow cvxpy
        for frame in alpha_frames:
            frame.scores = frame.scores.reindex(top_stocks).dropna()
        
        bundle.benchmark_weights = bundle.benchmark_weights.reindex(top_stocks).fillna(0)
        bundle.benchmark_weights /= bundle.benchmark_weights.sum()
        if bundle.cov_matrix is not None:
            bundle.cov_matrix = bundle.cov_matrix.loc[top_stocks, top_stocks]
    
    pipeline = PortfolioPipeline(
        candidate_selector=CandidateSelector(CandidateSelectorConfig(
            min_turnover_rate=0.001,
            exclude_limit_up_for_buy=True,
            exclude_limit_down_for_sell=True,
            market_cap_filter_pct=0.1
        )),
        risk_exposure_builder=RiskExposureBuilder(),
        cov_estimator=FactorCovEstimator(window=60, shrinkage=0.1),
        constraint_builder=ConstraintBuilder(ConstraintBuilderConfig(
            max_single_stock_weight=0.05,
            min_stock_count=20,
            max_stock_count=100
        )),
        optimizer=PortfolioOptimizer(OptimizerConfig(
            fallback_topk=50,
            risk_aversion=0.5,
            industry_soft_penalty=10.0,
            style_soft_penalty=5.0,
            turnover_soft_penalty=10.0
        )),
        post_processor=WeightPostProcessor(PostProcessConfig(min_weight_threshold=0.005)),
        position_scaler=PositionScaler(PositionScalerConfig()),
        exporter=PortfolioExporter(),
        reporter=RiskReporter(),
        degradation_manager=DegradationManager(DegradationConfig())
    )
    
    market_signal = MarketStateSignal(
        date=target_date,
        gross_exposure_scale=0.95,
        cash_ratio_signal=0.05,
        risk_on_off_signal="neutral",
        signal_source="index_model"
    )
    
    portfolio, report = pipeline.run(
        date=target_date,
        alpha_frames=alpha_frames,
        market_data=bundle,
        market_signal=market_signal
    )
    
    print("\n" + "="*50)
    print(f"Portfolio Optimization Result for {target_date}")
    print("="*50)
    print(f"Status: {portfolio.optimizer_status}")
    print(f"Is Degraded: {portfolio.is_degraded}")
    print(f"Total Gross Exposure: {portfolio.gross_exposure:.2%}")
    print(f"Stock Count: {report.stock_count}")
    print(f"Top 10 Weight: {report.top10_weight:.2%}")
    print(f"Herfindahl Index: {report.herfindahl_index:.4f}")
    
    print("\nTop 10 Holdings:")
    print(portfolio.target_weight.nlargest(10))

if __name__ == "__main__":
    main()
