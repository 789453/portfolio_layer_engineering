import pytest
import pandas as pd

from postprocess import WeightPostProcessor, PostProcessConfig, PositionScaler, PositionScalerConfig
from data_models import MarketStateSignal

def test_weight_post_processor():
    processor = WeightPostProcessor(PostProcessConfig(min_weight_threshold=0.01))
    
    raw = pd.Series({"A": 0.5, "B": 0.495, "C": 0.005})
    prev = pd.Series({"A": 0.4, "B": 0.6})
    
    final, target_pos, reb = processor.process(raw, prev)
    
    assert "C" not in final.index
    assert abs(final["A"] - (0.5 / 0.995)) < 1e-5
    
    # rebalance list test
    assert len(reb) > 0
    assert "BUY" in reb["direction"].values
    assert "SELL" in reb["direction"].values

def test_position_scaler():
    scaler = PositionScaler(PositionScalerConfig(scale_smoothing=0.5))
    target = pd.Series({"A": 0.5, "B": 0.5})
    
    signal = MarketStateSignal("2024", 0.8, 0.2, "neutral", "test", available=True)
    
    scaled, exp, cash = scaler.scale(target, signal, prev_gross_exposure=1.0)
    
    # 0.5 * 1.0 + 0.5 * 0.8 = 0.9
    assert abs(exp - 0.9) < 1e-5
    assert abs(scaled.sum() - 0.9) < 1e-5
    assert abs(cash - 0.1) < 1e-5
