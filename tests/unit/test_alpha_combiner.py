import pytest
import pandas as pd
import numpy as np

from signal_fusion import WeightedAverageFusion
from data_models import AlphaFrame

def test_weighted_average_fusion():
    date = "20240701"
    stocks = ["000001.SZ", "000002.SZ"]
    
    f1 = AlphaFrame(
        date=date,
        domain="A",
        model_id="m1",
        horizon=5,
        scores=pd.Series([1.0, 2.0], index=stocks),
        score_version="v1"
    )
    
    f2 = AlphaFrame(
        date=date,
        domain="B",
        model_id="m2",
        horizon=5,
        scores=pd.Series([2.0, 1.0], index=stocks),
        score_version="v1"
    )
    
    combiner = WeightedAverageFusion({"m1": 0.5, "m2": 0.5})
    result = combiner.fuse([f1, f2])
    
    assert result.date == date
    assert "A" in result.source_domains
    assert "B" in result.source_domains
    assert not result.is_degraded
    
    # 1.5, 1.5 normalized to mean=0 -> 0, 0
    assert abs(result.composite_score.mean()) < 1e-6
