from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from data_models import MarketStateSignal

@dataclass
class PostProcessConfig:
    min_weight_threshold: float = 0.001

class WeightPostProcessor:
    def __init__(self, config: PostProcessConfig):
        self.config = config

    def process(
        self,
        raw_weight: pd.Series,
        prev_weight: pd.Series,
        total_asset: Optional[float] = None,
        price_series: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        
        if raw_weight.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()
            
        cleaned = raw_weight.copy()
        cleaned[cleaned < self.config.min_weight_threshold] = 0.0
        
        total_cleaned = cleaned.sum()
        if total_cleaned > 0:
            cleaned = cleaned / total_cleaned
            
        cleaned = cleaned[cleaned > 0.0]

        if total_asset is not None and price_series is not None and not price_series.empty:
            target_position = self._discretize(cleaned, total_asset, price_series)
        else:
            target_position = pd.Series(dtype=float)

        rebalance_list = self._build_rebalance_list(cleaned, prev_weight)

        return cleaned, target_position, rebalance_list

    def _discretize(self, weight: pd.Series, total_asset: float, price: pd.Series) -> pd.Series:
        stocks = weight.index
        valid_prices = price.reindex(stocks).replace(0, np.nan)
        shares_float = weight * total_asset / valid_prices
        shares_rounded = (shares_float / 100).round() * 100
        return shares_rounded.clip(lower=0).fillna(0)

    def _build_rebalance_list(self, target: pd.Series, prev: pd.Series) -> pd.DataFrame:
        all_stocks = target.index.union(prev.index)
        target_full = target.reindex(all_stocks, fill_value=0.0)
        prev_full = prev.reindex(all_stocks, fill_value=0.0)
        delta = target_full - prev_full
        
        def get_direction(x):
            if x > 1e-5: return "BUY"
            if x < -1e-5: return "SELL"
            return "HOLD"
            
        direction = delta.apply(get_direction)
        df = pd.DataFrame({
            "ts_code": all_stocks,
            "current_weight": prev_full,
            "target_weight": target_full,
            "delta_weight": delta,
            "direction": direction,
        }).set_index("ts_code")
        
        return df[df["direction"] != "HOLD"]


@dataclass
class PositionScalerConfig:
    min_gross_exposure: float = 0.60
    max_gross_exposure: float = 1.00
    scale_smoothing: float = 0.30
    default_scale: float = 1.0

class PositionScaler:
    def __init__(self, config: PositionScalerConfig):
        self.config = config

    def scale(
        self,
        target_weight: pd.Series,
        market_signal: Optional[MarketStateSignal],
        prev_gross_exposure: float = 1.0,
    ) -> Tuple[pd.Series, float, float]:
        
        if market_signal is None or not market_signal.available:
            return target_weight, 1.0, 0.0

        scale = market_signal.gross_exposure_scale
        smoothed_scale = (
            self.config.scale_smoothing * prev_gross_exposure
            + (1 - self.config.scale_smoothing) * scale
        )
        
        smoothed_scale = float(np.clip(
            smoothed_scale,
            self.config.min_gross_exposure,
            self.config.max_gross_exposure
        ))

        scaled_weight = target_weight * smoothed_scale
        cash_ratio = max(0.0, 1.0 - scaled_weight.sum())

        return scaled_weight, smoothed_scale, cash_ratio
