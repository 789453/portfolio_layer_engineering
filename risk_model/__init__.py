from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np

from data_models import CandidateUniverse, RiskExposureFrame, ConstraintSet

class RiskExposureBuilder:
    STYLE_FACTORS = [
        "size",          # log(circ_mv)
        "value",         # val_rank or 1/pb
        "momentum",      # 20d return
        "long_momentum", # 120d return (excluding recent 20d)
        "volatility",    # 20d std of returns
        "liquidity",     # turnover_rate
        "beta",          # 60d OLS beta
        "growth",        # from fundamental
    ]

    def build(
        self,
        date: str,
        candidate_universe: CandidateUniverse,
        index_member_df: pd.DataFrame,
        daily_basic_df: pd.DataFrame,
        price_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        benchmark_weights: Optional[pd.Series] = None,
        index_price_df: Optional[pd.DataFrame] = None, # For beta calculation
    ) -> RiskExposureFrame:
        stocks = candidate_universe.primary
        if stocks.empty:
            return RiskExposureFrame.empty(date)

        industry_map = self._get_industry_map(index_member_df)
        industry_exposure = pd.get_dummies(
            industry_map.reindex(stocks, fill_value="unknown"),
            prefix="ind"
        ).astype(float)

        style_exposure = pd.DataFrame(index=stocks)
        style_exposure["size"] = self._calc_size(date, stocks, daily_basic_df)
        style_exposure["value"] = self._calc_value(date, stocks, daily_basic_df, fundamental_df)
        style_exposure["liquidity"] = self._calc_liquidity(date, stocks, daily_basic_df)
        style_exposure["momentum"] = self._calc_momentum(date, stocks, price_df, window=20)
        style_exposure["long_momentum"] = self._calc_long_momentum(date, stocks, price_df)
        style_exposure["volatility"] = self._calc_volatility(date, stocks, price_df, window=20)
        style_exposure["beta"] = self._calc_beta(date, stocks, price_df, index_price_df)
        style_exposure["growth"] = self._calc_growth(date, stocks, fundamental_df)
        
        # Fill missing styles with 0 for simplicity in this implementation
        for col in self.STYLE_FACTORS:
            if col not in style_exposure.columns:
                style_exposure[col] = 0.0

        style_exposure = style_exposure.fillna(style_exposure.mean()).fillna(0)
        style_exposure = style_exposure.apply(
            lambda col: (col - col.mean()) / (col.std() + 1e-8) if col.std() != 0 else col
        )

        bm_weights = benchmark_weights if benchmark_weights is not None else pd.Series(0.0, index=stocks)

        return RiskExposureFrame(
            date=date,
            industry_exposure=industry_exposure,
            style_exposure=style_exposure,
            benchmark_weights=bm_weights.reindex(stocks, fill_value=0.0),
        )

    def _get_industry_map(self, index_member_df: pd.DataFrame) -> pd.Series:
        if index_member_df is None or index_member_df.empty:
            return pd.Series(dtype=str)
        # Drop duplicates on ts_code keeping last
        df = index_member_df.drop_duplicates(subset=["ts_code"], keep="last")
        return df.set_index("ts_code")["l1_name"]

    def _calc_size(self, date: str, stocks: pd.Index, daily_basic_df: pd.DataFrame) -> pd.Series:
        if daily_basic_df is None or daily_basic_df.empty: return pd.Series(0.0, index=stocks)
        df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
        return np.log(df["circ_mv"].reindex(stocks) + 1)

    def _calc_value(self, date: str, stocks: pd.Index, daily_basic_df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.Series:
        if fundamental_df is not None and not fundamental_df.empty:
            df = fundamental_df[fundamental_df["trade_date"] == date].set_index("ts_code")
            if "val_rank" in df.columns:
                return df["val_rank"].reindex(stocks)
        if daily_basic_df is not None and not daily_basic_df.empty:
            df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
            pb = df["pb"].reindex(stocks)
            return 1.0 / pb.replace(0, np.nan)
        return pd.Series(0.0, index=stocks)

    def _calc_liquidity(self, date: str, stocks: pd.Index, daily_basic_df: pd.DataFrame) -> pd.Series:
        if daily_basic_df is None or daily_basic_df.empty: return pd.Series(0.0, index=stocks)
        df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
        return df["turnover_rate"].reindex(stocks)
        
    def _calc_momentum(self, date: str, stocks: pd.Index, price_df: pd.DataFrame, window: int = 20) -> pd.Series:
        if price_df is None or price_df.empty: return pd.Series(0.0, index=stocks)
        # Assuming price_df is sorted by date ascending
        p_df = price_df[price_df["trade_date"] <= date]
        if p_df.empty: return pd.Series(0.0, index=stocks)
        pivot_close = p_df.pivot(index="trade_date", columns="ts_code", values="close")
        if len(pivot_close) < window:
            window = len(pivot_close)
        if window <= 1:
            return pd.Series(0.0, index=stocks)
        
        # Return over window
        ret = pivot_close.iloc[-1] / pivot_close.iloc[-window] - 1
        return ret.reindex(stocks).fillna(0.0)
        
    def _calc_volatility(self, date: str, stocks: pd.Index, price_df: pd.DataFrame, window: int = 20) -> pd.Series:
        if price_df is None or price_df.empty: return pd.Series(0.0, index=stocks)
        p_df = price_df[price_df["trade_date"] <= date]
        if p_df.empty: return pd.Series(0.0, index=stocks)
        pivot_close = p_df.pivot(index="trade_date", columns="ts_code", values="close")
        if len(pivot_close) < 2:
            return pd.Series(0.0, index=stocks)
        
        ret = pivot_close.pct_change(fill_method=None).tail(window)
        vol = ret.std() * np.sqrt(252)
        return vol.reindex(stocks).fillna(0.0)

    def _calc_long_momentum(self, date: str, stocks: pd.Index, price_df: pd.DataFrame) -> pd.Series:
        if price_df is None or price_df.empty: return pd.Series(0.0, index=stocks)
        p_df = price_df[price_df["trade_date"] <= date]
        pivot_close = p_df.pivot(index="trade_date", columns="ts_code", values="close")
        if len(pivot_close) < 21:
            return pd.Series(0.0, index=stocks)
        
        # return from T-120 to T-20
        end_idx = max(0, len(pivot_close) - 20)
        start_idx = max(0, len(pivot_close) - 120)
        if start_idx == end_idx:
            return pd.Series(0.0, index=stocks)
            
        ret = pivot_close.iloc[end_idx-1] / pivot_close.iloc[start_idx] - 1
        return ret.reindex(stocks).fillna(0.0)
        
    def _calc_beta(self, date: str, stocks: pd.Index, price_df: pd.DataFrame, index_price_df: pd.DataFrame) -> pd.Series:
        if price_df is None or index_price_df is None or price_df.empty or index_price_df.empty:
            return pd.Series(0.0, index=stocks)
            
        p_df = price_df[price_df["trade_date"] <= date]
        idx_df = index_price_df[index_price_df["trade_date"] <= date]
        
        pivot_close = p_df.pivot(index="trade_date", columns="ts_code", values="close")
        idx_close = idx_df.set_index("trade_date")["close"]
        
        common_dates = pivot_close.index.intersection(idx_close.index)[-60:]
        if len(common_dates) < 5:
            return pd.Series(0.0, index=stocks)
            
        stock_rets = pivot_close.loc[common_dates].pct_change(fill_method=None).iloc[1:]
        idx_rets = idx_close.loc[common_dates].pct_change(fill_method=None).iloc[1:]
        
        # Calculate OLS beta vectorized: Cov(r_i, r_m) / Var(r_m)
        idx_var = idx_rets.var()
        if idx_var == 0 or pd.isna(idx_var):
            return pd.Series(0.0, index=stocks)
            
        cov_matrix = stock_rets.apply(lambda x: x.cov(idx_rets))
        beta = cov_matrix / idx_var
        return beta.reindex(stocks).fillna(0.0)

    def _calc_growth(self, date: str, stocks: pd.Index, fundamental_df: pd.DataFrame) -> pd.Series:
        if fundamental_df is None or fundamental_df.empty:
            return pd.Series(0.0, index=stocks)
        df = fundamental_df[fundamental_df["trade_date"] == date].set_index("ts_code")
        if "growth_rank" in df.columns:
            return df["growth_rank"].reindex(stocks).fillna(0.0)
        return pd.Series(0.0, index=stocks)
