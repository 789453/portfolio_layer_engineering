from dataclasses import dataclass
from typing import Dict, Set, Tuple
import pandas as pd
import numpy as np

from data_models import CompositeAlphaFrame, CandidateUniverse

@dataclass
class CandidateSelectorConfig:
    min_listed_days: int = 60
    min_turnover_rate: float = 0.001
    exclude_st: bool = True
    exclude_limit_up_for_buy: bool = True
    exclude_limit_down_for_sell: bool = True
    market_cap_filter_pct: float = 0.0

class CandidateSelector:
    def __init__(self, config: CandidateSelectorConfig):
        self.config = config

    def build(
        self,
        date: str,
        composite_alpha: CompositeAlphaFrame,
        stk_limit_df: pd.DataFrame,
        suspend_df: pd.DataFrame,
        daily_basic_df: pd.DataFrame,
        index_member_df: pd.DataFrame,
        price_df: pd.DataFrame = None, # Added price_df for close vs up_limit
    ) -> CandidateUniverse:
        all_stocks = composite_alpha.composite_score.index
        exclusion_reason: Dict[str, str] = {}

        # 1. 停牌
        suspended = self._get_suspended(date, suspend_df)
        for s in suspended:
            exclusion_reason[s] = "suspended"

        # 2. 涨跌停 (Do not exclude them entirely, just pass them to optimizer for asymmetric bounds)
        limit_up, limit_down = self._get_limits(date, stk_limit_df, price_df)
        
        if self.config.exclude_limit_up_for_buy:
            pass # Kept in universe but handled in ConstraintBuilder
        if self.config.exclude_limit_down_for_sell:
            pass # Kept in universe but handled in ConstraintBuilder

        # 3. 新股
        new_stocks = self._get_new_stocks(date, index_member_df, self.config.min_listed_days)
        for s in new_stocks:
            exclusion_reason.setdefault(s, "new_listing")

        # 4. 极低流动性
        illiquid = self._get_illiquid(date, daily_basic_df, self.config.min_turnover_rate)
        for s in illiquid:
            exclusion_reason.setdefault(s, "illiquid")

        # 5. ST股
        if self.config.exclude_st:
            st_stocks = self._get_st_stocks(index_member_df)
            for s in st_stocks:
                exclusion_reason.setdefault(s, "st_stock")

        # 6. 市值过滤
        if self.config.market_cap_filter_pct > 0:
            small_caps = self._get_small_caps(date, daily_basic_df, self.config.market_cap_filter_pct)
            for s in small_caps:
                exclusion_reason.setdefault(s, "small_cap")

        excluded = pd.Index(list(exclusion_reason.keys()))
        eligible = all_stocks.difference(excluded)

        valid_alpha = composite_alpha.composite_score.loc[eligible.intersection(composite_alpha.composite_score.index)].dropna()
        primary = valid_alpha.index
        reserve = eligible.difference(primary)

        return CandidateUniverse(
            date=date,
            primary=primary,
            reserve=reserve,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            limit_up_stocks=pd.Index(list(limit_up)),
            limit_down_stocks=pd.Index(list(limit_down)),
        )

    def _get_suspended(self, date: str, suspend_df: pd.DataFrame) -> Set[str]:
        if suspend_df is None or suspend_df.empty: return set()
        # suspend_date <= date AND (resume_date IS NULL OR resume_date > date)
        if "suspend_date" in suspend_df.columns:
            df = suspend_df[(suspend_df["suspend_date"] <= date)]
            if "resume_date" in df.columns:
                df = df[df["resume_date"].isna() | (df["resume_date"] > date)]
            return set(df["ts_code"].unique())
        # fallback to previous logic if suspend_date not available but trade_date is
        elif "trade_date" in suspend_df.columns:
            df = suspend_df[suspend_df["trade_date"] == date]
            if "suspend_type" in df.columns:
                return set(df[df["suspend_type"] == "S"]["ts_code"].unique())
        return set()

    def _get_limits(self, date: str, stk_limit_df: pd.DataFrame, price_df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
        if stk_limit_df is None or stk_limit_df.empty: return set(), set()
        df = stk_limit_df[stk_limit_df["trade_date"] == date]
        if df.empty: return set(), set()
        
        if price_df is not None and not price_df.empty:
            p_df = price_df[price_df["trade_date"] == date].set_index("ts_code")
            df = df.set_index("ts_code")
            joined = df.join(p_df[["close"]], how="inner")
            up = set(joined[joined["close"] >= joined["up_limit"] - 1e-4].index)
            down = set(joined[joined["close"] <= joined["down_limit"] + 1e-4].index)
            return up, down
        else:
            return set(), set()

    def _get_new_stocks(self, date: str, index_member_df: pd.DataFrame, min_days: int) -> Set[str]:
        if index_member_df is None or index_member_df.empty: return set()
        if "in_date" not in index_member_df.columns: return set()
        
        # Calculate days since in_date
        date_ts = pd.to_datetime(date)
        df = index_member_df.drop_duplicates("ts_code").copy()
        df["in_date"] = pd.to_datetime(df["in_date"], errors="coerce")
        df = df.dropna(subset=["in_date"])
        df["listed_days"] = (date_ts - df["in_date"]).dt.days
        return set(df[df["listed_days"] < min_days]["ts_code"].unique())

    def _get_illiquid(self, date: str, daily_basic_df: pd.DataFrame, min_turnover: float) -> Set[str]:
        if daily_basic_df is None or daily_basic_df.empty: return set()
        df = daily_basic_df[daily_basic_df["trade_date"] == date]
        if df.empty: return set()
        return set(df[df["turnover_rate"] < min_turnover * 100]["ts_code"].unique())

    def _get_st_stocks(self, index_member_df: pd.DataFrame) -> Set[str]:
        if index_member_df is None or index_member_df.empty: return set()
        if "name" not in index_member_df.columns: return set()
        st = index_member_df[index_member_df["name"].str.contains("ST", na=False)]
        return set(st["ts_code"].unique())

    def _get_small_caps(self, date: str, daily_basic_df: pd.DataFrame, pct: float) -> Set[str]:
        if daily_basic_df.empty: return set()
        df = daily_basic_df[daily_basic_df["trade_date"] == date]
        if df.empty: return set()
        threshold = df["total_mv"].quantile(pct)
        return set(df[df["total_mv"] < threshold]["ts_code"].unique())
