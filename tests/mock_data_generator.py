import pandas as pd
import numpy as np
from typing import List, Dict
import os
import json

from data_models import AlphaFrame, MarketDataBundle, MarketStateSignal

def load_schema(schema_path: str) -> Dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_mock_alpha_frames(date: str, stocks: List[str]) -> List[AlphaFrame]:
    frames = []
    for domain in ["A", "B", "C", "D", "E"]:
        scores = pd.Series(np.random.randn(len(stocks)), index=stocks)
        frames.append(AlphaFrame(
            date=date,
            domain=domain,
            model_id=f"model_{domain}_v1",
            horizon=5,
            scores=scores,
            score_version="v1",
            available=True
        ))
    return frames

def _generate_from_schema(schema: Dict, num_rows: int, date: str, stocks: List[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for col, meta in schema.items():
        if col == "ts_code":
            df[col] = stocks[:num_rows]
        elif col == "trade_date":
            df[col] = [date] * num_rows
        elif meta["dtype"] in ["Float32", "Float64"]:
            df[col] = np.random.uniform(
                meta["value_range"][0], 
                meta["value_range"][1] if meta["value_range"][1] != meta["value_range"][0] else meta["value_range"][0] + 1.0, 
                num_rows
            )
        elif meta["dtype"] == "Int8":
            df[col] = np.random.randint(0, 2, num_rows)
        else:
            # Strings
            if "name" in col or "industry" in col or "l1_name" in col:
                df[col] = np.random.choice(["IndustryA", "IndustryB", "IndustryC"], num_rows)
            elif col == "suspend_type":
                df[col] = np.random.choice(["S", "R"], num_rows)
            else:
                df[col] = ["mock_string"] * num_rows
    return df

def generate_mock_market_data(date: str, stocks: List[str], schema_path: str) -> MarketDataBundle:
    schemas = load_schema(schema_path)
    n = len(stocks)
    
    stk_limit = _generate_from_schema(schemas["raw_stk_limit"], n, date, stocks)
    suspend = _generate_from_schema(schemas["raw_suspend_d"], n, date, stocks)
    daily_basic = _generate_from_schema(schemas["raw_daily_basic"], n, date, stocks)
    index_member = _generate_from_schema(schemas["raw_index_member_all"], n, date, stocks)
    price = _generate_from_schema(schemas["raw_daily"], n, date, stocks)
    fundamental = _generate_from_schema(schemas["feat_feature_D_fundamental"], n, date, stocks)
    
    benchmark_weights = pd.Series(1.0 / n, index=stocks)
    price_today = price.set_index("ts_code")["close"]
    
    # Covariance matrix (diagonal for simplicity)
    cov_matrix = pd.DataFrame(np.eye(n) * 0.01, index=stocks, columns=stocks)
    
    return MarketDataBundle(
        stk_limit=stk_limit,
        suspend=suspend,
        daily_basic=daily_basic,
        index_member=index_member,
        price=price,
        fundamental=fundamental,
        benchmark_weights=benchmark_weights,
        cov_matrix=cov_matrix,
        total_asset=100000000.0,
        price_today=price_today
    )
