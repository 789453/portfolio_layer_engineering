import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FactorCovEstimator:
    """
    Estimates the multi-factor covariance matrix:
    Sigma = B * F * B.T + Delta
    where:
    - B is the matrix of factor exposures (industry + style) for N stocks (N x K).
    - F is the factor covariance matrix (K x K).
    - Delta is the specific (idiosyncratic) variance matrix (diagonal N x N).
    
    In the absence of a pre-calculated factor return database, this estimator
    uses a cross-sectional regression approximation based on the current day's exposures B
    and historical stock returns.
    """
    
    def __init__(self, window: int = 60, shrinkage: float = 0.0):
        self.window = window
        self.shrinkage = shrinkage

    def estimate(
        self,
        date: str,
        stocks: pd.Index,
        industry_exposure: pd.DataFrame,
        style_exposure: pd.DataFrame,
        historical_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculates the covariance matrix for the given stocks.
        Returns a DataFrame of shape (N, N).
        """
        if stocks.empty:
            return pd.DataFrame()
            
        if historical_prices is None or historical_prices.empty:
            logger.warning(f"No historical prices provided to FactorCovEstimator. Returning diagonal matrix.")
            return pd.DataFrame(np.diag([0.02]*len(stocks)), index=stocks, columns=stocks)
            
        # Combine exposures to form B matrix
        B_parts = []
        if not industry_exposure.empty:
            B_parts.append(industry_exposure.reindex(stocks).fillna(0))
        if not style_exposure.empty:
            B_parts.append(style_exposure.reindex(stocks).fillna(0))
            
        if not B_parts:
            logger.warning(f"No factor exposures provided to FactorCovEstimator. Returning diagonal matrix.")
            return pd.DataFrame(np.diag([0.02]*len(stocks)), index=stocks, columns=stocks)
            
        B = pd.concat(B_parts, axis=1) # Shape: (N, K)
        
        # Prepare historical returns R matrix (T x N)
        # Filter prices up to current date
        p_df = historical_prices[historical_prices["trade_date"] <= date].copy()
        if p_df.empty:
            return pd.DataFrame(np.diag([0.02]*len(stocks)), index=stocks, columns=stocks)
            
        pivot_close = p_df.pivot(index="trade_date", columns="ts_code", values="close")
        if len(pivot_close) < 2:
            return pd.DataFrame(np.diag([0.02]*len(stocks)), index=stocks, columns=stocks)
            
        # Calculate daily returns and get last 'window' days
        rets = pivot_close.pct_change(fill_method=None).tail(self.window)
        
        # Align returns columns with stocks (T x N)
        R = rets.reindex(columns=stocks).fillna(0)
        
        # Cross-sectional regression: R(t) = f(t) * B.T + u(t)
        # We estimate f(t) (1 x K) for each day t using OLS: f(t) = R(t) * B * (B.T * B)^-1
        
        B_mat = B.values
        # To avoid singular matrix, we add a small ridge penalty to B.T * B
        BtB = B_mat.T @ B_mat
        ridge_lambda = 1e-4 * np.trace(BtB) / BtB.shape[0] if BtB.shape[0] > 0 else 1e-4
        BtB_inv = np.linalg.pinv(BtB + np.eye(BtB.shape[0]) * ridge_lambda)
        
        # B_pinv = (B.T * B)^-1 * B.T -> shape: (K, N)
        B_pinv = BtB_inv @ B_mat.T
        
        # Factor returns f (T x K) = R (T x N) * B_pinv.T (N x K)
        f_rets = R.values @ B_pinv.T
        
        # Factor covariance matrix F (K x K)
        # Multiply by 252 to annualize
        F = np.cov(f_rets, rowvar=False) * 252
        if F.ndim == 0:
            F = np.array([[F]])
            
        # Specific returns u (T x N) = R - f * B.T
        u_rets = R.values - (f_rets @ B_mat.T)
        
        # Specific variance Delta (N x N, diagonal)
        # Annualized
        specific_var = np.var(u_rets, axis=0) * 252
        
        # Shrinkage for specific variance (optional but good practice to avoid extreme 0 variances)
        median_var = np.median(specific_var) if len(specific_var) > 0 else 0.02
        specific_var = specific_var * (1 - self.shrinkage) + median_var * self.shrinkage
        # Avoid zero variance
        specific_var = np.clip(specific_var, 1e-4, np.inf)
        
        Delta = np.diag(specific_var)
        
        # Reconstruct full covariance matrix Sigma = B * F * B.T + Delta
        Sigma = B_mat @ F @ B_mat.T + Delta
        
        # Ensure symmetry
        Sigma = (Sigma + Sigma.T) / 2
        
        return pd.DataFrame(Sigma, index=stocks, columns=stocks)
