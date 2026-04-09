from dataclasses import dataclass
import numpy as np

from data_models import ConstraintSet

@dataclass
class PenaltyMultipliers:
    industry_deviation: float
    style_deviation: float
    turnover_excess: float

class PenaltyCalibrator:
    """
    Dynamically calibrates penalty multipliers based on the scale of Alpha scores
    and the strictness of constraints to ensure the optimizer actually maximizes Alpha
    rather than just minimizing constraints.
    """
    def __init__(self, eta_industry: float = 5.0, eta_style: float = 3.0, eta_turnover: float = 4.0):
        self.eta_industry = eta_industry
        self.eta_style = eta_style
        self.eta_turnover = eta_turnover

    def calibrate(
        self,
        alpha_scores: np.ndarray,
        constraint_set: ConstraintSet,
    ) -> PenaltyMultipliers:
        # Avoid division by zero and get scale of objective
        alpha_scale = np.abs(alpha_scores).max() + 1e-8
        
        # Calculate dynamic multipliers
        ind_dev_ub = constraint_set.industry_deviation_ub + 1e-8
        ind_mult = self.eta_industry * alpha_scale / ind_dev_ub
        
        style_dev_ub = max(list(constraint_set.style_deviation_ub.values()) + [0.3]) + 1e-8
        style_mult = self.eta_style * alpha_scale / style_dev_ub
        
        to_ub = constraint_set.turnover_ub + 1e-8
        to_mult = self.eta_turnover * alpha_scale / to_ub
        
        return PenaltyMultipliers(
            industry_deviation=ind_mult,
            style_deviation=style_mult,
            turnover_excess=to_mult
        )
