# `portfolio_layer` 成熟化改造专业文档

**文档编号**：DOC-PORTFOLIO-003\
**版本**：v2.0.0\
**状态**：改造规范草案\
**基准文档**：DOC-PORTFOLIO-002 v1.0.0 + 评估报告 2026-03-25\
**目标**：将现有项目从"生产就绪仿真"阶段推进至"机构级生产可用"阶段

***

## 一、总体改造纲领与原则

### 1.1 现状诊断摘要

当前项目存在三个层次的问题，必须分别对症施策：

**工程结构层**：各子模块的实际业务逻辑大量堆积在 `__init__.py` 文件中，违背了 Python 包设计中"初始化文件只做对外导出"的基本原则。这导致模块的可读性、可测试性和可维护性严重受损——任何一处修改都可能造成导入链路的连锁反应。

**金融逻辑层**：协方差矩阵采用对角近似（`np.diag([0.02]*N)`），使得二次规划优化器名存实亡；涨跌停的处理逻辑抹去了"买入方向"与"卖出方向"的本质不对称性；软约束惩罚系数硬编码在目标函数内部，既无法自适应调参，也无法在超参数搜索中被纳入。

**测试与可观测性层**：`boundary/`、`robustness/`、`stress/`、`consistency/` 等测试目录均为空目录；`reporting/templates/` 为空；`configs/schemas/` 为空；`risk_model/covariance/` 与 `risk_model/factor_definitions/` 均为空目录——这些空目录恰恰对应着最关键的生产级能力缺口。

### 1.2 改造原则

改造不等于膨胀。本文档遵循以下核心原则：

- **数学先行**：工程改造服从金融逻辑的正确性，不允许为了代码美观而牺牲定量模型的严谨性。
- **最小外部依赖**：优先利用 `numpy`、`scipy`、`pandas`、`cvxpy` 等已在项目中出现的库；引入新依赖须有充分理由。
- **接口向后兼容**：对外暴露的 `data_models` 契约不做破坏性修改，仅做增量扩展。
- **可测试优先**：每个新增模块在设计时就必须考虑如何注入 Mock 数据进行单元测试。

***

## 二、工程结构全面重构

### 2.1 `__init__.py` 瘦身与职责分离

当前项目中，`signal_fusion/__init__.py`、`optimizer/__init__.py`、`risk_model/__init__.py` 等文件同时承担了"业务逻辑实现"和"包导出声明"两个角色。重构后，每个 `__init__.py` 只允许出现以下内容：

```
# signal_fusion/__init__.py （重构后样例）
from .alpha_combiner import AlphaCombiner
from .single_model import SingleModelFusion
from .weighted_average import WeightedAverageFusion
from .multi_domain import MultiDomainFusion
from .hierarchical import HierarchicalFusion

__all__ = [
    "AlphaCombiner",
    "SingleModelFusion",
    "WeightedAverageFusion",
    "MultiDomainFusion",
    "HierarchicalFusion",
]

```

每个子模块的完整文件清单应严格对应 `DOC-PORTFOLIO-002` 第 12 节定义的目录结构，当前空置的文件均须落地实现。

### 2.2 配置系统的完整化

`configs/` 目录当前仅有空的 `schemas/` 子目录，须补齐以下文件：

**`configs/base.yaml`**（基础配置，所有环境继承）：

```
optimizer:
  risk_aversion: 1.0
  turnover_penalty: 0.5
  penalty_multipliers:
    industry_deviation: 10.0    # 不再硬编码在代码中
    style_deviation: 5.0
    turnover_excess: 8.0
  solver: CLARABEL
  fallback_solver: SCS
  max_solve_time: 45.0
  fallback_topk: 50

candidate_selector:
  min_listed_days: 60
  min_turnover_rate: 0.001
  small_cap_threshold_wan: 20000   # 2亿元，单位万元
  exclude_st: true

risk_model:
  style_factors: [size, value, momentum, long_momentum, volatility, liquidity, beta]
  momentum_window: 20
  long_momentum_window: 120
  long_momentum_skip: 20
  beta_window: 60
  liquidity_window: 20
  covariance:
    method: ledoit_wolf      # 可选: diagonal, sample, ledoit_wolf, factor_model
    halflife_days: 63        # 因子收益半衰期（用于指数加权协方差）
    min_history_days: 120

position_scaler:
  min_gross_exposure: 0.60
  max_gross_exposure: 1.00
  scale_smoothing: 0.30

constraint_builder:
  max_single_stock_weight: 0.05
  industry_deviation_ub: 0.05
  style_deviation_ub:
    size: 0.5
    momentum: 0.3
  turnover_ub: 0.30           # 单边换手率上限
  total_weight_lb: 0.98
  total_weight_ub: 1.02

```

**`configs/schemas/config_schema.yaml`** 须使用 JSON Schema 或 Pydantic 模型对上述配置做类型与范围校验，防止非法配置在运行时才暴露。

### 2.3 数据模型扩展

现有 `data_models` 须新增两个数据类以支持改造后的功能：

```
# data_models/covariance_model.py
@dataclass
class FactorCovarianceModel:
    """因子协方差模型的完整表示"""
    date: str
    factor_names: List[str]           # 行业因子 + 风格因子名称列表
    factor_cov: np.ndarray            # 形状 (K, K)，因子协方差矩阵 F
    specific_risk: pd.Series          # index=ts_code，特质风险 σ_i（年化标准差）
    factor_exposure: pd.DataFrame     # 形状 (N, K)，即 B 矩阵
    estimation_method: str            # "ledoit_wolf" / "factor_model" / "diagonal"
    effective_sample_days: int        # 用于估计的有效历史天数

# data_models/asymmetric_constraint.py
@dataclass
class AsymmetricTradabilityMask:
    """涨跌停不对称可交易掩码"""
    date: str
    cannot_buy: pd.Index      # 涨停（当前无仓位者不可建仓）
    cannot_sell: pd.Index     # 跌停（当前有仓位者不可减仓）
    forced_weight_lb: pd.Series  # 对跌停持仓股，下界锁定为 w_prev[i]
    forced_weight_ub: pd.Series  # 对涨停非持仓股，上界锁定为 0

```

***

## 三、风险模型的核心升级：因子协方差矩阵

这是整个改造工程中数学分量最重的部分，也是当前系统最大的金融逻辑缺陷所在。

### 3.1 Barra 式因子协方差模型的正确实现

完整的多因子协方差矩阵由以下公式给出：

\[\
\Sigma = B \cdot F \cdot B^T + \Delta\
]

其中 (B \in \mathbb{R}^{N \times K}) 为因子暴露矩阵（由 `RiskExposureBuilder` 产出），(F \in \mathbb{R}^{K \times K}) 为因子收益率协方差矩阵，(\Delta = \mathrm{diag}(\sigma\_1^2, \ldots, \sigma\_N^2)) 为特质风险的对角矩阵。

这一结构之所以优于样本协方差矩阵，在于：当 (N \gg T)（股票数远大于历史天数）时，样本协方差矩阵必然是奇异的，而因子协方差结构通过将 (N) 维问题压缩到 (K) 维（(K) 通常为行业数 + 风格因子数，约 30–50），从根本上规避了这一问题。

### 3.2 因子收益率的估计

**步骤一：截面回归获取因子收益率时间序列**

对历史每个交易日 (t)，运行横截面加权最小二乘（WLS）回归：

\[\
r\_{i,t} = \sum\_{k=1}^{K} B\_{ik} \cdot f\_{k,t} + \epsilon\_{i,t}\
]

其中 (r\_{i,t}) 为股票 (i) 在第 (t) 日的收益率，权重矩阵取 (\sqrt{\text{circ\_mv}*{i,t}})（市值加权，压制小市值股对回归的扭曲影响），(f*{k,t}) 即为待估因子收益率向量。每日回归得到长度为 (T) 的因子收益率时间序列矩阵 (\mathbf{f} \in \mathbb{R}^{T \times K})。

**步骤二：指数加权因子协方差估计**

对因子收益率序列施以指数加权，半衰期 (\tau\_{1/2}) 建议取 63 个交易日（约一个季度）：

\[\
w\_t = \left(\frac{1}{2}\right)^{(T - t) / \tau\_{1/2}}, \quad \tilde{w}\_t = \frac{w\_t}{\sum\_s w\_s}\
]

\[\
\bar{f}*k = \sum\_t \tilde{w}t f{k,t}, \quad F*{kl} = \sum\_t \tilde{w}*t (f*{k,t} - \bar{f}*k)(f*{l,t} - \bar{f}\_l)\
]

**步骤三：Ledoit-Wolf 收缩稳健估计**

当因子数量 (K) 相对于历史窗口 (T) 不可忽略时，样本协方差矩阵 (F) 仍存在估计误差。引入 Oracle 近似收缩（OAS，即 `sklearn` 中 `OAS` 估计器）：

\[\
\hat{F} = (1 - \alpha) F\_{\text{sample}} + \alpha \cdot \mu\_F \cdot I\_K\
]

其中 (\alpha) 和 (\mu\_F) 均由数据驱动确定，无需人工调节，这是相比 Ledoit-Wolf 线性收缩的改进。

**步骤四：特质风险估计**

对每只股票，特质收益率为 (\hat{\epsilon}*{i,t} = r*{i,t} - B\_i^T f\_t)，特质方差估计为：

\[\
\hat{\sigma}\_i^2 = \sum\_t \tilde{w}*t \hat{\epsilon}*{i,t}^2\
]

为防止极端小票的特质风险过低（导致优化器把权重集中于此），须对 (\hat{\sigma}\_i) 施以截面下截断：

\[\
\hat{\sigma}\_i^{\text{adj}} = \max\left(\hat{\sigma}*i, ; P*{10}(\hat{\sigma})\right)\
]

即特质风险不低于截面第 10 百分位数。

### 3.3 `FactorCovEstimator` 实现规范

```
# risk_model/covariance/factor_cov_estimator.py

class FactorCovEstimator:
    """
    多因子协方差矩阵估计器。
    支持：diagonal（MVP降级）/ ledoit_wolf / factor_model
    """
    def __init__(self, config: CovarianceConfig): ...

    def estimate(
        self,
        date: str,
        factor_exposure: pd.DataFrame,     # B 矩阵，来自 RiskExposureFrame
        factor_return_history: pd.DataFrame, # 历史因子收益率，shape=(T, K)
        stock_return_history: pd.DataFrame,  # 历史个股收益率，shape=(T, N)
    ) -> FactorCovarianceModel:
        """主入口：根据配置选择估计方法"""
        if self.config.method == "diagonal":
            return self._diagonal_fallback(date, factor_exposure)
        elif self.config.method == "ledoit_wolf":
            return self._ledoit_wolf_estimate(date, factor_exposure, factor_return_history)
        elif self.config.method == "factor_model":
            return self._factor_model_estimate(
                date, factor_exposure,
                factor_return_history, stock_return_history
            )

    def to_dense_matrix(self, model: FactorCovarianceModel) -> np.ndarray:
        """将 FactorCovarianceModel 展开为 N×N 稠密矩阵（仅在 N 较小时使用）"""
        B = model.factor_exposure.values
        F = model.factor_cov
        delta = np.diag(model.specific_risk.values ** 2)
        return B @ F @ B.T + delta

    def compute_quad_form_efficient(
        self,
        w: np.ndarray,
        model: FactorCovarianceModel,
    ) -> float:
        """
        高效计算二次型 w^T Σ w，利用因子结构避免 N×N 矩阵显式构建：
        w^T Σ w = (B^T w)^T F (B^T w) + Σ_i σ_i^2 w_i^2
        时间复杂度 O(NK + K²) 而非 O(N²)
        """
        B = model.factor_exposure.values
        factor_loading = B.T @ w          # shape (K,)
        risk_factor = float(factor_loading @ model.factor_cov @ factor_loading)
        risk_specific = float(np.dot(w ** 2, model.specific_risk.values ** 2))
        return risk_factor + risk_specific

```

### 3.4 优化器中的高效二次型集成

在优化器中，不再将 (\Sigma) 展开为 (N \times N) 稠密矩阵，而是利用因子结构的等价变换：

\[\
w^T \Sigma w = \underbrace{(B^T w)^T F (B^T w)}*{\text{因子风险}} + \underbrace{\sum\_i \sigma\_i^2 w\_i^2}*{\text{特质风险}}\
]

在 `cvxpy` 中，这对应：

```
# optimizer/portfolio_optimizer.py（核心改动）

B = cov_model.factor_exposure.reindex(stocks).values   # (N, K)
F_chol = np.linalg.cholesky(cov_model.factor_cov)      # (K, K)
sigma_specific = cov_model.specific_risk.reindex(stocks).values  # (N,)

# 因子风险项：‖F_chol^T B^T w‖₂² = (B^T w)^T F (B^T w)
factor_loading = F_chol.T @ B.T @ w      # cvxpy expression, shape (K,)
risk_factor_term = cp.sum_squares(factor_loading)

# 特质风险项：w^T diag(σ²) w = ‖diag(σ) w‖₂²
risk_specific_term = cp.sum_squares(cp.multiply(sigma_specific, w))

risk_penalty = self.config.risk_aversion * (risk_factor_term + risk_specific_term)
objective_terms.append(-risk_penalty)

```

这一实现将协方差二次型的计算从 (O(N^2)) 降低到 (O(NK + K^2))，对 (N=3000)、(K=50) 的场景，理论加速比约为 1000 倍。

***

## 四、涨跌停不对称约束的正确建模

### 4.1 问题的根本性质

涨跌停过滤的"一刀切"做法在两个方向都会产生错误的调仓指令：

- **跌停持仓**：优化器若不知当前持有该股，便会在 `rebalance_list` 中生成"卖出至 0"的指令，但该指令在 T 日无法成交，导致执行层收到幻象指令，引发持仓与系统登记不一致。
- **涨停非持仓**：若一只涨停股的 Alpha 分数极高，直接从候选池剔除意味着优化器无法在结果中为其预留"明日建仓"的意图，从而在下一个可交易日产生大幅滞后买入。

### 4.2 不对称约束的数学表示

设当日持仓向量为 (w\_{\text{prev}})，令 (\mathcal{L}*{\text{up}}) 表示涨停股集合，(\mathcal{L}*{\text{dn}}) 表示跌停股集合，则不对称约束为：

**针对跌停持仓股**（(i \in \mathcal{L}*{\text{dn}}) 且 (w*{\text{prev},i} > 0)）：

\[\
w\_i \geq w\_{\text{prev},i} \quad \text{（不允许减仓，物理无法卖出）}\
]

**针对涨停非持仓股**（(i \in \mathcal{L}*{\text{up}}) 且 (w*{\text{prev},i} = 0)）：

\[\
w\_i = 0 \quad \text{（不允许新建仓，物理无法买入）}\
]

**针对涨停持仓股**（(i \in \mathcal{L}*{\text{up}}) 且 (w*{\text{prev},i} > 0)）：

\[\
0 \leq w\_i \leq w\_{\text{ub},i} \quad \text{（可减仓或继续持有，不可加仓至高于上限）}\
]

**针对跌停非持仓股**（(i \in \mathcal{L}*{\text{dn}}) 且 (w*{\text{prev},i} = 0)）：

可选择性地允许"抄底买入"，如业务上禁止则设 (w\_i = 0)。建议通过配置项 `allow_buy_limit_down: false` 控制。

### 4.3 `AsymmetricTradabilityMask` 在优化器中的集成

```
# candidate_selection/filters/tradability.py

def build_asymmetric_mask(
    date: str,
    price_df: pd.DataFrame,
    stk_limit_df: pd.DataFrame,
    prev_weights: pd.Series,
    tolerance: float = 1e-4,
) -> AsymmetricTradabilityMask:
    """
    构建涨跌停不对称可交易掩码。
    涨停判定：close >= up_limit - tolerance
    跌停判定：close <= down_limit + tolerance
    """
    day_df = price_df[price_df["trade_date"] == date].set_index("ts_code")
    limit_df = stk_limit_df[stk_limit_df["trade_date"] == date].set_index("ts_code")
    merged = day_df[["close"]].join(limit_df[["up_limit", "down_limit"]], how="inner")

    limit_up_mask = merged["close"] >= merged["up_limit"] - tolerance
    limit_dn_mask = merged["close"] <= merged["down_limit"] + tolerance

    limit_up_stocks = merged.index[limit_up_mask]
    limit_dn_stocks = merged.index[limit_dn_mask]

    all_stocks = merged.index
    forced_lb = pd.Series(0.0, index=all_stocks)
    forced_ub = pd.Series(np.inf, index=all_stocks)

    # 跌停且有持仓：下界锁定
    held_limit_dn = limit_dn_stocks.intersection(prev_weights[prev_weights > 0].index)
    forced_lb[held_limit_dn] = prev_weights.reindex(held_limit_dn, fill_value=0.0)

    # 涨停且无持仓：上界锁定为 0
    unheld_limit_up = limit_up_stocks.difference(prev_weights[prev_weights > 0].index)
    forced_ub[unheld_limit_up] = 0.0

    cannot_buy = unheld_limit_up
    cannot_sell = held_limit_dn

    return AsymmetricTradabilityMask(
        date=date,
        cannot_buy=cannot_buy,
        cannot_sell=cannot_sell,
        forced_weight_lb=forced_lb,
        forced_weight_ub=forced_ub,
    )

```

***

## 五、优化器惩罚系数的自适应标定

### 5.1 惩罚系数硬编码的危害分析

当前代码中 `-10.0 * deviation` 这类表达式的问题在于：目标函数中各项的量纲必须可公度，才能让乘法系数发挥有意义的权衡作用。Alpha Z-score 的数量级约为 (\[-3, 3])，单只股票权重约为 (\[0, 0.05])，因此 (\mu^T w) 的数量级约为 (0.1)–(0.3)；而行业偏差变量 `deviation` 的数量级约为 (\[0, 0.1])，若惩罚系数为 10，惩罚项数量级达 1，远超 Alpha 目标，优化器实质上退化为"最小化约束违约量"而非"最大化 Alpha"。

### 5.2 自适应惩罚系数标定方案

引入一个在每次优化求解前动态计算的 `PenaltyCalibrator`：

\[\
\rho\_{\text{ind}} = \eta\_{\text{ind}} \cdot \frac{|\mu|*\infty}{\delta*{\text{ind}}^{\text{max}}}\
]

其中 (|\mu|*\infty) 为 Alpha 向量的最大绝对值（刻画目标函数的"规模"），(\delta*{\text{ind}}^{\text{max}}) 为允许的最大行业偏差（刻画约束违约的"规模"），(\eta\_{\text{ind}}) 为无量纲强度倍数（通过超参数调优确定，建议初始值为 5）。类似地：

\[\
\rho\_{\text{sty}} = \eta\_{\text{sty}} \cdot \frac{|\mu|*\infty}{\epsilon*{\text{sty}}^{\text{max}}}\
]

\[\
\rho\_{\text{to}} = \eta\_{\text{to}} \cdot \frac{|\mu|*\infty}{\tau*{\text{max}}}\
]

这三个公式确保了不论 Alpha 分数的整体水平如何变化，惩罚系数都能自动与之对齐。

```
# optimizer/penalty_calibrator.py

@dataclass
class PenaltyMultipliers:
    industry_deviation: float
    style_deviation: float
    turnover_excess: float

class PenaltyCalibrator:
    def __init__(self, eta: dict):
        """eta: {'industry': 5.0, 'style': 3.0, 'turnover': 4.0}"""
        self.eta = eta

    def calibrate(
        self,
        alpha_scores: np.ndarray,
        constraint_set: ConstraintSet,
    ) -> PenaltyMultipliers:
        alpha_scale = np.abs(alpha_scores).max() + 1e-8
        return PenaltyMultipliers(
            industry_deviation=self.eta["industry"] * alpha_scale / (
                constraint_set.industry_deviation_ub + 1e-8),
            style_deviation=self.eta["style"] * alpha_scale / (
                max(constraint_set.style_deviation_ub.values(), default=0.3) + 1e-8),
            turnover_excess=self.eta["turnover"] * alpha_scale / (
                constraint_set.turnover_ub + 1e-8),
        )

```

### 5.3 换手率度量统一规范

全系统统一使用**单边换手率**定义：

\[\
\text{TO}*{\text{single}} = \frac{1}{2} \sum\_i |w\_i - w*{\text{prev},i}|\
]

在优化器目标函数的换手惩罚项中，须做相应换算：

\[\
\text{penalty}*{\text{turnover}} = -\rho*{\text{to}} \cdot \underbrace{\frac{1}{2} |w - w\_{\text{prev}}|*1}*{\text{单边换手率}}\
]

在 `cvxpy` 中：

```
single_sided_turnover = 0.5 * cp.norm1(w - w_prev)
objective_terms.append(-penalty_multipliers.turnover_excess * single_sided_turnover)
# 换手上限约束（同样使用单边定义）
constraints.append(single_sided_turnover <= constraint_set.turnover_ub)

```

所有在 `RiskReporter`、日志输出和 `TargetPortfolio.meta` 中报告的换手率，均须使用上述单边定义，并在字段名和注释中明确标注 `single_sided`。

***

## 六、大规模宇宙优化的性能解决方案

### 6.1 两阶段优化架构

针对全市场约 5000 只股票的场景，采用如下两阶段方案：

**阶段一（粗筛，线性规划，(O(N))）**：

构建一个简化的线性规划问题，仅使用 Alpha 分数和行业平衡约束，快速筛出约 500 只股票的核心池：

\[\
\max\_{w} ; \mu^T w \quad \text{s.t.} ; w \geq 0, ; \sum w = 1, ; w\_i \leq 0.02\
]

取权重前 500 的非零股票（或直接取 Alpha 分数前 500）构成精简候选池 (\mathcal{S}\_{500})。

**阶段二（精优化，带完整约束的 QP，(N=500)）**：

仅对 (\mathcal{S}\_{500}) 运行完整的二次规划（含因子协方差、软约束、换手惩罚），典型求解时间在 5 秒以内。

```
# optimizer/two_stage_optimizer.py

class TwoStageOptimizer:
    def __init__(
        self,
        stage1_topk: int = 500,
        stage2_optimizer: PortfolioOptimizer = None,
    ): ...

    def optimize(self, composite_alpha, risk_exposure,
                 constraint_set, prev_weights, cov_model) -> Tuple[pd.Series, str]:
        # Stage 1: 快速筛选
        stage1_stocks = self._fast_screen(
            composite_alpha, constraint_set, k=self.stage1_topk
        )
        # Stage 2: 精确优化
        alpha_subset = composite_alpha.filter(stage1_stocks)
        exposure_subset = risk_exposure.filter(stage1_stocks)
        constraint_subset = constraint_set.filter(stage1_stocks)
        cov_subset = cov_model.filter(stage1_stocks)
        return self.stage2_optimizer.optimize(
            alpha_subset, exposure_subset,
            constraint_subset, prev_weights, cov_subset
        )

```

### 6.2 求解器选择策略

建议实现一个 `SolverRouter`，根据问题规模动态选择求解器：

股票数 (N)

协方差是否引入

推荐求解器

备注

(N \leq 200)

可选

`CLARABEL`

最稳定

(200 < N \leq 800)

是

`CLARABEL` 或 `MOSEK`（若有许可证）

<br />

(N > 800)

是

`OSQP`（warm start）

迭代求解器，大规模更快

任意（降级）

否

`SCS`

鲁棒性最强

`OSQP` 对 L1 范数（换手惩罚）的处理需引入辅助变量 (t\_i \geq |w\_i - w\_{\text{prev},i}|)，`cvxpy` 会自动处理这一转化，无需手工实现。

***

## 七、测试框架的完整建设

当前项目中 `boundary/`、`robustness/`、`stress/`、`consistency/` 均为空目录，以下给出各层测试的具体实现规范。

### 7.1 `mock_data_generator.py` 的生产级扩展

现有的 `tests/mock_data_generator.py` 须支持以下场景的数据生成：

```
class ScenarioDataGenerator:
    """生成特定压力场景的测试数据"""

    def entire_industry_halted(
        self, date: str, industry_code: str, n_stocks: int = 30
    ) -> MockMarketBundle:
        """整个行业全部停牌的场景"""
        ...

    def extreme_alpha_concentration(
        self, date: str, top_k: int = 5, concentration_ratio: float = 0.95
    ) -> MockAlphaBundle:
        """Alpha 分数极度集中于少数股票的场景"""
        ...

    def massive_limit_up_day(
        self, date: str, limit_up_ratio: float = 0.30
    ) -> MockMarketBundle:
        """全市场 30% 股票涨停的场景（如政策利好当天）"""
        ...

    def near_singular_covariance(
        self, n_stocks: int = 300, condition_number: float = 1e8
    ) -> np.ndarray:
        """接近病态的协方差矩阵，用于测试数值稳定性"""
        ...

```

### 7.2 边界测试（`tests/boundary/`）

**`test_limit_up_down_asymmetry.py`**——验证不对称约束的核心逻辑：

```
class TestAsymmetricTradability:
    def test_held_limit_down_stock_not_sold(self):
        """持有跌停股时，目标权重不低于当前持仓权重"""
        prev = pd.Series({"000001.SZ": 0.03, "000002.SZ": 0.02})
        # 000001.SZ 当日跌停
        mask = build_asymmetric_mask(date, price_df_limit_dn, limit_df, prev)
        result_weight = optimizer.optimize(..., asymmetric_mask=mask)
        assert result_weight["000001.SZ"] >= prev["000001.SZ"] - 1e-6

    def test_new_limit_up_stock_not_bought(self):
        """无持仓的涨停股，目标权重为 0"""
        prev = pd.Series({"000002.SZ": 0.03})
        # 000003.SZ 当日涨停，且 prev 中无此股
        result_weight = optimizer.optimize(...)
        assert result_weight.get("000003.SZ", 0.0) < 1e-6

```

### 7.3 一致性测试（`tests/consistency/`）

**`test_no_lookahead.py`**——数据血缘追踪：

```
class TestNoLookahead:
    def test_candidate_selector_uses_only_t_day_data(self):
        """候选池构建只允许访问 date 当日及之前的数据"""
        # 通过 mock 拦截 DataLoader 的所有访问，记录实际使用的 trade_date
        with DataAccessTracker() as tracker:
            selector.build(date="20240701", ...)
        accessed_dates = tracker.get_accessed_dates()
        assert all(d <= "20240701" for d in accessed_dates), \
            f"候选池构建访问了未来数据: {[d for d in accessed_dates if d > '20240701']}"

```

### 7.4 稳健性测试（`tests/robustness/`）

**`test_covariance_robustness.py`**：

```
@pytest.mark.parametrize("condition_number", [1e2, 1e4, 1e6, 1e8])
def test_optimizer_handles_ill_conditioned_cov(condition_number):
    """病态协方差矩阵下优化器正常返回合法权重"""
    cov_model = ScenarioDataGenerator().near_singular_covariance(
        n_stocks=200, condition_number=condition_number
    )
    result, status = optimizer.optimize(..., cov_model=cov_model)
    assert status in ["optimal", "optimal_inaccurate", "degraded"]
    assert abs(result.sum() - 1.0) < 0.02
    assert (result >= -1e-6).all()

```

***

## 八、`PortfolioPipeline` 的可观测性与调试能力

### 8.1 结构化日志规范

当前项目的日志输出是非结构化的字符串。须改造为结构化 JSON 日志，便于下游监控系统（如 Grafana、ELK）接入：

```
# pipeline/portfolio_pipeline.py

import structlog
log = structlog.get_logger()

# 在 pipeline.run() 的关键节点：
log.info(
    "alpha_fusion_complete",
    date=date,
    fusion_method=composite_alpha.fusion_method,
    n_stocks_with_alpha=composite_alpha.composite_score.notna().sum(),
    is_degraded=composite_alpha.is_degraded,
    degraded_domains=composite_alpha.degraded_domains,
)

log.info(
    "optimization_complete",
    date=date,
    solver_status=optimizer_status,
    n_holdings=(raw_weight > 0).sum(),
    single_sided_turnover=float(turnover),
    gross_exposure=float(gross_exposure),
    solve_time_seconds=solve_time,
)

```

### 8.2 每日诊断快照

在 `pipeline/daily_runner.py` 中，须将每次运行的关键中间状态序列化落盘，用于复盘和回放：

```
# outputs/{date}/diagnostics/
# ├── composite_alpha.parquet       # 融合后 Alpha 分数
# ├── candidate_universe.parquet    # 候选池标记
# ├── risk_exposure.parquet         # 风险暴露矩阵
# ├── constraint_set.json           # 约束参数
# ├── covariance_model_metadata.json # 协方差估计元信息
# └── optimization_diagnostics.json # 求解时间、状态、惩罚系数

```

### 8.3 约束违约后验检查

在 `RiskReporter.report()` 中，须实现完整的后验约束检查：

```
def _check_violations(
    self,
    portfolio: TargetPortfolio,
    risk_exposure: RiskExposureFrame,
    constraint_set: ConstraintSet,
) -> List[str]:
    violations = []
    w = portfolio.target_weight
    # 行业偏离检查
    for ind in risk_exposure.industry_exposure.columns:
        ind_vec = risk_exposure.industry_exposure[ind]
        port_w = (ind_vec * w.reindex(ind_vec.index, fill_value=0)).sum()
        bm_w = (ind_vec * risk_exposure.benchmark_weights.reindex(
            ind_vec.index, fill_value=0)).sum()
        if abs(port_w - bm_w) > constraint_set.industry_deviation_ub + 1e-4:
            violations.append(
                f"INDUSTRY_DEVIATION:{ind}:{(port_w - bm_w):.4f}"
            )
    # 单票上限检查
    exceed = w[w > constraint_set.max_single_stock_weight + 1e-4]
    for ts, wt in exceed.items():
        violations.append(f"SINGLE_STOCK_EXCEED:{ts}:{wt:.4f}")
    return violations

```

***

## 九、降级管理器的层次化增强

### 9.1 降级事件的可追溯性

现有 `DegradationManager` 仅选择融合方法，须扩展为完整的降级事件记录系统：

```
# degradation/degradation_manager.py

@dataclass
class DegradationEvent:
    date: str
    trigger: str          # "domain_missing" / "optimizer_infeasible" / "empty_pool"
    severity: str         # "WARNING" / "ERROR" / "CRITICAL" / "FATAL"
    detail: str
    action_taken: str
    affected_module: str

class DegradationManager:
    def __init__(self, config: DegradationConfig):
        self.config = config
        self._events: List[DegradationEvent] = []

    def record_event(self, event: DegradationEvent) -> None:
        self._events.append(event)
        # 同时推送至结构化日志
        log.warning("degradation_event", **dataclasses.asdict(event))

    def get_daily_summary(self, date: str) -> Dict:
        """汇总当日所有降级事件，写入诊断快照"""
        day_events = [e for e in self._events if e.date == date]
        return {
            "date": date,
            "total_events": len(day_events),
            "fatal_count": sum(1 for e in day_events if e.severity == "FATAL"),
            "critical_count": sum(1 for e in day_events if e.severity == "CRITICAL"),
            "events": [dataclasses.asdict(e) for e in day_events],
        }

```

### 9.2 优化器失败的多级降级链

当主优化器失败时，须按以下优先级顺序尝试降级：

```
Level 0（正常）：带完整协方差的 QP + 全部软约束
    ↓ 失败（infeasible / timeout）
Level 1（轻度降级）：对角协方差 QP + 放宽换手约束上限至 1.5×
    ↓ 失败
Level 2（中度降级）：无协方差 QP，仅换手惩罚 + 行业约束软化
    ↓ 失败
Level 3（重度降级）：Top-K Buffer 等权（不涉及 QP）
    ↓ 仍无法产出（候选池为空）
Level 4（最终安全网）：维持上期持仓，is_degraded=True，强制人工确认

```

***

## 十、报告模块的完整实现

### 10.1 `reporting/templates/` 落地

须实现 `daily_risk_report.jinja2` 模板，生成包含以下部分的日度风险报告（HTML 格式，便于邮件发送）：

- **执行摘要**：日期、运行状态、是否降级、持仓股票数、换手率（单边）、总仓位
- **Alpha 融合状态**：各域可用性、融合方法、降级域列表
- **持仓概况**：前 10 大持仓、持仓分布直方图（按权重区间）
- **行业暴露**：主动行业暴露柱状图，标注超出约束的行业（红色警示）
- **风格暴露**：雷达图展示各风格因子主动暴露 vs. 约束边界
- **换手分析**：买入列表 Top 10、卖出列表 Top 10、净方向分析
- **约束违约汇总**：软约束松弛量（如有）、硬约束违约（如有）
- **降级事件日志**：当日所有 DegradationEvent 的清单

### 10.2 风险归因增强

在基础的行业/风格主动暴露之外，须实现基于因子协方差模型的**事前风险分解**：

\[\
\text{TE}^2 = w\_a^T \Sigma w\_a = \underbrace{(B^T w\_a)^T F (B^T w\_a)}*{\text{因子风险贡献}} + \underbrace{w\_a^T \Delta w\_a}*{\text{特质风险贡献}}\
]

进一步分解因子风险到各行业和各风格因子的贡献：

\[\
\text{FC}*k = (B\_k^T w\_a)^2 \cdot F*{kk} + 2 \sum\_{l \neq k} (B\_k^T w\_a)(B\_l^T w\_a) F\_{kl}\
]

这一分解使得风险报告从"事后检查约束是否满足"升级为"事前预见风险来源"，是机构级组合管理的基本要求。

***

## 十一、路线图与优先级排序

基于上述改造内容的复杂度和影响范围，建议按以下迭代顺序推进：

**Sprint 1（约 2 周）——工程基础修复**：\
将所有 `__init__.py` 中的业务逻辑迁移到对应的模块文件；补全 `configs/base.yaml` 及 Schema 校验；统一全项目换手率为单边定义；将所有硬编码惩罚系数迁移到配置文件。这些是代码正确性的前提，不触及金融逻辑。

**Sprint 2（约 3 周）——金融逻辑核心修复**：\
实现 `AsymmetricTradabilityMask` 及其在优化器中的集成；实现 `PenaltyCalibrator` 自适应标定；补全边界测试和一致性测试目录。这些直接影响实盘调仓指令的正确性。

**Sprint 3（约 4 周）——协方差模型升级**：\
实现 `FactorCovEstimator`（先实现 `ledoit_wolf` 方法）；在优化器中集成高效二次型计算；补全稳健性测试和压力测试目录。

**Sprint 4（约 3 周）——性能与可观测性**：\
实现 `TwoStageOptimizer` 和 `SolverRouter`；接入结构化日志；实现每日诊断快照落盘；完成 `reporting/templates/` 模板。

**Sprint 5（持续迭代）——增强版功能**：\
动态 IC 加权融合；完整 Barra 风格因子协方差（含 `factor_model` 方法）；跟踪误差约束；多基准支持。

***

## 十二、总结

本文档围绕三条主线完成了对 `portfolio_layer` 从"生产仿真"向"机构级生产可用"的升级规划：

**工程主线**：通过模块文件瘦身、配置系统完整化和诊断可观测性建设，使项目达到可在生产环境中被独立审计、回放和调试的工程标准。

**数学主线**：通过因子协方差矩阵的正确实现（Ledoit-Wolf/OAS 收缩、指数加权、高效二次型计算），真正赋予二次规划优化器感知股票相关性的能力；通过自适应惩罚系数标定，使软约束体系在 Alpha 分布变化时保持稳健的权衡语义。

**金融主线**：通过涨跌停不对称约束的正确建模，消除现有系统向执行层传递"幻象调仓指令"的风险；通过换手率定义的全局统一，消除参数调节时的直觉偏差；通过风险归因的事前分解，将报告模块从被动的合规检查工具升级为主动的风险管理工具。

以上改造完成后，`portfolio_layer` 将具备管理十亿至百亿级资金规模实盘组合的全部核心能力，同时保持高度的模块化和可测试性，能够承接后续动态 IC 加权、多因子 alpha 衰减、在线再平衡触发等更高阶功能的迭代扩展。

***

**版本历史**

版本

日期

变更

v2.0.0

2026-03-25

基于评估报告 2026-03-25，全面改造规范
