# Alpha 信号融合、组合优化与风险约束工程文档

**文档编号**：DOC-PORTFOLIO-002\
**版本**：v1.0.0\
**状态**：正式发布\
**维护方**：量化组合工程团队\
**最后更新**：2025-07

***

## 目录

1. [文档摘要](#1-文档摘要)
2. [目标与边界](#2-目标与边界)
3. [输入输出契约](#3-输入输出契约)
4. [Alpha / Beta / Risk 三元解耦原则](#4-alpha--beta--risk-三元解耦原则)
5. [Alpha 融合框架](#5-alpha-融合框架)
6. [候选池设计](#6-候选池设计)
7. [组合优化器设计](#7-组合优化器设计)
8. [风险模型与约束设计](#8-风险模型与约束设计)
9. [仓位缩放与指数信号子模块](#9-仓位缩放与指数信号子模块)
10. [降级机制与容错设计](#10-降级机制与容错设计)
11. [质量与测试设计](#11-质量与测试设计)
12. [项目目录结构](#12-项目目录结构)
13. [类接口与伪代码说明](#13-类接口与伪代码说明)
14. [MVP / 增强版 / 远期版路线图](#14-mvp--增强版--远期版路线图)
15. [附录：数据表字段参考](#15-附录数据表字段参考)

***

## 1. 文档摘要

本文档是"Alpha 信号融合、组合优化与风险约束"模块的**工程规范文档**，定位于**组合层（Portfolio Layer）**，位于预测层（Alpha Model）与执行层（Order/Execution）之间。

本文档的核心使命是回答以下一个主问题：

> **如何把个股** **`alpha_score`** **转化为可持有、可约束、可解释、可复现的目标组合** **`target_portfolio`？**

本文档不是研究论文，不是投资框架综述，也不讨论 Alpha 模型的训练逻辑、订单撮合、回测仿真或盘中人工干预。它是一份面向工程师和后续 AI 代码生成器的**正式工程规范**，所有设计均需可直接落地为 Python 项目。

***

## 2. 目标与边界

### 2.1 本层职责

| 职责       | 描述                                                          |
| :------- | :---------------------------------------------------------- |
| Alpha 融合 | 接收来自一个或多个 Alpha 模型的 `alpha_score`，合并为复合打分 `composite_alpha` |
| 候选池构建    | 基于可交易性、流动性、行业/市值条件筛选候选股票宇宙                                  |
| 风险暴露计算   | 为候选股票构建行业、风格、市值等多维暴露矩阵                                      |
| 约束建模     | 将业务与风险约束编码为优化器可消费的线性/二次约束                                   |
| 组合优化     | 在给定 Alpha、风险暴露和约束下求解目标权重                                    |
| 仓位后处理    | 对权重进行舍入、取整、最小持仓裁剪                                           |
| 仓位缩放     | 根据指数/市场状态信号调整总仓位比例                                          |
| 输出与报告    | 输出目标权重、再平衡列表、风险暴露报告                                         |

### 2.2 本层**不负责**的内容（硬边界）

- **不**重新训练或调整 Alpha 模型参数
- **不**模拟成交：不计算滑点、不生成 TWAP/VWAP 指令
- **不**执行盘中动态监控或人工干预
- **不**生成交易信号（下单决策属于执行层）
- **不**做回测成交仿真（回测层另行负责）
- **不**反向影响 Alpha 模型训练逻辑：指数预测信号**不得**以任何方式回流进入个股 Alpha 模型的标签定义、训练数据或特征池

### 2.3 层间接口示意

```
┌─────────────────────────────────────────────────────┐
│                    预测层（Alpha Layer）               │
│  alpha_score_A, alpha_score_B, ... alpha_score_N     │
│  index_signal (可选)                                  │
└────────────────────────┬────────────────────────────┘
                         │ AlphaFrame × N
                         ▼
┌─────────────────────────────────────────────────────┐
│              ★ 组合层（Portfolio Layer）★             │
│  本文档负责的全部内容                                  │
└────────────────────────┬────────────────────────────┘
                         │ TargetPortfolio
                         ▼
┌─────────────────────────────────────────────────────┐
│                   执行层（Execution Layer）            │
│  订单路由、成交模拟、实时监控                           │
└─────────────────────────────────────────────────────┘
```

***

## 3. 输入输出契约

### 3.1 输入数据对象

#### 3.1.1 `AlphaFrame`（单模型 Alpha 打分帧）

```python
@dataclass
class AlphaFrame:
    date: str                          # 交易日，格式 YYYYMMDD
    domain: str                        # 域标识，如 "A"/"B"/"C"/"D"/"E"
    model_id: str                      # 模型唯一标识，如 "lgb_v3_5d"
    horizon: int                       # 预测 horizon，单位交易日，如 1/5/10/20
    scores: pd.Series                  # index=ts_code, value=float alpha_score
    score_version: str                 # 模型版本号，如 "20250701"
    available: bool = True             # 该帧是否可用（降级机制使用）
    meta: dict = field(default_factory=dict)  # 附加元信息
```

字段规范：

- `scores` 必须是**截面标准化后**的分数（均值为 0，标准差为 1），或百分位 rank（\[0,1]），两种形式均可，但同一融合流水线内必须统一。
- `available=False` 时，该帧在 `AlphaCombiner` 中被跳过，并触发降级逻辑。

#### 3.1.2 `CompositeAlphaFrame`（复合 Alpha 打分帧）

```python
@dataclass
class CompositeAlphaFrame:
    date: str
    composite_score: pd.Series        # index=ts_code, value=float 复合打分
    source_domains: List[str]         # 参与融合的域列表
    fusion_method: str                # 融合方法标识，如 "weighted_avg_v2"
    domain_weights: Dict[str, float]  # 各域权重
    is_degraded: bool = False         # 是否处于降级运行状态
    degraded_domains: List[str] = field(default_factory=list)  # 缺席域列表
```

#### 3.1.3 `CandidateUniverse`（候选股票宇宙）

```python
@dataclass
class CandidateUniverse:
    date: str
    primary: pd.Index                 # 主候选池，ts_code 集合
    reserve: pd.Index                 # 替补池，ts_code 集合（可为空）
    excluded: pd.Index                # 本日明确排除的股票（涨跌停/停牌/新股等）
    exclusion_reason: Dict[str, str]  # ts_code -> 排除原因
```

#### 3.1.4 `RiskExposureFrame`（风险暴露矩阵）

```python
@dataclass
class RiskExposureFrame:
    date: str
    industry_exposure: pd.DataFrame   # shape=(N_stock, N_industry)，哑变量矩阵
    style_exposure: pd.DataFrame      # shape=(N_stock, N_style)，标准化风格因子
    # style 列包含：size, beta, momentum, volatility, value, liquidity, growth 等
    benchmark_weights: pd.Series      # index=ts_code，基准权重（如全为 0 则为纯多头）
```

#### 3.1.5 `ConstraintSet`（约束集合）

```python
@dataclass
class ConstraintSet:
    # 权重约束
    weight_lb: pd.Series              # 每只股票权重下界（通常为 0 或小正数）
    weight_ub: pd.Series              # 每只股票权重上界（如 0.05）
    total_weight_lb: float = 0.95     # 总权重下界
    total_weight_ub: float = 1.05     # 总权重上界（允许轻微杠杆或现金）

    # 行业约束
    industry_deviation_ub: float = 0.05    # 单行业相对基准偏离上限
    industry_abs_ub: float = 0.30          # 单行业绝对权重上限

    # 风格约束
    style_deviation_ub: Dict[str, float] = field(default_factory=dict)
    # 例：{"size": 0.5, "momentum": 0.3}，单位为标准差

    # 换手约束
    turnover_ub: float = 0.30              # 单次换手率上限（相对当前持仓）

    # 流动性约束
    liquidity_adv_fraction_ub: float = 0.10   # 单票持仓不超过其 ADV 的 X%
    min_liquidity_score: float = 0.0           # 候选股最低流动性评分阈值

    # 集中度约束
    max_single_stock_weight: float = 0.05
    min_stock_count: int = 30
    max_stock_count: int = 150

    # 主动风险预算
    tracking_error_ub: Optional[float] = None     # 跟踪误差上限（年化）
    active_risk_budget: Optional[float] = None    # 主动风险预算
```

#### 3.1.6 市场辅助数据（逐日传入，非核心对象）

以下数据以 `pd.DataFrame` 形式通过各模块构造函数注入或通过数据加载器按需获取：

| 数据源表                            | 用途                     | 关键字段                                                                                                     |
| :------------------------------ | :--------------------- | :------------------------------------------------------------------------------------------------------- |
| `raw_index_member_all`          | 行业归属、指数成分              | `ts_code`, `index_code`, `in_date`, `out_date`                                                           |
| `raw_index_daily_basic_circ_mv` | 流通市值、行业市值权重            | `ts_code`, `trade_date`, `circ_mv`                                                                       |
| `raw_daily_basic`               | 估值、规模、流动性代理            | `total_mv`, `circ_mv`, `free_share`, `turnover_rate`, `volume_ratio`, `pe_ttm`, `pb`, `ps_ttm`, `dv_ttm` |
| `raw_stk_limit`                 | 涨跌停标识                  | `ts_code`, `trade_date`, `up_limit`, `down_limit`                                                        |
| `raw_suspend_d`                 | 停复牌标识                  | `ts_code`, `suspend_date`, `resume_date`                                                                 |
| `feat_feature_D_fundamental`    | 估值 rank、规模 rank、压缩估值分数 | `ts_code`, `trade_date`, `val_rank`, `size_rank`, `val_score_compressed`                                 |

### 3.2 输出数据对象

#### 3.2.1 `TargetPortfolio`（目标组合）

```python
@dataclass
class TargetPortfolio:
    date: str
    target_weight: pd.Series          # index=ts_code，目标权重，sum≈1.0
    target_position: pd.Series        # index=ts_code，目标股数（需传入总资产和股价）
    rebalance_list: pd.DataFrame      # columns: [ts_code, current_weight, target_weight, delta_weight, direction]
    gross_exposure: float             # 总暴露（仓位缩放后）
    cash_ratio: float                 # 现金比例
    optimizer_status: str             # "optimal" / "feasible" / "degraded" / "failed"
    fusion_method: str                # 本次使用的融合方法
    is_degraded: bool = False
    meta: dict = field(default_factory=dict)
```

#### 3.2.2 `PortfolioRiskReport`（组合风险报告）

```python
@dataclass
class PortfolioRiskReport:
    date: str
    # 暴露统计
    industry_exposure_active: pd.Series    # 各行业主动暴露（相对基准偏离）
    style_exposure_active: pd.Series       # 各风格因子主动暴露
    top10_weight: float                    # 前十大持仓权重合计
    stock_count: int                       # 持仓股票数量
    # 风险估计
    estimated_tracking_error: float       # 估计跟踪误差（年化）
    estimated_active_risk: float          # 主动风险估计
    # 换手统计
    turnover_rate: float                  # 本次换手率
    # 违约检查结果
    constraint_violations: List[str]      # 软约束或后验违约列表
    # 集中度
    herfindahl_index: float               # 赫芬达尔指数（越小越分散）
```

***

## 4. Alpha / Beta / Risk 三元解耦原则

本系统的设计基石是严格区分以下三个概念，**绝不允许三者混入同一个黑箱分数**：

### 4.1 Alpha（个股相对收益预测）

**定义**：Alpha 是对个股截面超额收益的预测，与市场整体方向无关。

- 表现形式：`alpha_score`，截面标准化分数或百分位 rank
- 来源：Alpha 模型输出，可来自单域、多域、多 horizon
- 在组合层的用途：作为优化器目标函数的线性收益项 ( \mu^T w )
- **禁止行为**：不允许把市场仓位偏好、行业集中度偏好、换手偏好混入 `alpha_score`

### 4.2 Beta（市场方向 / 总仓位 / 指数风险预算）

**定义**：Beta 是对市场整体方向和总暴露的判断，决定组合的绝对风险暴露水平。

- 表现形式：`gross_exposure_scale`（\[0.0, 1.2]）、`cash_ratio_signal`、`risk_on_off_signal`
- 来源：独立的指数预测子模块或宏观信号（见第 9 节）
- 在组合层的用途：对最终权重做总仓位缩放
- **禁止行为**：不允许把 Beta 信号直接加入 Alpha 模型训练标签，不允许让 Beta 信号影响个股排序逻辑

### 4.3 Risk（约束与风险预算）

**定义**：Risk 是对组合暴露的管控边界，包括行业、风格、市值、流动性、集中度、换手等维度。

- 表现形式：`ConstraintSet`、`RiskExposureFrame`
- 来源：风险模型（Barra-like 因子暴露）+ 业务规则
- 在组合层的用途：约束优化问题的可行域，以及事后归因报告
- **禁止行为**：不允许把风险约束的惩罚项直接折算为 Alpha 分数

### 4.4 三元关系示意

```
Alpha（个股排序）  ─────────────→  目标权重 w*
                                       ↑
Beta（市场仓位）   ─→  gross_scale ───┤  × gross_scale
                                       ↑
Risk（约束边界）   ─→  可行域 Ω  ────┘  s.t. w ∈ Ω
```

**原则性声明**：

> Alpha 高不等于仓位一定高。个股最终仓位由 Alpha、风险约束和 Beta 仓位共同决定，三者独立运作，共同收敛于目标权重。

***

## 5. Alpha 融合框架

### 5.1 融合框架总览

本系统支持四类融合模式，按复杂度递增：

| 融合模式              | 适用场景               | 特点         |
| :---------------- | :----------------- | :--------- |
| 单模型直接使用           | 早期 MVP，只有一个模型上线    | 最简单，直接透传   |
| 多模型加权平均           | 同域多版本或多 horizon 融合 | 版本稳定后可提效   |
| 多域打分融合（A/B/C/D/E） | 多数据域模型并行上线         | 需要处理域缺席的降级 |
| 分层融合（先域内再域间）      | 每个域内多个模型，域间再融合     | 最精细，推荐生产使用 |

### 5.2 模式一：单模型直接使用

**适用场景**：仅有一个 Alpha 模型上线，或其他模型均处于不可用状态时的最终降级模式。

**接口**：

```python
class SingleModelFusion:
    """
    直接透传单个 AlphaFrame 的 scores，无需融合计算。
    """
    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        assert len(frames) == 1 or (len(frames) > 1 and sum(f.available for f in frames) == 1)
        frame = next(f for f in frames if f.available)
        return CompositeAlphaFrame(
            date=frame.date,
            composite_score=frame.scores,
            source_domains=[frame.domain],
            fusion_method="single_model",
            domain_weights={frame.domain: 1.0},
        )
```

**注意**：此模式下组合层必须在 `TargetPortfolio.is_degraded` 中标注降级状态。

### 5.3 模式二：多模型加权平均

**适用场景**：

- 同一域内有多个版本模型需要集成（如 lgb\_v2 + lgb\_v3）
- 多个预测 horizon（如 5d + 10d + 20d）需要加权合成

**融合公式**：

\[
\text{composite\_score}*i = \sum*{m=1}^{M} w\_m \cdot \text{score}*{i,m}, \quad \sum*{m=1}^{M} w\_m = 1
]

**权重设定策略**（按优先级）：

1. **固定权重**：手工指定，适合稳定上线的模型组合
2. **IC 加权**：( w\_m \propto \overline{\text{IC}}\_m )，以近 N 日滚动 IC 均值作为权重
3. **IC\_IR 加权**：( w\_m \propto \overline{\text{IC}}\_m / \sigma(\text{IC}\_m) )，更稳健

**接口**：

```python
class WeightedAverageFusion:
    def __init__(self, weights: Dict[str, float], normalize: bool = True):
        """
        weights: {model_id: weight}
        normalize: 是否对权重重新归一化（处理部分模型缺席时的情况）
        """
        self.weights = weights
        self.normalize = normalize

    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        available_frames = [f for f in frames if f.available]
        if not available_frames:
            raise RuntimeError("所有 AlphaFrame 均不可用，无法融合")

        scores_df = pd.DataFrame({f.model_id: f.scores for f in available_frames})
        w = {mid: self.weights.get(mid, 0.0) for mid in scores_df.columns}

        if self.normalize:
            total = sum(w.values())
            w = {k: v / total for k, v in w.items()}

        composite = sum(scores_df[mid] * wt for mid, wt in w.items())
        # 截面重新标准化
        composite = (composite - composite.mean()) / (composite.std() + 1e-8)

        return CompositeAlphaFrame(
            date=available_frames[0].date,
            composite_score=composite,
            source_domains=list({f.domain for f in available_frames}),
            fusion_method="weighted_average",
            domain_weights=w,
            is_degraded=len(available_frames) < len(frames),
            degraded_domains=[f.domain for f in frames if not f.available],
        )
```

### 5.4 模式三：多域打分融合（A/B/C/D/E 域）

**适用场景**：多个独立数据域分别训练了模型，需要跨域融合。

**域定义示意**（可根据项目实际调整）：

| 域标识 | 数据域描述       | 数据起始时限       |
| :-- | :---------- | :----------- |
| A   | 基础行情与技术类特征  | 最早可用         |
| B   | 基本面与估值类特征   | 较早可用         |
| C   | 资金流与北向资金类特征 | 中等，约 2017 年后 |
| D   | 筹码分布类特征     | 较晚，约 2019 年后 |
| E   | 日内高频聚合类特征   | 最晚，数据量最敏感    |

**域间融合公式**：

\[
\text{composite\_score}*i = \sum*{d \in \mathcal{D}} w\_d \cdot \text{score}\_{i,d}^{(\text{norm})}
]

其中 ( \mathcal{D} ) 为当日可用域集合，( \text{score}\_{i,d}^{(\text{norm})} ) 为各域截面标准化后的分数。

**域缺席降级规则**（详见第 10 节）：

```
若 |可用域| >= 3：正常融合，标注 is_degraded=False
若 |可用域| == 2：降级融合，标注 is_degraded=True，告警但不阻塞
若 |可用域| == 1：最低降级，单域运行，强制告警
若 |可用域| == 0：直接报错，组合层停止运行
```

**接口**：

```python
class MultiDomainFusion:
    def __init__(self, domain_weights: Dict[str, float], min_available_domains: int = 1):
        self.domain_weights = domain_weights   # {domain_id: weight}
        self.min_available_domains = min_available_domains

    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        # 按 domain 聚合（每域只取一个代表性分数，若多模型则先做域内加权平均）
        domain_scores: Dict[str, pd.Series] = {}
        for f in frames:
            if f.available:
                if f.domain in domain_scores:
                    # 同域多模型取均值（简化逻辑，可替换为 IC 加权）
                    domain_scores[f.domain] = (domain_scores[f.domain] + f.scores) / 2
                else:
                    domain_scores[f.domain] = f.scores

        available_domains = list(domain_scores.keys())
        if len(available_domains) < self.min_available_domains:
            raise RuntimeError(f"可用域数 {len(available_domains)} 低于最低要求 {self.min_available_domains}")

        # 权重归一化
        total_w = sum(self.domain_weights.get(d, 0.0) for d in available_domains)
        if total_w <= 0:
            raise ValueError("所有可用域的权重总和为 0，检查配置")

        composite = pd.Series(0.0, index=domain_scores[available_domains[0]].index)
        used_weights = {}
        for d in available_domains:
            w = self.domain_weights.get(d, 0.0) / total_w
            score_norm = domain_scores[d]
            # 截面标准化
            score_norm = (score_norm - score_norm.mean()) / (score_norm.std() + 1e-8)
            composite += w * score_norm
            used_weights[d] = w

        # 整体再标准化
        composite = (composite - composite.mean()) / (composite.std() + 1e-8)

        degraded_domains = [f.domain for f in frames if not f.available]
        return CompositeAlphaFrame(
            date=frames[0].date,
            composite_score=composite,
            source_domains=available_domains,
            fusion_method="multi_domain",
            domain_weights=used_weights,
            is_degraded=len(degraded_domains) > 0,
            degraded_domains=degraded_domains,
        )
```

### 5.5 模式四：分层融合（先域内再域间）

**适用场景**：每个域内有多个模型（不同 horizon、不同版本），需要先在域内做精细融合，再做域间汇总。

**两阶段逻辑**：

```
阶段一（域内融合）：
  对每个域 d：
    score_d = Σ_m w_{d,m} * score_{d,m}   （IC 加权或固定权重）
    score_d_norm = zscore(score_d)

阶段二（域间融合）：
  composite = Σ_d W_d * score_d_norm        （域间权重）
  composite_norm = zscore(composite)
```

**伪代码**：

```python
class HierarchicalFusion:
    def __init__(
        self,
        intra_domain_weights: Dict[str, Dict[str, float]],  # {domain: {model_id: weight}}
        inter_domain_weights: Dict[str, float],              # {domain: weight}
    ):
        self.intra = intra_domain_weights
        self.inter = inter_domain_weights

    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        # 阶段一：域内融合
        domain_scores: Dict[str, pd.Series] = {}
        for domain in set(f.domain for f in frames):
            domain_frames = [f for f in frames if f.domain == domain and f.available]
            if not domain_frames:
                continue
            intra_w = self.intra.get(domain, {})
            total_intra = sum(intra_w.get(f.model_id, 1.0) for f in domain_frames)
            merged = sum(
                (intra_w.get(f.model_id, 1.0) / total_intra) * f.scores
                for f in domain_frames
            )
            domain_scores[domain] = zscore(merged)

        # 阶段二：域间融合（同 MultiDomainFusion）
        available_domains = list(domain_scores.keys())
        total_inter = sum(self.inter.get(d, 0.0) for d in available_domains)
        composite = sum(
            (self.inter.get(d, 0.0) / total_inter) * domain_scores[d]
            for d in available_domains
        )
        composite = zscore(composite)

        degraded_domains = [
            d for d in self.inter if d not in available_domains
        ]
        return CompositeAlphaFrame(
            date=frames[0].date,
            composite_score=composite,
            source_domains=available_domains,
            fusion_method="hierarchical",
            domain_weights={d: self.inter.get(d, 0.0) / total_inter for d in available_domains},
            is_degraded=len(degraded_domains) > 0,
            degraded_domains=degraded_domains,
        )
```

### 5.6 融合后标准化要求

所有融合方法在输出 `composite_score` 前**必须满足**：

1. 截面均值为 0，标准差为 1（z-score 标准化）
2. 处理 `NaN`：对未参与任何域打分的股票，`composite_score` 填充为 `NaN`，在 `CandidateSelector` 中将其排除出主候选池
3. 极端值截断：建议对 ( |z| > 3 ) 的值 winsorize 至 (\pm 3)

***

## 6. 候选池设计

### 6.1 候选池的两层结构

```
全市场股票宇宙（~5000 只）
    │
    ▼ 可交易性过滤
主候选池（Primary Universe，~1000–3000 只）
    │
    ▼ Alpha 排序 + 流动性过滤
核心选股池（Core Selection Pool，Top-K）
    │
    ├── 最终持仓（Final Holdings）
    └── 替补池（Reserve Pool，备选补充）
```

### 6.2 `CandidateSelector` 过滤逻辑

```python
class CandidateSelector:
    """
    负责从全市场宇宙中筛选主候选池，识别替补池和排除集。
    """
    def __init__(self, config: CandidateSelectorConfig):
        self.config = config

    def build(
        self,
        date: str,
        composite_alpha: CompositeAlphaFrame,
        stk_limit_df: pd.DataFrame,      # raw_stk_limit
        suspend_df: pd.DataFrame,         # raw_suspend_d
        daily_basic_df: pd.DataFrame,     # raw_daily_basic
        index_member_df: pd.DataFrame,    # raw_index_member_all
    ) -> CandidateUniverse:
        all_stocks = composite_alpha.composite_score.index

        exclusion_reason: Dict[str, str] = {}

        # 步骤 1：排除停牌
        suspended = self._get_suspended(date, suspend_df)
        for s in suspended:
            exclusion_reason[s] = "suspended"

        # 步骤 2：排除涨停（持有无法卖出）/ 跌停（无法买入）
        limit_up = self._get_limit_up(date, stk_limit_df)
        limit_down = self._get_limit_down(date, stk_limit_df)
        for s in limit_up:
            exclusion_reason.setdefault(s, "limit_up")
        for s in limit_down:
            exclusion_reason.setdefault(s, "limit_down")

        # 步骤 3：排除上市不足 N 天的新股
        new_stocks = self._get_new_stocks(date, daily_basic_df, min_listed_days=self.config.min_listed_days)
        for s in new_stocks:
            exclusion_reason.setdefault(s, "new_listing")

        # 步骤 4：排除极低流动性
        illiquid = self._get_illiquid(date, daily_basic_df, min_turnover=self.config.min_turnover_rate)
        for s in illiquid:
            exclusion_reason.setdefault(s, "illiquid")

        # 步骤 5：排除 ST/*ST 股票（可选）
        if self.config.exclude_st:
            st_stocks = self._get_st_stocks(index_member_df)
            for s in st_stocks:
                exclusion_reason.setdefault(s, "st_stock")

        excluded = pd.Index(exclusion_reason.keys())
        eligible = all_stocks.difference(excluded)

        # 主候选池：有效 alpha 分数的可交易股票
        valid_alpha = composite_alpha.composite_score.loc[eligible].dropna()
        primary = valid_alpha.index

        # 替补池：alpha 分数缺失但可交易（可按市值等加入）
        reserve = eligible.difference(primary)

        return CandidateUniverse(
            date=date,
            primary=primary,
            reserve=reserve,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
        )
```

### 6.3 CandidateSelectorConfig 配置参数

```python
@dataclass
class CandidateSelectorConfig:
    min_listed_days: int = 60          # 上市最少天数
    min_turnover_rate: float = 0.001   # 最低换手率（0.1%）
    exclude_st: bool = True            # 是否排除 ST 股
    exclude_limit_up_for_buy: bool = True   # 新买入时排除涨停
    exclude_limit_down_for_sell: bool = True # 平仓时排除跌停（持仓管理另处理）
    market_cap_filter_pct: float = 0.0      # 排除市值最低 X% 的股票（0.0=不过滤）
```

### 6.4 空候选池的处理

```python
if len(candidate_universe.primary) == 0:
    # 严重边界情况：主候选池为空
    # 步骤 1：尝试使用替补池
    if len(candidate_universe.reserve) > 0:
        logger.warning(f"[{date}] 主候选池为空，切换至替补池")
        candidate_universe.primary = candidate_universe.reserve
    else:
        # 步骤 2：回退至上一个有效交易日的持仓（持仓不动）
        logger.error(f"[{date}] 主候选池和替补池均为空，维持上一期持仓")
        return TargetPortfolio.hold_previous(date=date)
```

***

## 7. 组合优化器设计

### 7.1 优化器目标函数

标准二次规划目标函数：

\[
\max\_{w} ; \mu^T w - \frac{\lambda}{2} w^T \Sigma w - \gamma \cdot \text{TC}(w, w\_{\text{prev}})
]

其中：

- ( \mu \in \mathbb{R}^N )：归一化的 composite alpha 分数（作为预期收益代理）
- ( w \in \mathbb{R}^N )：目标权重向量
- ( \lambda \geq 0 )：风险厌恶系数，控制 alpha 与风险的权衡
- ( \Sigma \in \mathbb{R}^{N \times N} )：协方差矩阵（因子模型近似）
- ( \gamma \geq 0 )：换手成本惩罚系数
- ( \text{TC}(w, w\_{\text{prev}}) = \sum\_i c\_i |w\_i - w\_{\text{prev},i}| )：换手成本（L1 惩罚）

**简化版目标函数**（MVP 阶段，不使用协方差矩阵）：

\[
\max\_{w} ; \mu^T w - \gamma \cdot |w - w\_{\text{prev}}|\_1
]

### 7.2 标准约束集合

\[
\begin{aligned}
& \sum\_i w\_i = 1 \quad \text{（满仓约束）} \\
& w\_i \geq 0 \quad \forall i \quad \text{（纯多头）} \\
& w\_i \leq w\_{\text{ub},i} \quad \forall i \quad \text{（单票上限）} \\
& \sum\_{i \in \text{ind}*k} w\_i \leq B\_k + \delta\_k^{\text{ub}} \quad \forall k \quad \text{（行业偏离上限）} \\
& \sum*{i \in \text{ind}*k} w\_i \geq B\_k - \delta\_k^{\text{lb}} \quad \forall k \quad \text{（行业偏离下限）} \\
& |f\_j^T w - f\_j^T w*{\text{bm}}| \leq \epsilon\_j \quad \forall j \quad \text{（风格因子偏离约束）} \\
& |w - w\_{\text{prev}}|\_1 \leq \tau \quad \text{（换手率约束）}
\end{aligned}
]

### 7.3 五种 Top-K 组合构造模式

#### 7.3.1 Top-K 等权（Equal Weight）

```python
def build_topk_equal_weight(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    k: int,
) -> pd.Series:
    """最简基线，前 K 只股票等权持有。"""
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    topk = scores.nlargest(k).index
    weight = pd.Series(1.0 / k, index=topk)
    return weight
```

**适用场景**：基线评估，验证 Alpha 有效性。

#### 7.3.2 Top-K 分数加权（Score Weighted）

```python
def build_topk_score_weighted(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    k: int,
    softmax_temp: float = 1.0,
) -> pd.Series:
    """
    按 softmax 变换后的 alpha 分数分配权重。
    softmax_temp 越大越趋近等权，越小越集中于高分股。
    """
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    topk_scores = scores.nlargest(k)
    # softmax 权重
    exp_s = np.exp(topk_scores / softmax_temp)
    weight = exp_s / exp_s.sum()
    return weight
```

#### 7.3.3 Top-K + Buffer（缓冲区机制）

**目的**：减少因股票排名小幅波动导致的无效换手。

```python
def build_topk_with_buffer(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    prev_holdings: pd.Index,
    k: int,
    buffer_ratio: float = 0.2,
) -> pd.Index:
    """
    缓冲区逻辑：
    - hard_k = k * (1 - buffer_ratio)：前 hard_k 只强制买入
    - buffer_k = k * buffer_ratio：从 [hard_k+1, k+buffer_k] 范围中，
      优先保留上期已持有的股票，不足时补充新高分股
    """
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    sorted_scores = scores.sort_values(ascending=False)

    hard_k = int(k * (1 - buffer_ratio))
    buffer_k = k - hard_k
    extra_pool_size = int(k * buffer_ratio * 2)  # 额外备选区间

    hard_picks = sorted_scores.iloc[:hard_k].index
    buffer_zone = sorted_scores.iloc[hard_k: hard_k + extra_pool_size].index

    # 从 buffer_zone 中优先保留上期持仓
    retained = prev_holdings.intersection(buffer_zone)
    new_picks = buffer_zone.difference(prev_holdings)

    buffer_picks = retained.append(new_picks)[:buffer_k]
    return hard_picks.append(buffer_picks)
```

#### 7.3.4 Top-K + 行业/风格/单票约束（规则约束）

```python
def build_topk_with_constraints(
    composite_alpha: CompositeAlphaFrame,
    candidate_universe: CandidateUniverse,
    constraint_set: ConstraintSet,
    industry_map: pd.Series,          # ts_code -> industry_code
    benchmark_industry_weights: pd.Series,  # industry_code -> benchmark_weight
    k: int,
) -> pd.Series:
    """
    贪心策略：
    1. 按 alpha 分数从高到低遍历候选股
    2. 每次加入一只股票前检查：
       - 行业权重是否超过基准 + 行业偏离上限
       - 单票权重是否超过上限
    3. 达到 k 只后停止
    4. 对选出的股票做分数加权 + 权重裁剪
    """
    scores = composite_alpha.composite_score.loc[candidate_universe.primary].dropna()
    sorted_scores = scores.sort_values(ascending=False)

    selected = []
    industry_current_weight: Dict[str, float] = defaultdict(float)
    per_stock_weight = 1.0 / k  # 估算单票权重

    for stock in sorted_scores.index:
        if len(selected) >= k:
            break
        ind = industry_map.get(stock, "unknown")
        bm_ind_w = benchmark_industry_weights.get(ind, 0.0)
        new_ind_w = industry_current_weight[ind] + per_stock_weight
        if new_ind_w > bm_ind_w + constraint_set.industry_deviation_ub:
            continue  # 跳过，行业超限
        selected.append(stock)
        industry_current_weight[ind] += per_stock_weight

    if not selected:
        raise RuntimeError("规则约束过于严格，无法选出任何股票")

    # 分数加权
    selected_scores = sorted_scores.loc[selected]
    exp_s = np.exp(selected_scores - selected_scores.max())
    weight = exp_s / exp_s.sum()
    weight = weight.clip(upper=constraint_set.max_single_stock_weight)
    weight = weight / weight.sum()
    return weight
```

#### 7.3.5 Top-K + 二次优化器（Risk-Adjusted Objective）

见第 7.4 节。

### 7.4 `PortfolioOptimizer` 设计

```python
class PortfolioOptimizer:
    """
    基于 cvxpy 的二次规划组合优化器。
    支持完整约束集合。
    """
    def __init__(self, config: OptimizerConfig):
        self.config = config

    def optimize(
        self,
        composite_alpha: CompositeAlphaFrame,
        risk_exposure: RiskExposureFrame,
        constraint_set: ConstraintSet,
        prev_weights: pd.Series,         # 上期持仓权重，index=ts_code
        cov_matrix: Optional[pd.DataFrame] = None,  # 因子协方差近似，可为 None
    ) -> Tuple[pd.Series, str]:
        """
        返回：(target_weight, solver_status)
        """
        import cvxpy as cp

        stocks = composite_alpha.composite_score.dropna().index
        n = len(stocks)
        mu = composite_alpha.composite_score.loc[stocks].values  # alpha 向量

        # 决策变量
        w = cp.Variable(n, nonneg=True)

        # 目标函数
        objective_terms = [mu @ w]

        # 风险惩罚项（若提供协方差矩阵）
        if cov_matrix is not None:
            Sigma = cov_matrix.loc[stocks, stocks].values
            objective_terms.append(-self.config.risk_aversion * cp.quad_form(w, Sigma))

        # 换手成本惩罚
        w_prev = prev_weights.reindex(stocks, fill_value=0.0).values
        objective_terms.append(-self.config.turnover_penalty * cp.norm1(w - w_prev))

        objective = cp.Maximize(sum(objective_terms))

        # 约束
        constraints = []

        # 总权重约束
        constraints.append(cp.sum(w) >= constraint_set.total_weight_lb)
        constraints.append(cp.sum(w) <= constraint_set.total_weight_ub)

        # 单票权重上限
        w_ub = constraint_set.weight_ub.reindex(stocks, fill_value=constraint_set.max_single_stock_weight).values
        w_lb = constraint_set.weight_lb.reindex(stocks, fill_value=0.0).values
        constraints.append(w <= w_ub)
        constraints.append(w >= w_lb)

        # 行业约束
        industry_dummies = risk_exposure.industry_exposure.loc[stocks]
        bm_industry_w = (risk_exposure.benchmark_weights.reindex(stocks, fill_value=0.0).values
                         @ industry_dummies.values)
        for k, ind in enumerate(industry_dummies.columns):
            ind_vec = industry_dummies[ind].values
            bm_w = bm_industry_w[k] if hasattr(bm_industry_w, '__len__') else 0.0
            constraints.append(ind_vec @ w <= bm_w + constraint_set.industry_deviation_ub)
            constraints.append(ind_vec @ w >= max(0, bm_w - constraint_set.industry_deviation_ub))

        # 风格偏离约束
        if constraint_set.style_deviation_ub:
            style_mat = risk_exposure.style_exposure.loc[stocks]
            bm_style = (risk_exposure.benchmark_weights.reindex(stocks, fill_value=0.0).values
                        @ style_mat.values)
            for j, style in enumerate(style_mat.columns):
                if style in constraint_set.style_deviation_ub:
                    eps = constraint_set.style_deviation_ub[style]
                    f_vec = style_mat[style].values
                    bm_f = bm_style[j] if hasattr(bm_style, '__len__') else 0.0
                    constraints.append(f_vec @ w <= bm_f + eps)
                    constraints.append(f_vec @ w >= bm_f - eps)

        # 换手约束
        constraints.append(cp.norm1(w - w_prev) <= constraint_set.turnover_ub * 2)

        # 求解
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception as e:
            logger.error(f"优化器求解失败：{e}，尝试降级求解")
            prob.solve(solver=cp.SCS)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            weight_series = pd.Series(w.value, index=stocks)
            weight_series = weight_series.clip(lower=0).pipe(lambda s: s / s.sum())
            return weight_series, prob.status
        else:
            # 优化失败，降级为等权 Top-K
            logger.warning(f"优化器状态 {prob.status}，降级为等权 Top-K")
            k = self.config.fallback_topk
            topk = composite_alpha.composite_score.loc[stocks].nlargest(k).index
            weight_series = pd.Series(1.0 / k, index=topk)
            return weight_series, "degraded"
```

### 7.5 `OptimizerConfig` 参数

```python
@dataclass
class OptimizerConfig:
    risk_aversion: float = 1.0           # 风险厌恶系数 λ
    turnover_penalty: float = 0.5        # 换手惩罚系数 γ
    solver: str = "CLARABEL"             # 主求解器
    fallback_solver: str = "SCS"         # 备用求解器
    fallback_topk: int = 50              # 优化失败时等权 Top-K 的 K 值
    max_solve_time: float = 60.0         # 最大求解时间（秒）
    warm_start: bool = True              # 是否使用热启动
```

### 7.6 协方差矩阵近似（Barra-like 因子模型）

当股票数量超过 300 时，直接使用样本协方差矩阵在数值上不稳定，推荐使用因子模型分解：

\[
\Sigma = B F B^T + D
]

其中：

- ( B \in \mathbb{R}^{N \times K} )：因子暴露矩阵（行业 + 风格）
- ( F \in \mathbb{R}^{K \times K} )：因子协方差矩阵（从历史收益回归估计）
- ( D = \text{diag}(\sigma\_1^2, \ldots, \sigma\_N^2) )：特质风险对角矩阵

**MVP 阶段近似**：若不具备完整 Barra 因子协方差估计能力，可使用以下简化方案：

- 只保留行业哑变量作为因子，忽略风格因子
- 或直接设 ( \lambda = 0 )，不使用协方差矩阵，仅依赖约束控制风险暴露

***

## 8. 风险模型与约束设计

### 8.1 `RiskExposureBuilder` 设计

```python
class RiskExposureBuilder:
    """
    构建风险暴露矩阵，包括行业哑变量和风格因子。
    """
    STYLE_FACTORS = [
        "size",        # log(circ_mv)
        "value",       # 1/pb 或 val_rank
        "momentum",    # 过去 20 日涨跌幅
        "volatility",  # 过去 20 日收益率标准差
        "liquidity",   # 换手率
        "beta",        # 市场 beta（60 日滚动）
        "growth",      # 可从 fundamental 特征获取
    ]

    def build(
        self,
        date: str,
        candidate_universe: CandidateUniverse,
        index_member_df: pd.DataFrame,
        daily_basic_df: pd.DataFrame,
        price_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,  # feat_feature_D_fundamental
        benchmark_weights: Optional[pd.Series] = None,
    ) -> RiskExposureFrame:

        stocks = candidate_universe.primary

        # 行业暴露：哑变量矩阵
        industry_map = self._get_industry_map(date, index_member_df)
        industry_exposure = pd.get_dummies(
            industry_map.reindex(stocks, fill_value="unknown"),
            prefix="ind"
        ).astype(float)

        # 风格暴露：逐因子计算并截面标准化
        style_exposure = pd.DataFrame(index=stocks)
        style_exposure["size"] = self._calc_size(date, stocks, daily_basic_df)
        style_exposure["value"] = self._calc_value(date, stocks, daily_basic_df, fundamental_df)
        style_exposure["momentum"] = self._calc_momentum(date, stocks, price_df, window=20)
        style_exposure["volatility"] = self._calc_volatility(date, stocks, price_df, window=20)
        style_exposure["liquidity"] = self._calc_liquidity(date, stocks, daily_basic_df)
        style_exposure["beta"] = self._calc_beta(date, stocks, price_df, window=60)

        # 截面标准化
        style_exposure = style_exposure.apply(
            lambda col: (col - col.mean()) / (col.std() + 1e-8)
        )

        bm_weights = benchmark_weights if benchmark_weights is not None else pd.Series(0.0, index=stocks)

        return RiskExposureFrame(
            date=date,
            industry_exposure=industry_exposure,
            style_exposure=style_exposure,
            benchmark_weights=bm_weights.reindex(stocks, fill_value=0.0),
        )

    def _calc_size(self, date, stocks, daily_basic_df) -> pd.Series:
        """使用流通市值对数作为 size 因子"""
        df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
        return np.log(df["circ_mv"].reindex(stocks) + 1)

    def _calc_value(self, date, stocks, daily_basic_df, fundamental_df) -> pd.Series:
        """优先使用 fundamental_df 的 val_rank，fallback 到 1/pb"""
        try:
            fund = fundamental_df[fundamental_df["trade_date"] == date].set_index("ts_code")
            return fund["val_rank"].reindex(stocks)
        except KeyError:
            df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
            pb = df["pb"].reindex(stocks)
            return 1.0 / pb.replace(0, np.nan)

    # ... 其他因子计算方法类似，此处省略
```

### 8.2 Barra-like 风格因子完整列表

| 因子名             | 数据来源                                                         | 计算方式                         | 备注     |
| :-------------- | :----------------------------------------------------------- | :--------------------------- | :----- |
| `size`          | `raw_daily_basic.circ_mv`                                    | ( \ln(\text{circ\_mv}) )     | 流通市值对数 |
| `value`         | `feat_feature_D_fundamental.val_rank` 或 `raw_daily_basic.pb` | `val_rank` 或 ( 1/\text{pb} ) | 估值因子   |
| `momentum`      | 日线复权收益                                                       | 过去 20 日累计对数收益                | 中短期动量  |
| `long_momentum` | 日线复权收益                                                       | 过去 120 日（去除近 20 日）收益         | 长期动量   |
| `volatility`    | 日线收益率                                                        | 20 日标准差 × (\sqrt{252})       | 特质波动率  |
| `liquidity`     | `raw_daily_basic.turnover_rate`                              | 20 日均换手率                     | 流动性    |
| `beta`          | 日线收益率 vs 指数                                                  | 60 日滚动 OLS beta              | 系统性风险  |
| `growth`        | `feat_feature_D_fundamental`                                 | 营收增速 rank 等                  | 可选     |
| `leverage`      | `raw_daily_basic` 或 financial                                | 资产负债率代理                      | 可选     |

### 8.3 `ConstraintBuilder` 设计

```python
class ConstraintBuilder:
    """
    根据业务配置和当日市场数据动态构建约束集合。
    """
    def __init__(self, config: ConstraintBuilderConfig):
        self.config = config

    def build(
        self,
        date: str,
        candidate_universe: CandidateUniverse,
        risk_exposure: RiskExposureFrame,
        daily_basic_df: pd.DataFrame,
        prev_weights: pd.Series,
    ) -> ConstraintSet:
        stocks = candidate_universe.primary

        # 单票权重上限（基于流动性动态调整）
        weight_ub = self._build_weight_ub(date, stocks, daily_basic_df)

        # 流动性约束（ADV fraction）
        # 若无 ADV 数据，用 turnover_rate * circ_mv 估算
        adv_series = self._estimate_adv(date, stocks, daily_basic_df)

        return ConstraintSet(
            weight_lb=pd.Series(0.0, index=stocks),
            weight_ub=weight_ub,
            total_weight_lb=self.config.total_weight_lb,
            total_weight_ub=self.config.total_weight_ub,
            industry_deviation_ub=self.config.industry_deviation_ub,
            industry_abs_ub=self.config.industry_abs_ub,
            style_deviation_ub=self.config.style_deviation_ub,
            turnover_ub=self.config.turnover_ub,
            liquidity_adv_fraction_ub=self.config.liquidity_adv_fraction_ub,
            max_single_stock_weight=self.config.max_single_stock_weight,
            min_stock_count=self.config.min_stock_count,
            max_stock_count=self.config.max_stock_count,
            tracking_error_ub=self.config.tracking_error_ub,
        )

    def _build_weight_ub(self, date, stocks, daily_basic_df) -> pd.Series:
        """
        动态单票上限：
        - 基础上限：config.max_single_stock_weight
        - 若为小市值股：额外收紧至 0.5 × 基础上限
        """
        df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
        circ_mv = df["circ_mv"].reindex(stocks, fill_value=np.nan)
        base_ub = self.config.max_single_stock_weight
        ub = pd.Series(base_ub, index=stocks)
        # 市值低于 20 亿（20e4 万元）的股票收紧权重上限
        small_cap_mask = circ_mv < self.config.small_cap_threshold
        ub[small_cap_mask] = base_ub * 0.5
        return ub

    def _estimate_adv(self, date, stocks, daily_basic_df) -> pd.Series:
        """估算日均成交额（Average Daily Value traded）"""
        df = daily_basic_df[daily_basic_df["trade_date"] == date].set_index("ts_code")
        return (df["turnover_rate"].reindex(stocks) * df["circ_mv"].reindex(stocks) / 100)
```

### 8.4 主动风险预算与跟踪误差估计

**跟踪误差近似**（因子模型）：

\[
\text{TE}^2 \approx (w - w\_{\text{bm}})^T \Sigma (w - w\_{\text{bm}}) = (w\_a)^T (B F B^T + D) (w\_a)
]

其中 ( w\_a = w - w\_{\text{bm}} ) 为主动权重向量。

**跟踪误差约束的 cvxpy 实现**：

```python
if constraint_set.tracking_error_ub is not None and cov_matrix is not None:
    w_active = w - bm_weights
    Sigma_np = cov_matrix.loc[stocks, stocks].values
    # TE² ≤ TE_ub² / 252（年化转日化）
    te_daily_sq = (constraint_set.tracking_error_ub ** 2) / 252
    constraints.append(cp.quad_form(w_active, Sigma_np) <= te_daily_sq)
```

### 8.5 约束优先级与软约束处理

当约束集合导致不可行域时，按以下优先级松弛约束：

```
优先级 1（绝对硬约束，不可松弛）：
  - w_i >= 0（纯多头）
  - 停牌/涨跌停股票 w_i = 0

优先级 2（强约束，轻微松弛）：
  - 总权重约束 (±0.02 松弛)
  - 单票权重上限（放大 1.5 倍）

优先级 3（软约束，可较大幅度松弛）：
  - 行业偏离约束（放大 2 倍）
  - 风格偏离约束（放大 2 倍）
  - 跟踪误差约束（放大 1.5 倍）

优先级 4（最后松弛）：
  - 换手率约束
```

软约束松弛实现建议：在目标函数中添加惩罚项代替硬约束（penalty relaxation），如：

\[
\text{objective} \mathrel{-}= \rho\_k \cdot \max(0, f\_k^T w - \epsilon\_k)^2
]

***

## 9. 仓位缩放与指数信号子模块

### 9.1 模块定位与边界

**核心原则**：指数预测模型的输出**只能**作用于组合层的总仓位缩放，**绝对禁止**：

1. 回流进入个股 Alpha 模型的特征工程或训练标签
2. 影响候选股票的 alpha\_score 排序
3. 直接修改 `CompositeAlphaFrame` 中的个股分数

```
指数预测模块（独立服务） ──→ MarketStateSignal
                                    │
                                    ▼
                          PositionScaler（组合层内部子模块）
                                    │
                                    ▼
               target_weight（个股相对权重）× gross_scale
                                    │
                                    ▼
                         最终持仓（含现金比例）
```

### 9.2 `MarketStateSignal` 数据对象

```python
@dataclass
class MarketStateSignal:
    date: str
    gross_exposure_scale: float     # 总仓位比例，[0.0, 1.2]，1.0 = 满仓
    cash_ratio_signal: float        # 建议现金比例，[0.0, 1.0]
    risk_on_off_signal: str         # "risk_on" / "neutral" / "risk_off"
    signal_source: str              # 信号来源标识，如 "index_model_v2"
    confidence: float = 1.0         # 信号置信度，[0, 1]
    available: bool = True          # 信号是否可用（不可用时默认满仓）
```

### 9.3 `PositionScaler` 设计

```python
class PositionScaler:
    """
    根据市场状态信号对组合权重进行总仓位缩放。
    仅调整 gross_exposure，不改变股票间的相对权重关系。
    """
    def __init__(self, config: PositionScalerConfig):
        self.config = config

    def scale(
        self,
        target_weight: pd.Series,
        market_signal: Optional[MarketStateSignal],
        prev_gross_exposure: float = 1.0,
    ) -> Tuple[pd.Series, float, float]:
        """
        返回：(scaled_weight, gross_exposure, cash_ratio)
        """
        if market_signal is None or not market_signal.available:
            # 信号不可用：默认满仓
            return target_weight, 1.0, 0.0

        scale = market_signal.gross_exposure_scale

        # 平滑缩放：避免仓位突变（EMA 平滑）
        smoothed_scale = (
            self.config.scale_smoothing * prev_gross_exposure
            + (1 - self.config.scale_smoothing) * scale
        )

        # 应用仓位上下限
        smoothed_scale = np.clip(
            smoothed_scale,
            self.config.min_gross_exposure,
            self.config.max_gross_exposure
        )

        # 缩放后的权重（相对权重不变）
        scaled_weight = target_weight * smoothed_scale
        cash_ratio = 1.0 - scaled_weight.sum()

        return scaled_weight, smoothed_scale, cash_ratio

@dataclass
class PositionScalerConfig:
    min_gross_exposure: float = 0.60   # 最低仓位（60%）
    max_gross_exposure: float = 1.00   # 最高仓位（100%，不加杠杆）
    scale_smoothing: float = 0.30      # EMA 平滑系数（前期权重）
    default_scale: float = 1.0         # 信号缺失时默认仓位
```

### 9.4 指数信号与个股 Alpha 的关系保障机制

在代码层面，必须通过以下手段**确保信号隔离**：

1. `MarketStateSignal` 的产生路径与 `AlphaFrame` 的产生路径**物理隔离**（不同模块、不同配置文件）
2. `AlphaCombiner` 的接口签名不接受 `MarketStateSignal` 类型参数
3. `MarketStateSignal` 对 `target_weight` 的作用**仅发生在** `PositionScaler.scale()` 调用时，该调用在优化器之后

***

## 10. 降级机制与容错设计

### 10.1 域缺席降级矩阵

| 可用域数    | 运行状态 | 行为             | 告警级别                  |
| :------ | :--- | :------------- | :-------------------- |
| 5（全部可用） | 正常   | 分层融合           | INFO                  |
| 4       | 轻度降级 | 多域融合（4域）       | WARNING               |
| 3       | 中度降级 | 多域融合（3域）       | WARNING               |
| 2       | 重度降级 | 双域融合           | ERROR（但继续运行）          |
| 1       | 最低降级 | 单域直接使用         | CRITICAL（继续运行，强制人工确认） |
| 0       | 停止运行 | 维持上期持仓，输出空目标组合 | FATAL                 |

### 10.2 `DegradationManager` 设计

```python
class DegradationManager:
    """
    统一管理组合层各子模块的降级逻辑。
    """
    def __init__(self, config: DegradationConfig):
        self.config = config

    def select_fusion_method(self, frames: List[AlphaFrame]) -> AlphaCombiner:
        """根据当前可用 frames 选择合适的融合方法"""
        available_domains = [f.domain for f in frames if f.available]
        n = len(available_domains)

        if n == 0:
            raise FatalDegradationError("无可用 Alpha 域，系统无法运行")
        elif n == 1:
            logger.critical(f"仅有 1 个可用域: {available_domains}，切换为 SingleModelFusion")
            return SingleModelFusion()
        elif n < self.config.min_domains_for_hierarchical:
            logger.error(f"可用域数 {n} 不足，切换为 MultiDomainFusion")
            return MultiDomainFusion(
                domain_weights=self.config.fallback_domain_weights,
                min_available_domains=1,
            )
        else:
            return HierarchicalFusion(
                intra_domain_weights=self.config.intra_domain_weights,
                inter_domain_weights=self.config.inter_domain_weights,
            )

    def handle_empty_candidate_pool(
        self,
        date: str,
        candidate_universe: CandidateUniverse,
        prev_portfolio: Optional[TargetPortfolio],
    ) -> Optional[TargetPortfolio]:
        """处理候选池为空的边界情况"""
        if len(candidate_universe.primary) > 0:
            return None  # 正常，不需要降级处理

        if len(candidate_universe.reserve) > 0:
            logger.warning(f"[{date}] 主候选池为空，扩展至替补池")
            candidate_universe.primary = candidate_universe.reserve
            return None

        if prev_portfolio is not None:
            logger.error(f"[{date}] 候选池完全为空，维持上期持仓")
            return TargetPortfolio(
                date=date,
                target_weight=prev_portfolio.target_weight,
                target_position=prev_portfolio.target_position,
                rebalance_list=pd.DataFrame(),
                gross_exposure=prev_portfolio.gross_exposure,
                cash_ratio=prev_portfolio.cash_ratio,
                optimizer_status="hold_previous",
                fusion_method="none",
                is_degraded=True,
            )

        raise FatalDegradationError(f"[{date}] 候选池为空且无历史持仓，无法生成目标组合")
```

### 10.3 数据缺失降级规则

| 缺失数据                            | 影响模块                                       | 降级行为                                 |
| :------------------------------ | :----------------------------------------- | :----------------------------------- |
| `raw_stk_limit` 缺失              | `CandidateSelector`                        | 跳过涨跌停过滤，记录 WARNING                   |
| `raw_suspend_d` 缺失              | `CandidateSelector`                        | 跳过停牌过滤，记录 WARNING                    |
| `raw_daily_basic` 缺失            | `RiskExposureBuilder`, `ConstraintBuilder` | 使用默认值或上期数据，style 因子设为 NaN，优化器跳过协方差项  |
| 行业映射缺失（某只股票）                    | `RiskExposureBuilder`                      | 归入"未知"行业，不施加行业约束                     |
| 基准权重缺失                          | `ConstraintBuilder`                        | 基准权重全设为 0，退化为纯多头绝对约束                 |
| `feat_feature_D_fundamental` 缺失 | `RiskExposureBuilder`                      | 降级使用 `raw_daily_basic` 中的 `pb` 等原始字段 |

***

## 11. 质量与测试设计

### 11.1 测试分层策略

```
tests/
├── unit/               # 单元测试：函数级别，隔离依赖
├── integration/        # 集成测试：alpha -> target_weight 全链路
├── boundary/           # 边界测试：极端输入与边界条件
├── robustness/         # 稳健性测试：参数扰动测试
├── stress/             # 压力测试：极端市场情景
└── consistency/        # 一致性测试：确定性与可复现性
```

### 11.2 单元测试规范

#### 11.2.1 融合权重测试

```python
# tests/unit/test_alpha_combiner.py

class TestWeightedAverageFusion:
    def test_weights_sum_to_one(self):
        """融合后权重必须归一化为 1.0"""
        ...

    def test_degraded_reweight_correctly(self):
        """一个模型不可用时，剩余权重正确归一化"""
        frames = [
            AlphaFrame(domain="A", available=True, scores=mock_scores(50)),
            AlphaFrame(domain="B", available=False, scores=None),
        ]
        result = WeightedAverageFusion({"A": 0.6, "B": 0.4}).fuse(frames)
        assert abs(result.domain_weights["A"] - 1.0) < 1e-6
        assert result.is_degraded is True

    def test_output_is_cross_sectionally_standardized(self):
        """输出 composite_score 必须满足截面均值≈0，标准差≈1"""
        ...

    def test_no_nan_in_output_for_available_stocks(self):
        """所有参与融合且 score 非 NaN 的股票，输出不为 NaN"""
        ...
```

#### 11.2.2 约束构建测试

```python
class TestConstraintBuilder:
    def test_weight_ub_respects_max_config(self):
        """所有股票权重上限不超过配置的 max_single_stock_weight"""
        ...

    def test_small_cap_weight_ub_halved(self):
        """小市值股票的权重上限为标准上限的 50%"""
        ...

    def test_constraint_set_serializable(self):
        """ConstraintSet 必须可序列化（JSON/pickle）用于复现"""
        ...
```

#### 11.2.3 优化器输入合法性测试

```python
class TestOptimizerInputValidation:
    def test_rejects_nan_alpha(self):
        """alpha 向量含 NaN 时抛出 ValueError"""
        ...

    def test_rejects_non_psd_covariance(self):
        """非半正定协方差矩阵时给出警告并使用对角近似"""
        ...

    def test_accepts_empty_prev_weights(self):
        """上期持仓为空时（首次建仓），优化器正常运行"""
        ...
```

### 11.3 集成测试规范

```python
class TestAlphaToPortfolioPipeline:
    """端到端集成测试：从 alpha_score 到 target_weight"""

    @pytest.fixture
    def mock_pipeline(self):
        """构造一个完整的组合层 pipeline 实例"""
        ...

    def test_full_pipeline_produces_valid_portfolio(self, mock_pipeline):
        """完整 pipeline 输出满足：权重之和≈1，单票≤上限，行业偏离在限内"""
        result = mock_pipeline.run(date="20240701")
        assert abs(result.target_weight.sum() - 1.0) < 0.02
        assert (result.target_weight <= config.max_single_stock_weight + 1e-6).all()

    def test_pipeline_consistent_across_runs(self, mock_pipeline):
        """相同输入，两次运行结果完全一致（确定性）"""
        r1 = mock_pipeline.run(date="20240701")
        r2 = mock_pipeline.run(date="20240701")
        pd.testing.assert_series_equal(r1.target_weight, r2.target_weight)

    def test_pipeline_handles_domain_b_unavailable(self, mock_pipeline):
        """B 域不可用时，pipeline 降级运行并标注 is_degraded=True"""
        ...
```

### 11.4 边界测试规范

```python
class TestBoundaryConditions:

    def test_empty_candidate_pool(self):
        """候选池为空时，返回上期持仓而非崩溃"""
        ...

    def test_all_stocks_limit_up(self):
        """所有股票均涨停时，候选池为空，触发降级"""
        ...

    def test_missing_industry_for_all_stocks(self):
        """所有股票行业归属缺失，优化器跳过行业约束，正常运行"""
        ...

    def test_missing_market_cap_data(self):
        """市值数据完全缺失，风格因子中 size 因子设为均值 0"""
        ...

    def test_single_stock_in_candidate_pool(self):
        """候选池只有 1 只股票，输出该股票权重 = 1.0"""
        ...

    def test_surge_in_untradeable_stocks(self):
        """不可交易股票激增（超过候选池 50%），系统告警但不崩溃"""
        ...
```

### 11.5 稳健性测试规范

```python
class TestRobustness:

    @pytest.mark.parametrize("k", [20, 50, 100, 200])
    def test_different_topk_values(self, k):
        """不同 K 值下，输出组合均有效"""
        ...

    @pytest.mark.parametrize("industry_dev_ub", [0.02, 0.05, 0.10, 0.20])
    def test_different_industry_constraint_strengths(self, industry_dev_ub):
        """不同行业约束强度下，组合均可行且行业偏离不超过约束"""
        ...

    @pytest.mark.parametrize("risk_aversion", [0.0, 0.5, 1.0, 5.0, 10.0])
    def test_risk_aversion_parameter_range(self, risk_aversion):
        """风险厌恶系数从 0 到 10，优化器均收敛"""
        ...

    def test_score_concentration(self):
        """alpha 分数极度集中在少数股票时（top1 分数 >> 其他），权重约束仍有效"""
        ...
```

### 11.6 压力测试规范

```python
class TestStressScenarios:

    def test_entire_industry_limit_up(self):
        """某一行业所有股票全部涨停（卖出受限），系统能正确处理约束并输出有效权重"""
        scenario = StressScenario(
            date="20240701",
            limit_up_industry="801010",  # 农林牧渔
        )
        result = run_pipeline_with_scenario(scenario)
        assert result.optimizer_status in ["optimal", "feasible", "degraded"]

    def test_extreme_low_liquidity(self):
        """市场极低流动性（全市场换手率 < 0.01%），流动性约束自动放宽"""
        ...

    def test_alpha_score_all_equal(self):
        """所有股票 alpha 分数完全相同，优化器输出分散化权重"""
        ...

    def test_large_universe_performance(self):
        """候选池 3000 只股票，优化器求解时间 < 60 秒"""
        import time
        start = time.time()
        result = run_pipeline(n_stocks=3000)
        elapsed = time.time() - start
        assert elapsed < 60.0
```

### 11.7 一致性与可复现性测试

```python
class TestConsistency:

    def test_deterministic_given_same_snapshot(self):
        """相同日期快照，任意时间运行，目标组合结果完全一致"""
        snapshot = load_snapshot("20240701")
        r1 = run_pipeline(snapshot)
        r2 = run_pipeline(snapshot)
        assert (r1.target_weight - r2.target_weight).abs().max() < 1e-8

    def test_no_look_ahead_in_candidate_selection(self):
        """候选池构建只使用 date 当日及之前的数据，不使用未来数据"""
        # 使用数据血缘追踪工具验证
        ...

    def test_rebalance_list_consistent_with_weights(self):
        """再平衡列表中的 delta_weight 与 (target - current) 完全一致"""
        ...
```

***

## 12. 项目目录结构

```
portfolio_layer/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/                          # 配置文件（YAML）
│   ├── base.yaml                     # 基础默认配置
│   ├── production.yaml               # 生产环境配置
│   ├── backtest.yaml                 # 回测环境配置
│   └── schemas/                      # 配置 Schema 验证
│       └── config_schema.yaml
│
├── signal_fusion/                    # Alpha 融合层
│   ├── __init__.py
│   ├── alpha_combiner.py             # AlphaCombiner 接口类
│   ├── single_model.py               # SingleModelFusion
│   ├── weighted_average.py           # WeightedAverageFusion
│   ├── multi_domain.py               # MultiDomainFusion
│   ├── hierarchical.py               # HierarchicalFusion
│   └── utils.py                      # zscore, winsorize 等工具函数
│
├── candidate_selection/              # 候选池构建层
│   ├── __init__.py
│   ├── candidate_selector.py         # CandidateSelector
│   └── filters/
│       ├── tradability.py            # 停牌/涨跌停/新股过滤
│       ├── liquidity.py              # 流动性过滤
│       └── market_cap.py             # 市值过滤
│
├── risk_model/                       # 风险暴露模型层
│   ├── __init__.py
│   ├── risk_exposure_builder.py      # RiskExposureBuilder
│   ├── factor_definitions/
│   │   ├── industry_factor.py        # 行业哑变量
│   │   ├── size_factor.py
│   │   ├── value_factor.py
│   │   ├── momentum_factor.py
│   │   ├── volatility_factor.py
│   │   ├── liquidity_factor.py
│   │   └── beta_factor.py
│   └── covariance/
│       ├── factor_cov_estimator.py   # 因子协方差矩阵估计
│       └── shrinkage.py              # Ledoit-Wolf 收缩估计
│
├── constraints/                      # 约束构建层
│   ├── __init__.py
│   ├── constraint_builder.py         # ConstraintBuilder
│   ├── constraint_set.py             # ConstraintSet 数据类
│   └── penalty_relaxer.py            # 软约束松弛逻辑
│
├── optimizer/                        # 组合优化层
│   ├── __init__.py
│   ├── portfolio_optimizer.py        # PortfolioOptimizer（cvxpy 实现）
│   ├── topk_builders.py              # Top-K 系列构造函数
│   ├── optimizer_config.py           # OptimizerConfig
│   └── solver_utils.py              # 求解器工具，热启动，超时处理
│
├── postprocess/                      # 权重后处理层
│   ├── __init__.py
│   ├── weight_post_processor.py      # WeightPostProcessor
│   ├── rounding.py                   # 权重离散化/取整
│   └── position_scaler.py            # PositionScaler（仓位缩放）
│
├── reporting/                        # 报告与归因层
│   ├── __init__.py
│   ├── risk_reporter.py              # RiskReporter
│   ├── attribution.py               # 风险归因
│   └── templates/
│       └── daily_risk_report.jinja2
│
├── data_models/                      # 数据对象定义
│   ├── __init__.py
│   ├── alpha_frame.py                # AlphaFrame, CompositeAlphaFrame
│   ├── candidate_universe.py         # CandidateUniverse
│   ├── risk_exposure_frame.py        # RiskExposureFrame
│   ├── constraint_set.py             # ConstraintSet
│   ├── target_portfolio.py           # TargetPortfolio
│   ├── portfolio_risk_report.py      # PortfolioRiskReport
│   └── market_state_signal.py        # MarketStateSignal
│
├── degradation/                      # 降级管理层
│   ├── __init__.py
│   ├── degradation_manager.py        # DegradationManager
│   └── exceptions.py                 # FatalDegradationError 等自定义异常
│
├── pipeline/                         # 流水线编排层
│   ├── __init__.py
│   ├── portfolio_pipeline.py         # 主流水线，组合各模块
│   └── daily_runner.py              # 日度运行入口
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── mock_alpha_frames.py
│   │   ├── mock_market_data.py
│   │   └── mock_snapshots/          # 固定日期快照用于一致性测试
│   ├── unit/
│   ├── integration/
│   ├── boundary/
│   ├── robustness/
│   ├── stress/
│   └── consistency/
│
└── notebooks/                        # 研究与调试 Notebook（不进入生产）
    ├── explore_fusion_methods.ipynb
    └── optimizer_tuning.ipynb
```

***

## 13. 类接口与伪代码说明

### 13.1 核心模块接口总览

#### `AlphaCombiner`（抽象基类）

```python
from abc import ABC, abstractmethod

class AlphaCombiner(ABC):
    """
    所有 Alpha 融合方法的抽象基类。
    所有子类必须实现 fuse() 方法。
    """
    @abstractmethod
    def fuse(self, frames: List[AlphaFrame]) -> CompositeAlphaFrame:
        """
        输入：一批 AlphaFrame（部分可能 available=False）
        输出：CompositeAlphaFrame（截面标准化的复合打分）
        """
        pass

    def validate_frames(self, frames: List[AlphaFrame]) -> None:
        """通用校验：date 一致性，scores 格式等"""
        dates = [f.date for f in frames if f.available]
        assert len(set(dates)) <= 1, f"AlphaFrame 日期不一致: {set(dates)}"
```

#### `CandidateSelector`

```python
class CandidateSelector:
    def __init__(self, config: CandidateSelectorConfig): ...
    def build(self, date, composite_alpha, stk_limit_df, suspend_df,
              daily_basic_df, index_member_df) -> CandidateUniverse: ...
    # 内部方法（下划线前缀）
    def _get_suspended(self, date, suspend_df) -> Set[str]: ...
    def _get_limit_up(self, date, stk_limit_df) -> Set[str]: ...
    def _get_limit_down(self, date, stk_limit_df) -> Set[str]: ...
    def _get_new_stocks(self, date, daily_basic_df, min_listed_days) -> Set[str]: ...
    def _get_illiquid(self, date, daily_basic_df, min_turnover) -> Set[str]: ...
    def _get_st_stocks(self, index_member_df) -> Set[str]: ...
```

#### `RiskExposureBuilder`

```python
class RiskExposureBuilder:
    def __init__(self, config: RiskExposureConfig): ...
    def build(self, date, candidate_universe, index_member_df, daily_basic_df,
              price_df, fundamental_df, benchmark_weights) -> RiskExposureFrame: ...
    # 因子计算接口（每个因子一个方法，便于单独测试）
    def calc_factor(self, factor_name: str, date, stocks, **data) -> pd.Series: ...
```

#### `ConstraintBuilder`

```python
class ConstraintBuilder:
    def __init__(self, config: ConstraintBuilderConfig): ...
    def build(self, date, candidate_universe, risk_exposure,
              daily_basic_df, prev_weights) -> ConstraintSet: ...
    def relax(self, constraint_set: ConstraintSet,
              relaxation_level: int) -> ConstraintSet: ...
```

#### `PortfolioOptimizer`

```python
class PortfolioOptimizer:
    def __init__(self, config: OptimizerConfig): ...
    def optimize(self, composite_alpha, risk_exposure, constraint_set,
                 prev_weights, cov_matrix=None) -> Tuple[pd.Series, str]: ...
    def _build_problem(self, ...) -> cp.Problem: ...
    def _fallback_topk(self, composite_alpha, k) -> pd.Series: ...
```

#### `WeightPostProcessor`

```python
class WeightPostProcessor:
    """
    对优化器输出的原始权重做以下后处理：
    1. 清除微小持仓（低于 min_weight_threshold）
    2. 权重归一化
    3. 离散化（如需按手数取整）
    4. 生成再平衡列表
    """
    def __init__(self, config: PostProcessConfig): ...

    def process(
        self,
        raw_weight: pd.Series,
        prev_weight: pd.Series,
        total_asset: Optional[float] = None,
        price_series: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        返回：(final_weight, target_position, rebalance_list)
        """
        # 步骤 1：清除微小持仓
        cleaned = raw_weight.copy()
        cleaned[cleaned < self.config.min_weight_threshold] = 0.0
        cleaned = cleaned / cleaned.sum()  # 归一化

        # 步骤 2：股数离散化（可选）
        if total_asset is not None and price_series is not None:
            target_position = self._discretize(cleaned, total_asset, price_series)
        else:
            target_position = pd.Series(dtype=float)

        # 步骤 3：生成再平衡列表
        rebalance_list = self._build_rebalance_list(cleaned, prev_weight)

        return cleaned, target_position, rebalance_list

    def _discretize(self, weight, total_asset, price) -> pd.Series:
        """按手（100 股）取整"""
        shares_float = weight * total_asset / price
        shares_rounded = (shares_float / 100).round() * 100
        return shares_rounded.clip(lower=0)

    def _build_rebalance_list(self, target, prev) -> pd.DataFrame:
        all_stocks = target.index.union(prev.index)
        target_full = target.reindex(all_stocks, fill_value=0.0)
        prev_full = prev.reindex(all_stocks, fill_value=0.0)
        delta = target_full - prev_full
        direction = delta.apply(lambda x: "BUY" if x > 0 else ("SELL" if x < 0 else "HOLD"))
        return pd.DataFrame({
            "ts_code": all_stocks,
            "current_weight": prev_full,
            "target_weight": target_full,
            "delta_weight": delta,
            "direction": direction,
        }).set_index("ts_code")
```

#### `PortfolioExporter`

```python
class PortfolioExporter:
    """
    将 TargetPortfolio 导出为标准化格式（CSV、Parquet、数据库）。
    """
    def export(
        self,
        portfolio: TargetPortfolio,
        output_path: str,
        format: str = "parquet",
    ) -> None:
        data = {
            "date": portfolio.date,
            "ts_code": portfolio.target_weight.index.tolist(),
            "target_weight": portfolio.target_weight.values.tolist(),
            "is_degraded": portfolio.is_degraded,
            "optimizer_status": portfolio.optimizer_status,
            "fusion_method": portfolio.fusion_method,
        }
        df = pd.DataFrame(data)
        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
```

#### `RiskReporter`

```python
class RiskReporter:
    """
    生成组合风险归因报告。
    """
    def report(
        self,
        portfolio: TargetPortfolio,
        risk_exposure: RiskExposureFrame,
        prev_portfolio: Optional[TargetPortfolio] = None,
    ) -> PortfolioRiskReport:
        w = portfolio.target_weight
        bm_w = risk_exposure.benchmark_weights

        # 行业主动暴露
        ind_exp = risk_exposure.industry_exposure
        portfolio_industry = ind_exp.T @ w.reindex(ind_exp.index, fill_value=0)
        bm_industry = ind_exp.T @ bm_w.reindex(ind_exp.index, fill_value=0)
        industry_active = portfolio_industry - bm_industry

        # 风格主动暴露
        sty_exp = risk_exposure.style_exposure
        portfolio_style = sty_exp.T @ w.reindex(sty_exp.index, fill_value=0)
        bm_style = sty_exp.T @ bm_w.reindex(sty_exp.index, fill_value=0)
        style_active = portfolio_style - bm_style

        # 换手率
        turnover = 0.0
        if prev_portfolio is not None:
            prev_w = prev_portfolio.target_weight
            all_stocks = w.index.union(prev_w.index)
            delta = w.reindex(all_stocks, fill_value=0) - prev_w.reindex(all_stocks, fill_value=0)
            turnover = delta.abs().sum() / 2

        # 集中度
        top10_weight = w.nlargest(10).sum()
        hhi = (w ** 2).sum()

        return PortfolioRiskReport(
            date=portfolio.date,
            industry_exposure_active=industry_active,
            style_exposure_active=style_active,
            top10_weight=top10_weight,
            stock_count=(w > 0).sum(),
            estimated_tracking_error=0.0,   # 需协方差矩阵，MVP 阶段留空
            estimated_active_risk=0.0,
            turnover_rate=turnover,
            constraint_violations=self._check_violations(portfolio, risk_exposure),
            herfindahl_index=hhi,
        )
```

### 13.2 主流水线 `PortfolioPipeline`

```python
class PortfolioPipeline:
    """
    组合层主流水线，编排所有子模块。
    """
    def __init__(
        self,
        alpha_combiner: AlphaCombiner,
        candidate_selector: CandidateSelector,
        risk_exposure_builder: RiskExposureBuilder,
        constraint_builder: ConstraintBuilder,
        optimizer: PortfolioOptimizer,
        post_processor: WeightPostProcessor,
        position_scaler: PositionScaler,
        exporter: PortfolioExporter,
        reporter: RiskReporter,
        degradation_manager: DegradationManager,
    ):
        self.alpha_combiner = alpha_combiner
        self.candidate_selector = candidate_selector
        self.risk_builder = risk_exposure_builder
        self.constraint_builder = constraint_builder
        self.optimizer = optimizer
        self.post_processor = post_processor
        self.position_scaler = position_scaler
        self.exporter = exporter
        self.reporter = reporter
        self.degradation = degradation_manager

    def run(
        self,
        date: str,
        alpha_frames: List[AlphaFrame],
        market_data: MarketDataBundle,
        prev_portfolio: Optional[TargetPortfolio] = None,
        market_signal: Optional[MarketStateSignal] = None,
    ) -> Tuple[TargetPortfolio, PortfolioRiskReport]:
        """
        完整运行一天的组合生成流程。
        """
        logger.info(f"[{date}] 开始组合层流水线运行")

        # Step 1: Alpha 融合
        combiner = self.degradation.select_fusion_method(alpha_frames)
        composite_alpha = combiner.fuse(alpha_frames)
        logger.info(f"[{date}] Alpha 融合完成，使用方法: {composite_alpha.fusion_method}")

        # Step 2: 候选池构建
        candidate_universe = self.candidate_selector.build(
            date, composite_alpha,
            market_data.stk_limit, market_data.suspend,
            market_data.daily_basic, market_data.index_member,
        )
        # 降级处理
        fallback = self.degradation.handle_empty_candidate_pool(
            date, candidate_universe, prev_portfolio
        )
        if fallback is not None:
            return fallback, self.reporter.report(fallback, RiskExposureFrame.empty(date))

        # Step 3: 风险暴露构建
        risk_exposure = self.risk_builder.build(
            date, candidate_universe,
            market_data.index_member, market_data.daily_basic,
            market_data.price, market_data.fundamental,
            benchmark_weights=market_data.benchmark_weights,
        )

        # Step 4: 约束构建
        prev_weights = prev_portfolio.target_weight if prev_portfolio else pd.Series(dtype=float)
        constraint_set = self.constraint_builder.build(
            date, candidate_universe, risk_exposure,
            market_data.daily_basic, prev_weights,
        )

        # Step 5: 组合优化
        raw_weight, optimizer_status = self.optimizer.optimize(
            composite_alpha, risk_exposure, constraint_set,
            prev_weights, cov_matrix=market_data.cov_matrix,
        )

        # Step 6: 权重后处理
        final_weight, target_position, rebalance_list = self.post_processor.process(
            raw_weight, prev_weights,
            total_asset=market_data.total_asset,
            price_series=market_data.price_today,
        )

        # Step 7: 仓位缩放（指数信号）
        prev_gross = prev_portfolio.gross_exposure if prev_portfolio else 1.0
        scaled_weight, gross_exposure, cash_ratio = self.position_scaler.scale(
            final_weight, market_signal, prev_gross
        )

        # Step 8: 构建 TargetPortfolio
        portfolio = TargetPortfolio(
            date=date,
            target_weight=scaled_weight,
            target_position=target_position,
            rebalance_list=rebalance_list,
            gross_exposure=gross_exposure,
            cash_ratio=cash_ratio,
            optimizer_status=optimizer_status,
            fusion_method=composite_alpha.fusion_method,
            is_degraded=composite_alpha.is_degraded or optimizer_status == "degraded",
        )

        # Step 9: 风险报告
        risk_report = self.reporter.report(portfolio, risk_exposure, prev_portfolio)

        # Step 10: 导出
        self.exporter.export(portfolio, output_path=f"outputs/{date}/portfolio.parquet")

        logger.info(f"[{date}] 组合层流水线完成，持仓数: {risk_report.stock_count}，"
                    f"换手率: {risk_report.turnover_rate:.2%}，状态: {optimizer_status}")
        return portfolio, risk_report
```

***

## 14. MVP / 增强版 / 远期版路线图

### 14.1 MVP 版本（最小可行版本）

**目标**：在最短时间内上线一个能工作的组合层，用于验证 Alpha 有效性。

**包含内容**：

| 模块       | MVP 实现                                                |
| :------- | :---------------------------------------------------- |
| Alpha 融合 | `SingleModelFusion` 或 `WeightedAverageFusion`（最多 2 域） |
| 候选池      | 基础停牌/涨跌停/新股过滤，不含流动性过滤                                 |
| 风险暴露     | 仅行业哑变量（来自 `raw_index_member_all`）                     |
| 约束       | 单票上限 5%、行业偏离 10%、总权重 = 1                              |
| 优化器      | `TopK Equal Weight` 或 `TopK Score Weighted`（无二次规划）    |
| 仓位缩放     | 固定满仓（不使用指数信号）                                         |
| 报告       | 基础行业暴露报告 + 换手率统计                                      |

**MVP 不包含**：

- 协方差矩阵
- 风格因子
- 换手惩罚（优化器层面）
- 仓位缩放
- 替补池机制
- 软约束松弛

### 14.2 增强版（生产就绪版本）

**在 MVP 基础上新增**：

| 模块       | 增强实现                                   |
| :------- | :------------------------------------- |
| Alpha 融合 | `HierarchicalFusion`（全 5 域，含降级机制）      |
| 候选池      | 完整流动性过滤 + ADV 约束 + 市值分层 + Buffer 机制    |
| 风险暴露     | 行业 + 完整 7 个风格因子（含 beta、momentum）       |
| 约束       | 全套 `ConstraintSet`，含风格偏离约束和换手约束        |
| 优化器      | `PortfolioOptimizer`（cvxpy 二次规划，含换手惩罚） |
| 协方差矩阵    | 行业因子协方差模型（简化版 Barra）                   |
| 仓位缩放     | `PositionScaler`（EMA 平滑 + 上下限）         |
| 软约束松弛    | `PenaltyRelaxer`（优先级松弛）                |
| 报告       | 完整 `PortfolioRiskReport`，含风格归因         |
| 测试       | 完整 unit + integration + boundary 测试集   |

### 14.3 远期版（研究增强版）

**在增强版基础上进一步**：

| 功能          | 描述                                 |
| :---------- | :--------------------------------- |
| 动态 IC 加权    | 融合权重按滚动 IC 动态更新                    |
| 完整 Barra 模型 | 包含风格因子协方差估计（Ledoit-Wolf 收缩）        |
| 多期换手优化      | 考虑 T+1 至 T+N 的预期换手成本（动态规划）         |
| 跟踪误差约束      | 基于完整协方差矩阵的 TE 约束                   |
| 风险预算分配      | 按行业/风格分配风险预算                       |
| 指数信号集成      | 完整 `MarketStateSignal`（来自独立预测模型）   |
| 稳健性优化       | Worst-case alpha 优化（应对 Alpha 估计误差） |
| 在线监控        | 实时检测组合暴露偏离触发再平衡告警                  |
| 多基准支持       | 支持沪深 300、中证 500、中证 1000 等多个基准      |

### 14.4 版本演进决策原则

> **原则：简单基线必须与复杂优化器并存，不允许在 Alpha 未经验证时过早引入复杂优化器。**

推荐演进路径：

```
Week 1-2：MVP（TopK + 基础约束）
    → 验证 Alpha IC 有效性
    → 验证候选池逻辑的正确性

Week 3-4：引入行业约束 + TopK Buffer
    → 验证行业中性化效果
    → 开始收集换手数据

Month 2：引入二次规划优化器
    → 先用对角协方差（忽略因子相关性）
    → 逐步加入换手惩罚

Month 3+：引入完整风格因子 + 动态融合权重
    → 只有在风格因子 IC 经验证后才启用风格约束
```

***

## 15. 附录：数据表字段参考

### 15.1 `raw_stk_limit` 关键字段

| 字段名          | 类型    | 说明               |
| :----------- | :---- | :--------------- |
| `ts_code`    | str   | 股票代码             |
| `trade_date` | str   | 交易日期 YYYYMMDD    |
| `up_limit`   | float | 涨停价              |
| `down_limit` | float | 跌停价              |
| `close`      | float | 收盘价（用于判断是否触及涨跌停） |

**涨停判断逻辑**：`close >= up_limit`（或 `close == up_limit`，取决于数据提供方规范）。

### 15.2 `raw_suspend_d` 关键字段

| 字段名            | 类型  | 说明               |
| :------------- | :-- | :--------------- |
| `ts_code`      | str | 股票代码             |
| `suspend_date` | str | 停牌日期             |
| `resume_date`  | str | 复牌日期（为空表示仍处于停牌中） |
| `suspend_type` | str | 停牌类型             |

**停牌判断逻辑**：`suspend_date <= target_date AND (resume_date IS NULL OR resume_date > target_date)`。

### 15.3 `raw_daily_basic` 关键字段

| 字段名             | 类型    | 用途                    |
| :-------------- | :---- | :-------------------- |
| `ts_code`       | str   | 股票代码                  |
| `trade_date`    | str   | 交易日                   |
| `total_mv`      | float | 总市值（万元），用于市值过滤        |
| `circ_mv`       | float | 流通市值（万元），`size` 因子    |
| `free_share`    | float | 自由流通股本，流动性代理          |
| `turnover_rate` | float | 换手率（%），`liquidity` 因子 |
| `volume_ratio`  | float | 量比，异常成交识别             |
| `pe_ttm`        | float | PE（TTM），估值参考          |
| `pb`            | float | 市净率，`value` 因子备用      |
| `ps_ttm`        | float | 市销率（TTM）              |
| `dv_ttm`        | float | 股息率（TTM）              |

### 15.4 `feat_feature_D_fundamental` 关键字段

| 字段名                    | 类型    | 用途                   |
| :--------------------- | :---- | :------------------- |
| `ts_code`              | str   | 股票代码                 |
| `trade_date`           | str   | 交易日                  |
| `val_rank`             | float | 估值 rank（截面百分位，越高越便宜） |
| `size_rank`            | float | 规模 rank（截面百分位，越高越大盘） |
| `val_score_compressed` | float | 压缩估值分数，已处理极端值        |

***

## 附录 B：关键决策记录

### B.1 为何选择 cvxpy 而非自实现优化器

- cvxpy 支持多个后端求解器（CLARABEL、SCS、OSQP），可在约束不可行时自动切换
- 接口语义清晰，约束表达式与数学公式直接对应，便于审计
- 支持热启动（warm start），日度再优化效率高
- 缺点：大型问题（N > 3000）可能较慢，此时可预筛候选池至 500–1000 只股票

### B.2 为何不在组合层做 Alpha 衰减

Alpha 衰减（decay）属于预测层的时间序列建模问题，不属于组合层职责。组合层只接收当日最新的 `alpha_score`，由预测层决定是否在 alpha 中体现衰减特性。

### B.3 为何用换手惩罚而非换手硬约束

硬约束可能导致优化问题不可行（尤其在市场极端日），而惩罚项通过调节 (\gamma) 参数实现软控制，更稳健。生产中推荐同时设置换手惩罚项 + 较宽松的换手率硬约束上限（如 50%）作为安全兜底。

### B.4 `alpha_score` 的截面标准化时机

标准化应在**融合层输入前**完成（即在 `AlphaFrame` 进入 `AlphaCombiner` 之前），确保各域分数在同一量纲上，避免高方差域主导融合结果。若 `AlphaFrame` 传入时已标准化，`AlphaCombiner` 内部需跳过重复标准化。通过 `AlphaFrame.meta["is_normalized"]` 标识字段来传递这一信息。

***

*文档结束*

**版本历史**：

| 版本     | 日期      | 变更                   |
| :----- | :------ | :------------------- |
| v1.0.0 | 2025-07 | 初始版本，涵盖 MVP 至增强版完整设计 |

