
# Portfolio Layer Engineering (量化组合层) 深度技术白皮书

## 一、 系统概述与定位 (System Overview)

在现代机构级量化交易系统中，系统通常被严格划分为三个核心层次：

1. **预测层 (Alpha Layer)**：负责处理海量异构数据，通过多因子模型、机器学习或深度学习模型，输出微观个股级别的收益预测（Alpha Z-score）。
2. **组合层 (Portfolio Layer)**：本项目的核心。它承接预测层输出的个股信号，并结合宏观/中观约束（市场状态、行业中性、风格暴露、换手率限制等），通过数学凸优化（Quadratic Programming, QP）求解出最优的**目标持仓权重（Target Portfolio）**。
3. **执行层 (Execution Layer)**：接收组合层发出的调仓指令（Rebalance List），结合高频微观结构进行订单拆分（如TWAP/VWAP）、算法交易与路由。

### 1.1 核心逻辑与使命

本项目的根本逻辑是解决一个带有复杂约束的**二次规划问题**：

$$
\max_{w} \left( \mu^T w - \lambda \cdot w^T \Sigma w - c \cdot \|w - w_{prev}\|_1 - \text{Penalties} \right)
$$

- **最大化预期收益 ($\mu^T w$)**：尽可能多地买入 Alpha 分数高的股票。
- **控制组合风险 ($w^T \Sigma w$)**：通过多因子协方差矩阵控制组合波动率。
- **控制交易成本 ($\|w - w_{prev}\|_1$)**：惩罚过度换手带来的滑点与手续费。
- **满足合规与风控约束 (Penalties)**：如单票上限、行业中性偏离、风格暴露、涨跌停无法交易等。

### 1.2 架构设计亮点

本项目采用了严谨的 **领域驱动设计（DDD）**，实现了 Alpha、Beta、Risk 与 Constraints 的四元解耦。

- **高韧性容灾**：构建了从数据缺失降级、软约束松弛到无解回退 Top-K 的三级生产防线。
- **极限性能**：底层数据加载采用 `PyArrow` 谓词下推技术；协方差计算采用因子降维矩阵变换；优化求解器自适应适配。
- **数据契约强类型**：系统内部彻底屏弃裸字典，全部采用 `dataclasses` 强类型对象，确保上下游协作的确定性。

---

## 二、 核心数据契约与接口定义 (Data Interfaces)

为了方便与其他量化项目（如 Alpha 挖掘平台、执行层网关）对接，本系统在 `portfolio_layer/data_models/__init__.py` 中定义了严格的数据契约。**所有传入与传出数据必须符合以下结构**。

### 2.1 上游输入数据 (Inputs)

启动 Portfolio Pipeline，需要提供三大核心数据束：

#### 1. `AlphaFrame` (预测信号帧)

代表单日、单一域或单一模型的 Alpha 分数快照。支持多模型多域输入，系统内部会自动进行信号融合。

- `date` (str): 交易日期，如 `"20260313"`。
- `domain` (str): 信号域，如 `"A"`（量价）、`"B"`（基本面）、`"C"`（另类数据）。
- `model_id` (str): 产生该信号的模型标识。
- `scores` (pd.Series): 核心预测数据。Index 为 `ts_code` (股票代码)，Value 为归一化后的 Alpha 分数。
- `horizon` (int): 预测的时间窗口（如 5 天）。

#### 2. `MarketDataBundle` (市场数据束)

系统运转所需的市场快照。为解决海量历史数据内存爆炸问题，项目通过 `ParquetDataLoader` 实时按需加载。

- `daily_basic` (pd.DataFrame): 当日基础数据（总市值 `total_mv`、流通市值 `circ_mv`、换手率 `turnover_rate` 等）。
- `stk_limit` (pd.DataFrame): 涨跌停价格表（`up_limit`, `down_limit`）。
- `suspend` (pd.DataFrame): 停牌数据表。
- `price` (pd.DataFrame): 近期量价时序数据，用于计算动量、波动率及特质风险。
- `fundamental` (pd.DataFrame): 截面财务特征（如估值因子等）。
- `index_member` (pd.DataFrame): 指数成分股（用于提取基准权重与新股过滤）。
- `benchmark_weights` (pd.Series): 业绩基准权重（如沪深300、中证500的成分权重）。

#### 3. `MarketStateSignal` (可选，宏观择时信号)

控制组合总敞口的开关。

- `gross_exposure_scale` (float): 建议的总多头仓位（例如大盘环境极差时，传入 0.8，表示最多持有 80% 仓位）。
- `cash_ratio_signal` (float): 建议现金比例。

### 2.2 下游输出数据 (Outputs)

流水线运行完毕后，输出一个严格的元组 `(TargetPortfolio, PortfolioRiskReport)`。

#### 1. `TargetPortfolio` (目标组合——对接执行层)

- `target_weight` (pd.Series): 优化后的绝对目标权重（求和等于 `gross_exposure`）。
- `target_position` (pd.Series): 转换后的目标持仓股数或名义金额。
- `rebalance_list` (pd.DataFrame): **执行层最关心的数据**。包含了当前持仓与目标持仓的差额，明确标示了每一只股票的**买入/卖出方向及目标数量**。
- `optimizer_status` (str): 优化器最终状态（`optimal`, `optimal_inaccurate` 或降级 `degraded`）。
- `is_degraded` (bool): 布尔值，标识本次运行是否触发了非标准逻辑（如数据缺失或优化失败导致的回退），用于下游预警。

#### 2. `PortfolioRiskReport` (组合风险报告——对接风控层)

- `stock_count` (int): 最终选中的股票数量。
- `turnover_rate` (float): 预估单边换手率。
- `industry_exposure_active` (pd.Series): 各行业相对于 Benchmark 的主动偏离权重。
- `style_exposure_active` (pd.Series): 各风格因子的主动偏离暴露。
- `constraint_violations` (List[str]): 若触发了软约束，此处记录超标的具体项目与超标幅度。

---

## 三、 详细流水线与处理细节 (Detailed Pipeline Workflow)

在 `portfolio_layer/pipeline/__init__.py` 中，`PortfolioPipeline.run()` 方法串联了 10 个标准阶段。以下是每个阶段的处理细节与数学逻辑：

### 阶段 1：Alpha 降级与融合 (Signal Fusion)

量化实盘中，经常遇到某个数据源当天延迟或缺失。系统首先调用 `DegradationManager` 评估 `AlphaFrame` 的完整性。

- **处理**：若所有域正常，采用多域加权或树状层次融合（Hierarchical Fusion）。若缺失部分域，自动降级为可用域的重新加权，并将缺失域记录到日志。
- **输出**：生成一份归一化、且无缺失值的 `CompositeAlphaFrame`。

### 阶段 2：候选池清洗与不对称可交易过滤 (Candidate Selection)

剔除物理上不可交易或不符合合规要求的股票。

- **处理细节**：过滤停牌股、上市不满 60 天的新股、ST 股以及日换手率极低（如 < 0.1%）的僵尸股。
- **【核心细节】涨跌停的不对称处理**：这是本系统的亮点。涨跌停股票**不能被简单地从候选池中剔除**！
  - 若将跌停股剔除，优化器认为目标权重为 0，会向执行层发送“卖出跌停股”的幻象指令（实际上卖不掉）。
  - 若将涨停股剔除，如果该股 Alpha 极高，优化器无法为其保留目标权重，导致后续踏空。
  - **正确操作**：将它们保留在候选池中，但在后续 Constraint 层施加**不对称掩码**（跌停且持仓：锁定下界为当前权重；涨停且无持仓：锁定上界为 0）。

### 阶段 3：Barra 风险暴露构建 (Risk Exposure Build)

将个股映射到低维度的因子空间。

- **处理细节**：根据个股的时序与截面数据，计算其在传统 Barra 风格因子（Size, Value, Momentum, Volatility, Liquidity, Beta）以及行业哑变量（Industry Dummies）上的暴露度，产出 `RiskExposureFrame` 矩阵 $B$。

### 阶段 4：多因子协方差估计 (Factor Covariance Estimation)

如果直接计算 5000 只股票的协方差矩阵，会得到一个 $5000 \times 5000$ 的稠密矩阵，不仅计算极其耗时，且由于样本天数 $T \ll N$，矩阵必然是奇异的（不可逆的）。

- **处理细节**：系统采用 `FactorCovEstimator`，利用公式 $\Sigma = B \cdot F \cdot B^T + \Delta$ 将维度降级。通过历史收益率对暴露矩阵 $B$ 做截面回归，倒推因子收益率协方差 $F$ ($K \times K$，通常 K<50)，并提取对角特质风险 $\Delta$。
- **技术亮点**：采用 Ledoit-Wolf 或 Oracle Approximating Shrinkage (OAS) 对样本协方差进行收缩，保障矩阵在极值行情下的正定性与数值稳定性。

### 阶段 5：动态约束生成 (Dynamic Constraints)

结合当前持仓（`prev_weights`）与基准权重，生成优化求解域。

- 构建单票权重上下界（如 `0 <= w <= 0.05`）。
- 构建总仓位上下界（如 `0.98 <= sum(w) <= 1.02`）。
- 构建行业最大主动偏离界限（如 `|w_ind - bm_ind| <= 0.05`）。

### 阶段 6：软约束凸优化求解 (Convex Optimization) - 【系统的数学心脏】

在 `optimizer/portfolio_optimizer.py` 中，使用 `cvxpy` 构建凸优化问题。

- **软约束松弛体系**：如果将行业中性和换手率设置为硬约束，在 Alpha 信号极度集中或基准剧烈变动时，优化器会直接报 `Infeasible`（无解）。本系统将行业偏离和换手率转换为软约束，引入松弛变量（`deviation`）：

  $$
  \text{Constraints: } \text{ind\_exposure} - \text{bm\_exposure} \le \text{limit} + \text{deviation}
  $$

  $$
  \text{Objective: } \max (\dots - \text{penalty} \times \text{deviation})
  $$
- **自适应惩罚系数 (`PenaltyCalibrator`)**：传统系统常将 penalty 写死为 `10.0`。但在不同市场阶段，Alpha 分数的绝对尺度会波动。如果 Alpha 极小，惩罚极大，优化器会退化为“最小化调仓工具”；反之则无视约束。`PenaltyCalibrator` 会根据每日 Alpha 分数的 $L_{\infty}$ 范数动态标定行业、风格和换手的惩罚系数，保持优化语义的恒定。
- **降级机制**：若主求解器（如 CLARABEL / OSQP）超时或失败，自动切换至 SCS 求解器；若彻底失败，降级为选股池的 Top-K 等权分配，确保实盘绝不宕机。

### 阶段 7 & 8：后处理与仓位缩放 (Postprocess & Scaling)

- **碎股清理**：将优化出的小于 0.5% 的极微小仓位清零，避免产生大量无效的零碎订单，增加执行层压力。
- **平滑缩放**：根据 `MarketStateSignal` 对总仓位进行同比例缩放。为了防止信号剧烈跳动，会与前一日总仓位做 EMA 平滑。

### 阶段 9 & 10：风控归因与结果导出

生成风险报告，进行事后约束检查，并将结果写入 `outputs/YYYYMMDD/portfolio.parquet`。

---

## 四、 核心架构与操作细节剖析 (Deep Dive & Architecture)

### 4.1 PyArrow 数据加载层：性能革命

量化系统中，I/O 是最常见的瓶颈。本系统在 `data_loaders.py` 中实现了 `ParquetDataLoader`。
**痛点**：过去使用 `pd.read_parquet()`，读取 5 年历史全量逐日数据时，需将上百 GB 数据拉入内存再做 Filter，导致 OOM（内存溢出）。
**解决方案**：采用 `pyarrow.dataset` 的**谓词下推（Predicate Pushdown）**。

```python
dataset = ds.dataset(file_path, format="parquet")
# 过滤条件在 C++ 底层扫描磁盘时直接生效，不符合的数据根本不会反序列化进内存
table = dataset.to_table(filter=ds.field("trade_date") == date) 
```

这使得读取单日截面数据的耗时从数秒降低到了毫秒级，为全量并发回测扫清了障碍。

### 4.2 高效二次型计算：绕过 $O(N^2)$ 的陷阱

在优化器目标函数中，风险惩罚项为 $w^T \Sigma w$。若显式构建 $N \times N$ 的 $\Sigma$，在 cvxpy 中计算耗时极长。
**优化操作**：利用协方差因子模型 $\Sigma = B F B^T + \text{diag}(\sigma^2)$ 进行数学等价替换：

```python
# F_chol 是 K x K 的因子协方差矩阵的 Cholesky 分解
factor_loading = F_chol.T @ B.T @ w   # 维度压缩到 K
risk_factor_term = cp.sum_squares(factor_loading)
risk_specific_term = cp.sum_squares(cp.multiply(sigma_specific, w))
risk_penalty = risk_factor_term + risk_specific_term
```

这一操作将时间复杂度从 $O(N^2)$ 降为 $O(NK + K^2)$，对于 $N=3000, K=50$ 的场景，理论加速比近千倍。

### 4.3 目录结构与领域划分

- `candidate_selection/`: 负责回答“今天到底哪些股票允许买卖”。
- `constraints/`: 负责将投资经理的业务规则翻译为数学不等式。
- `data_models/`: 定义所有数据的 Type Hint 和 Dataclass，是整个工程的“宪法”。
- `degradation/`: 异常流控制中枢，拦截并降级所有可能导致实盘中断的异常。
- `optimizer/`: 纯粹的数学引擎，只认矩阵不认业务。

---

## 五、 如何使用与测试部署 (Usage & Deployment)

### 5.1 环境配置

项目推荐使用 `conda` 管理环境。依赖的核心包包括 `pandas`, `numpy`, `pyarrow`, `cvxpy` (及其依赖求解器如 `scs`, `osqp`, `clarabel`), `pytest`。

```powershell
conda create -n universal python=3.11
conda activate universal
pip install pandas numpy pyarrow cvxpy pytest structlog
```

### 5.2 运行真实流水线仿真

开发者可直接通过 `run_real_data_pipeline.py` 进行单日端到端穿透测试。
该脚本会：

1. 从 `data/Raw_data` 及 `Feature_data` 读取所需的 Parquet 切片。
2. 构造虚拟 Alpha（或读取真实 Alpha）。
3. 执行从过滤、风险估计到优化的全套流程。

```powershell
cd D:\Trading\portfolio_layer_engineering\portfolio_layer
python run_real_data_pipeline.py
```

**调参说明**：在 `run_real_data_pipeline.py` 中，通过修改 `OptimizerConfig` 即可调整组合风格：

- 增大 `risk_aversion`：持仓会更加分散，降低波动率。
- 增大 `eta_turnover`：系统会变得“懒惰”，即使有新 Alpha 也不轻易调仓，节省手续费。
- 修改 `fallback_topk`：控制当优化器无解时，等权买入前多少只股票作为备用方案。

### 5.3 测试体系保障

项目在 `tests/` 下构建了单元测试（Unit）与集成测试（Integration）。

- **`test_alpha_combiner.py`**: 测试多域 Alpha 缺失时的融合回退逻辑。
- **`test_integration.py`**: 利用 `mock_data_generator.py` 动态生成虚拟行情与因子，验证端到端调仓逻辑是否连贯。
  执行命令：`python -m pytest tests/`

---

## 六、 潜在问题、使用避坑与改进方向 (Issues & Future Improvements)

根据本项目的《深度评估与改造文档》，系统目前虽已达到“生产就绪仿真”级别，但要跃升至管理百亿资金的“机构级生产可用”状态，仍需注意以下潜在问题及改进建议：

### 6.1 当前潜在问题与避坑指南 (Pitfalls)

1. **极端行情下的换手率崩塌**：
   如果市场发生剧烈风格切换（如微盘股暴跌，基准成分股权重巨变），前一日的持仓与今日的约束会发生严重冲突。如果强行锁定 `turnover_ub = 0.2`（单日换手20%），优化器极大概率会报错。
   **应对**：务必开启 `OptimizerConfig.use_dynamic_penalty = True`，利用软约束让系统在“超限”与“超额”间寻找数学平衡，而非直接宕机。
2. **涨跌停的“抄底与砸盘”假象**：
   默认情况下跌停股无法卖出。但若未配置 `allow_buy_limit_down=False`，优化器可能因为某跌停股 Alpha 极高且价格诱人，给出买入指令。实盘中跌停板买入面临巨大不确定性。建议在约束构建时封死该动作。
3. **5000只全市场优化的性能瓶颈**：
   当 `cvxpy` 约束矩阵极大时，构建问题（而非求解）本身可能耗时十几秒，这对高频/日内多次调仓系统是不可接受的。

### 6.2 架构进阶与改进方向 (Roadmap to Institutional Grade)

如果您计划进一步迭代此项目，建议按照以下路线图进行：

#### 改进 1：全量 Barra 多因子协方差闭环 (Risk Model Upgrade)

目前 `FactorCovEstimator` 在测试脚本中尚未完全连通底层历史因子序列的截面回归计算。
**改进方案**：引入 `sklearn` 的 `OAS`（Oracle Approximating Shrinkage）估计器，对历史收益率进行 63 天半衰期的指数加权 WLS（加权最小二乘）截面回归。这将使优化器真正拥有“感知股票间隐性关联”的能力。

#### 改进 2：两阶段极速优化器 (Two-Stage Optimizer)

针对全市场规模的性能问题。
**改进方案**：

- **Stage 1 (粗筛)**：放弃协方差，仅使用 Alpha 向量和简单的上下界约束，跑一次极速的线性规划（Linear Programming），耗时 < 0.1秒，选出 Top 500 核心候选池。
- **Stage 2 (精修)**：在 500 只股票的小宇宙内，载入完整的 Barra 协方差、不对称约束、换手率软惩罚，跑完整的二次规划。耗时可控制在 1-2 秒内。

#### 改进 3：可观测性与事前风险归因分解 (Observability & Risk Attribution)

现有的 `RiskReporter` 主要做结果统计。
**改进方案**：

- **结构化日志**：引入 `structlog`，将所有融合方法、求解时间、违约偏差量以 JSON 格式输出，方便对接 ELK 或 Grafana 监控大盘。
- **风险分解 (Risk Decomposition)**：利用计算出的权重 $w$，事前计算并输出总跟踪误差 $TE^2 = w_a^T \Sigma w_a$，并进一步分解为：
  - 行业风险贡献度 (%)
  - 风格因子风险贡献度 (%)
  - 特质风险贡献度 (%)
    这样，基金经理在看到调仓指令的同时，就能明确知道今天的风险来源是暴露在了“动量因子”还是“某个特定行业”上。

#### 改进 4：动态 IC 加权信号融合

目前的 Alpha 融合往往采用固定权重。未来可引入时序模块，滚动计算每个 Domain Alpha 最近 20 天的 Rank IC（秩相关系数）或 IC_IR，使用最大化 IR 框架动态调节 `CompositeAlpha` 的组合权重。

---

## 结语

本 Portfolio Layer 工程通过优雅的领域驱动设计，成功地将复杂的金融业务逻辑转化为严谨的数学凸优化问题。它不仅具备处理真实异构 Parquet 数据的吞吐能力，其特有的软约束松弛与不对称可交易过滤机制，更是在确保量化策略“能跑通”的同时“能落地”。严格遵循本手册的数据契约与调用规范，您将能够无缝地将其嵌入到任何复杂的机构级量化流水线之中。
