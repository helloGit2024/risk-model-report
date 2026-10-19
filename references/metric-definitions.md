# 指标口径（全库锁死，禁止重复实现）

任何指标只在一处实现。改口径必须改实现处，不允许在报告层"就地再算一遍"。

---

## 1. 口径表

| 指标 | 口径 | 唯一实现 |
|---|---|---|
| KS | `max(TPR − FPR)` | `riskpkg/metrics.py::RiskMetrics.calc_ks` |
| AUC | ROC 曲线下面积 | `RiskMetrics.calc_auc` |
| Gini | `2 × AUC − 1` | `RiskMetrics.calc_gini` |
| KS 切点 | 使 KS 最大的分数阈值 | `RiskMetrics.ks_cutoff` |
| 过拟合 | train→test 的 KS/AUC 绝对衰减 | `RiskMetrics.overfit_gap` |
| WOE | `ln(Bad% / Good%)`（与 scorecardpy 一致，分数越高风险越高） | `RiskMetrics.calc_iv_woe` |
| IV | `Σ(Bad% − Good%) × WOE` | `RiskMetrics.calc_iv` |
| PSI | `Σ(q − p) × ln(q / p)`，基准分箱，空箱保留并平滑 | `RiskMetrics.calc_psi` / `stability.py::StabilityMonitor`（全库唯一 PSI 入口） |
| PSI 矩阵 | 月 × 月 两两 PSI | `RiskMetrics.calc_psi_matrix` |
| Lift | 分片坏率 / 整体坏率；**head=高分=高风险，tail=低分=低风险** | `RiskMetrics.calc_lift_slice` |
| 捕获率 | 头部 top-N% 命中的坏样本 / 全部坏样本 | `RiskMetrics.calc_capture_rate` |
| 坏账率 | 坏 / (好 + 坏)，**灰样本不计入分母** | `sampler.py::DataStatistics` |
| 单调性 | 分箱坏率的单调方向判定 | `RiskMetrics.calc_monotonicity` |
| Brier | 概率校准的平方误差 | `RiskMetrics.calc_brier` |
| Hosmer-Lemeshow | 校准度检验（返回统计量与 p 值，需 scipy） | `RiskMetrics.hosmer_lemeshow` |
| 校准表 | 预测 vs 实际 + gap（按预测概率分箱） | `RiskMetrics.calibration_table` |

---

## 2. 易错点（历史踩过的坑）

| 坑 | 说明 |
|---|---|
| lift 头尾方向 | 曾把"头部累计 lift"赋给"尾部"。现在语义固定：`lift-1%`=最高分 1%；`lift-尾部10%`=最低分 10% |
| PSI 基准 | 基准分箱来自 **train**，不是全量。跨数据集比较必须统一用 train 分箱，否则 PSI 不可比 |
| 灰样本 | 坏账率分母不含灰样本；但样本统计里单列展示灰样本数 |
| 空箱平滑 | PSI 计算时空箱保留并平滑，不能丢弃，否则分母不一致 |
| Gini vs AUC | 报告同时给两者，但对外沟通口径统一用 **KS + AUC**，Gini 仅作补充 |

---

## 3. 评分卡场景

`riskpkg/scorecard.py::ScoreConverter` 提供概率 ↔ 分数的双向转换（`p_to_score` / `score_to_p`），支持 PDO / base_score / base_odds。

出评分卡报告时：
- **分数越高 = 风险越高** 的评分卡：直接把分数列传给 `--model`。
- **分数越高 = 越优质** 的评分卡（如信用分）：先取反或 `max - score` 再传入，否则 lift / 捕获率方向全反。
- 校准度 sheet 需要**概率**（不是分数）。若只有分数，先用 `score_to_p` 转成概率列再传 `--model`。
