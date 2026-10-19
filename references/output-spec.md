# 输出规格（16 个 Sheet）

报告为单个 `.xlsx`，默认 16 个 sheet，顺序由模板 `sheet_order` 控制，可用 `sheet_include` 裁剪。

---

## 1. Sheet 清单

| # | Sheet | 内容 | 关键列 |
|---|---|---|---|
| 1 | 总览 | 模型名、样本量、整体坏账率、渠道数、各数据集时间区间与跨度校验 | 指标 / 值（kv 表） |
| 2 | 样本统计 | 渠道维度的好/坏/灰样本数与坏账率 | channle, CntGood, CntBad, bad_rate |
| 3 | 逐月样本统计 | 上块=逐月，下块=渠道×月 | month, CntRec, CntBad, bad_rate |
| 4 | 建模样本统计 | train/test/oot 的样本量与坏账率 | data_flag, CntRec, CntBad, bad_rate |
| 5 | 建模效果统计 | 上块=数据集，下块=渠道 | ks, auc, gini, lift-1%/5%/10%/20%/尾部10%, 捕获率-10%/20% |
| 6 | 逐月模型效果 | 上块=逐月，下块=渠道×月 | month, ks, auc, bad_rate |
| 7 | 模型分10箱 | 10 等分（按样本数） | data_flag, 分箱, 好/坏样本数, 样本占比, 箱内坏率, 区间lift, 累计lift |
| 8 | 模型分20箱 | 同上，20 等分 | 同上 |
| 9 | 分箱坏账趋势 | 整体 + 各渠道的月度分箱坏账率（两级表头） | 分箱 × 月 |
| 10 | PSI热力矩阵 | 月 × 月 的分数 PSI | 月份行列，值为 PSI |
| 11 | 校准度 | 按预测概率分箱的预测 vs 实际 + gap（含 HL 检验） | data_flag, 预测均值, 实际坏率, gap |
| 12 | 特征全景 | Top 特征的 train/test/oot 统计 | feature, 缺失率, 均值, IV… |
| 13 | 特征PSI | 特征级稳定性 | feature, psi_test, psi_oot |
| 14 | 单变量分析 | 变量 × 分箱 的 train/test/oot 分布与 IV（两级表头） | variable, bin, count_distr_*, badprob_*, total_iv_*, psi_* |
| 15 | 单变量分箱按月分析 | 特征 × 分箱 × 月份 趋势（LIFT 带数据条） | variable, bin, 用户数占比 / 逾期率 / LIFT × 月 |
| 16 | 模型图表 | ROC、KS、分数分布、PSI 热力图、Lift 图 | 图片页（9 张） |

---

## 2. 降级后的 12 个 Sheet

缺少宽表 / 特征重要性 / 主动 `--no-univariate` 时，跳过 **特征全景、特征PSI、单变量分析、单变量分箱按月分析** 四项，其余 12 项完整输出。这是设计内的降级，日志会打印原因。

---

## 3. 版式规格（定版）

| 项 | 规则 |
|---|---|
| 表头 | 蓝底白字加粗（`#4472C4`），居中，自动换行，冻结 |
| 大标题 | 浅蓝底（`#D9E1F2`），13 号加粗，跨列合并 |
| 块间空行 | 同一 sheet 内相邻数据块之间空 **2** 行（`gap_rows`，compact 模板 = 1） |
| 组间空行 | 同一表内按首列分组（data_flag / 变量 / 渠道）之间空 **1** 行（`group_gap`，compact = 0） |
| 条件格式 | **不得跨越空行**（否则色阶/数据条会把空行算进刻度，这是产出验收的红线） |
| 数字格式 | 比率类 4 位小数，百分比 2 位，样本量千分位，PSI 用 General 保留原精度 |
| 美化预设 | metric=三色阶+阈值高亮；psi=绿→黄→红；bin=坏率数据条+lift 色阶+gap 三色阶；univar=IV/PSI 数据条；sample=坏账率色阶 |
| 字体 | 微软雅黑 10 号（compact = 9 号） |

---

## 4. 产出验收

```bash
python validate_output.py --report out/report.xlsx                 # 16 sheet 全量校验
python validate_output.py --report out/report.xlsx --expect-sheets 总览,建模效果统计,模型分10箱
python validate_output.py --report out/report.xlsx --json
```

检查项：sheet 齐全且不为空、条件格式未跨空行、关键列存在（ks/auc、坏率/lift、gap、psi、iv）。
退出码 `0`=通过，`1`=存在 ERROR。

> 校验依赖 `openpyxl`（只读校验用；写入引擎仍是 `xlsxwriter`，因为 openpyxl 不支持数据条/色阶扩展）。
