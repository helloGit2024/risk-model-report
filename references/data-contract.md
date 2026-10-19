# 输入数据契约

出报告前先看这份契约。**跑一次完整报告约 20~90 秒**，契约不符的典型表现是"跑完才报错"或"静默少 sheet"，所以约定：**先 `preflight_check.py`，再 `generate_report.py`**。

---

## 1. 四类输入

| 文件 | 必需 | 格式 | 用途 |
|---|---|---|---|
| 模型结果表 `--rnt` | ✅ | csv / pkl | 全部指标的主数据源（分数 + 标签 + 日期 + 数据集标识） |
| 配置 `--config` | ✅ | json | 列名映射与阈值（不传时取默认值 `y/dateback/data_flag/channle`） |
| 原始宽表 `--wide` | ⬜ | csv / pkl | 特征分析（特征全景 / 特征PSI / 单变量分析 / 单变量分箱按月） |
| 特征重要性 `--importance` | ⬜ | csv | 决定分析哪些特征、按什么顺序 |

---

## 2. 模型结果表（rnt）列要求

| 列 | 配置项 | 类型 | 取值要求 | 缺失后果 |
|---|---|---|---|---|
| 模型分数列 | `--model`（命令行传入） | 数值 | 越大代表风险越高（见 §5 方向约定）；缺失率建议 <5% | **ERROR** 报告无法生成 |
| 标签列 | `target` | 整数 | `0=好 / 1=坏`；灰样本需在 `other_label_value` 声明 | **ERROR** |
| 日期列 | `date` | 可解析为日期 | 也可直接给 `month` 列（`YYYY-MM`） | **ERROR** 逐月分析全线为空 |
| 数据集标识 | `data_flag` | 字符串 | `train / test / oot`（大小写敏感） | **ERROR** 无法做效果对比 |
| 渠道列 | `channle_name` | 字符串 | 可选；存在则自动出渠道类分析 | 渠道 sheet 为空 |

行数建议：每个数据集 ≥ 1000 行、坏样本 ≥ 100 例，否则 KS/PSI 结论不稳定。

---

## 3. 配置文件字段

```json
{
  "target": "y",
  "date": "dateback",
  "data_flag": "data_flag",
  "channle_name": "channle",
  "other_label_value": [-1],
  "other_y_lst": ["y_m1", "y_m2"],
  "eval_univariate": true,
  "bins": 10,
  "importance_top": 30,
  "psi_threshold": 0.1,
  "psi_alert_threshold": 0.25,
  "iv_threshold": 0.02,
  "ks_overfit_threshold": 0.05
}
```

兼容别名：`channle`/`channel` → `channle_name`；`psiThreshold` → `psi_threshold`；`importance_top_n` → `importance_top`。

---

## 4. 输入降级矩阵

| 场景 | 条件 | 结果 |
|---|---|---|
| 1 两者都缺 | 无 wide、无 importance | **12 个 sheet**，跳过特征全景 / 特征PSI / 单变量分析 / 单变量分箱按月；耗时降到约 1/3 |
| 2 只缺宽表 | 有 importance、无 wide | 同上（importance 无法落地到特征列） |
| 3 只缺重要性 | 有 wide、无 importance | 同上（不知道该分析哪些特征） |
| 4 主动关闭 | `--no-univariate` | 同上（只想快速看模型效果时用） |
| 5 缺 oot | `data_flag` 无 oot | 仍能出报告，但**无跨时间稳定性诊断**，PSI 只算 train↔test |
| 6 缺渠道列 | 无 `channle_name` | 渠道类 sheet 为空（样本统计 / 渠道模型效果 / 渠道月 / 分箱坏账趋势渠道块） |

降级会在日志里明确打印原因（`输入降级（…）→ 跳过 4 个特征类 sheet`），不要靠"找不到 sheet"来反推。

---

## 5. 分数方向约定（评分卡必读）

全库统一按 **分数越高 = 风险越高** 计算 lift / 捕获率 / 分箱表：
- `lift-1%` 指**最高分 1%** 的 lift；`lift-尾部10%` 指**最低分 10%**。
- **方向由数据判定，不看模型类型**。`preflight_check.py` 的 **W7** 会自动检测
  （`corr(score, y)` 符号 + 十分位坏率单调比例）：
  - **无 W7** → 分高即高风险，直接跑；
  - **报 W7** → 分高是优质客户，出报告前先做方向转换：

```python
df["risk_score"] = -df["credit_score"]      # 或 用 max - score
```

转换后再把 `--model risk_score` 传进去。方向搞反的表现：lift 全部 <1、捕获率异常低、KS 正常但分箱坏率单调递减。

> **注意：「信用分 / 评分卡」不等于方向反。** 多数风控模型分（含本 skill 自带样例、
> 以及 `lgbdoc` 这类多模型结果表）本身就是"分高 = 高风险"，**无需取反**。
> 是否取反一律以 W7 为准，不要凭模型名推断。

---

## 6. 前置校验（必跑）

```bash
cd scripts
python preflight_check.py --config cfg.json --model SCORE --rnt rnt.csv \
        --wide wide.csv --importance imp.csv        # 人类可读
python preflight_check.py --config cfg.json --model SCORE --rnt rnt.csv --json
```

退出码：`0`=可继续（可能带 WARN）；`1`=必须修（有 ERROR）。

| 码 | 含义 | 处理 |
|---|---|---|
| E1 | 文件不存在 / 配置解析失败 | 检查路径 |
| E2 | 结果表缺必需列 | 补齐分数列 / 标签列 / 日期列 / 数据集标识列 |
| E3 | 标签非 0-1 | 改标签，或在 `other_label_value` 声明灰样本取值 |
| W1 | 缺 test / oot | 确认是否真不需要过拟合 / 稳定性诊断 |
| W2 | 坏率为 0 或 1 | 数据有误，KS/AUC 无意义 |
| W3 | 坏率 <0.5% 或 >50% | 结论需谨慎，考虑采样或换口径 |
| W4 | 分数缺失率 >5% | 排查打分链路 |
| W5 | 宽表 / 重要性缺失或特征命中率低 | 会触发降级（见 §4） |
| W6 | 日期解析失败率 >5% | 逐月分析会空 |
