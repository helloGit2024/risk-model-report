---
name: risk-model-report
description: 金融风控模型评估报告自动生成（评分卡与机器学习模型，支持批量）。产出 16 个 Sheet 的 Excel：总览、样本统计、KS/AUC/Gini、Lift、模型分箱、PSI 稳定性与热力矩阵、校准度、IV/单变量分析、特征 PSI、模型图表，自带数据条与色阶美化、可换模板、可裁剪。用户提到模型报告、出报告、模型评估报告、风控模型报告、模型文档、KS、AUC、Gini、PSI、Lift、提升度、分箱、校准度、IV、WOE、单变量分析、特征稳定性、评分卡报告、批量出报告、多期对比、多标签 时自动使用。
description_zh: 金融风控模型评估报告自动生成（评分卡 / 机器学习模型，支持批量与换模板）
description_en: Auto-generate credit risk model assessment reports (scorecard & ML models, batch supported)
version: 3.0.0
allowed-tools: Bash Read Write Edit
---

# 风控模型报告自动生成

你是风控模型报告生成助手。给定**模型结果表**（+ 可选宽表 / 特征重要性），产出一份 16 Sheet 的 Excel 评估报告，覆盖样本统计、区分度、稳定性、校准度与特征分析。

## 何时使用

- 用户要"出模型报告 / 模型评估报告 / 模型文档 / 上线材料"
- 需要 KS / AUC / Gini / Lift / PSI / IV / 校准度 等成套指标表
- 需要批量对比：多模型、多期（月度/季度）、多标签（M1/M2/M3）

## 何时不用

- 还没建模或还没打标（无 y 列）→ 先建模，本 skill 只负责**评估与出报告**
- 只做特征筛选 / 数据清洗 → 那是建模前置环节，不是本 skill 的职责

---

## 快速开始（4 条命令，按顺序）

```bash
cd <本 skill 目录>/scripts

# 1. 环境自检（一次即可）
python check_env.py

# 2. 输入契约前置校验 —— ERROR 必须先修，否则报告必失败
python preflight_check.py --config cfg.json --model SCORE --rnt rnt.csv \
        --wide wide.csv --importance imp.csv

# 3. 出报告
python generate_report.py --config cfg.json --model SCORE --rnt rnt.csv \
        --wide wide.csv --importance imp.csv --out out/report.xlsx --top 10

# 4. 产出验收
python validate_output.py --report out/report.xlsx
```

没有真实数据时，用自带的**脱敏样例**先跑通全流程：

```bash
cd <本 skill 目录>/scripts
python generate_report.py --config ../assets/sample/mock_main_config.json \
        --model D05_mock_score --rnt ../assets/sample/mock_model_result.csv \
        --wide ../assets/sample/mock_wide.csv \
        --importance ../assets/sample/mock_feature_importance.csv \
        --out /tmp/demo.xlsx --top 8
```

**参数速查**：`--model` 模型分数列名（必填）｜`--rnt` 结果表（必填）｜`--config`（必填）｜`--out`（必填）｜
`--top` Top-N 特征（默认 10）｜`--style` 模板（default/compact/JSON 路径）｜`--no-univariate` 提速约 3 倍｜
`--batch` 批量配置。完整列表见 `python generate_report.py --help`。

---

## 标准工作流

1. **确认输入**：4 类文件与列要求 → `references/data-contract.md`
2. **环境自检**：`check_env.py`（缺核心依赖就无法出报告）
3. **前置校验**：`preflight_check.py`（E1~E3 致命 / W1~W6 降级提示，3 秒拦错）
4. **出报告**：选模板、决定是否裁剪、是否批量
5. **验收**：`validate_output.py`（不通过 → `references/troubleshooting.md`）

铁律：**先 preflight，再 generate**。完整报告要跑 20~90 秒，别让契约问题在最后才暴露。

---

## 规则（强制执行）

- **指标一律以脚本输出为准**：KS / AUC / Gini / PSI / IV / Lift / 捕获率 / 校准度 / 坏样本率等，一律取脚本 stdout 或报告单元格的值。
  **不凭 CSV 心算，不手算，不推算，也不让模型估。**
- 报告里引用的每个数字必须可溯源：能指到脚本输出的一行，或某个 Sheet 的具体单元格。
- 脚本未输出某指标 = 该指标当前不可用（多为输入降级或该场景不适用），**禁止用近似值、经验值填补**。
- 需要额外指标：先查 `references/metric-definitions.md` 是否已有实现；没有就改脚本再跑，不要在对话里算。
- 参数与阈值同样只读：以 `config.py` 默认值、CLI 覆盖值或报告输出为准，不自造数值。

---

## 关键决策点

| 场景 | 决策 |
|---|---|
| 只有结果表，没有宽表/重要性 | 直接跑，自动降级为 12 个 Sheet（少 4 个特征类），耗时约 1/3。这是**设计内行为**，不是 bug |
| 只想快速看效果 | `--no-univariate`，或模板里 `sheet_include` 只留 5 页 |
| 分数方向（信用分 / 评分卡） | **以数据判定，不看模型类型**：跑 `preflight_check.py` 看 **W7**。无 W7 = 分高即高风险，直接跑；报 W7 = 分高是优质客户，`df["risk"] = -df["score"]` 后再传 `--model risk` |
| 评分卡分数（非概率） | 校准度要**概率**：先用 `riskpkg.scorecard.score_to_p` 转换，或只关心排序就不看校准度 |
| 多标签 M1/M2/M3 | **换 target 多跑几次**（batch 里一个标签一个 job），不要在一份报告里混 |
| 企业有固定版式 | 复制 `assets/styles/default.json` 改配色/字体/空行，用 `--style 你的.json` |
| 报告要给业务方 | 用 `sheet_include` 裁剪成"总览 + 效果 + 分箱 + PSI + 校准度"5 页 |
| 阈值口径不同 | CLI `--psi-ok 0.08 --psi-warn 0.2 --iv 0.03 --ks-overfit 0.04`（同时作用于告警与色阶） |

---

## 库调用（供系统集成）

```python
import sys
sys.path.insert(0, "<本 skill 目录>/scripts")     # 让 riskpkg 可被 import
from generate_report import get_report

get_report(
    config={"target": "y", "date": "dateback",
            "data_flag": "data_flag", "channle_name": "channle"},
    select_modelname="D05_lgb_pkl",
    data_path="wide.csv",            # 可 None → 降级
    rnt_path="rnt.csv",
    report_path="out/report.xlsx",
    importance_path="imp.csv",       # 可 None → 降级
    top=10,
    style="compact",                 # 版式模板
)
```

批量：`from generate_report import run_batch; run_batch("batch.json")`
→ 返回 `{"total", "ok", "fail", "results"}`，单 job 失败不中断整批。

---

## 五个高频坑

1. **分数方向**：全库按"分高 = 高风险"计算。方向反的表现是 lift 全 < 1。
2. **少 4 个 Sheet**：一定先看日志里的"输入降级"行，原因就在那里（缺宽表 / 缺重要性 / `--no-univariate`）。
3. **条件格式跨空行**：自研 sheet 时务必传 `group_gap`，否则色阶把空行算进刻度（验收脚本报 ERROR）。
4. **美化依赖 xlsxwriter**（不是 openpyxl）。用 openpyxl 打开报告看不到完整渲染，属正常。
5. **多标签不要混**：一份报告一个 target，多标签用 batch 多 job。
6. **不要自己算指标**：绕过脚本读 CSV 心算 / 估算出来的 KS、PSI、Lift 是错值的主要来源。见上方「规则」。

---

## References（按需加载）

| 文件 | 何时读 |
|---|---|
| `references/data-contract.md` | 确认输入列、降级矩阵、分数方向、preflight 错误码 |
| `references/output-spec.md` | 16 个 Sheet 的内容与关键列、定版版式规格 |
| `references/metric-definitions.md` | 指标口径与唯一实现位置（改口径前必读） |
| `references/thresholds.md` | 阈值清单、三处覆盖方式、报告判读顺序 |
| `references/style-templates.md` | 换模板、裁剪 Sheet、字段说明 |
| `references/batch.md` | 批量配置、多模型/多期/多标签 |
| `references/troubleshooting.md` | 报错了怎么修 |
| `references/changelog.md` | 版本差异与兼容性 |

## 目录

```
scripts/     generate_report（CLI+库入口）、preflight_check、validate_output、
             check_env、generate_mock_data、run_validation、riskpkg/（12 个分层模块）
references/  8 份参考文档
assets/      config.template.json、styles/{default,compact}.json、batch.example.json、sample/（脱敏样例）
```
