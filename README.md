# risk-model-report

金融风控模型评估报告自动生成工具。给定模型结果表（+ 可选宽表 / 特征重要性），产出一份 16 Sheet 的 Excel 评估报告，覆盖样本统计、区分度（KS / AUC / Gini）、Lift、分箱、PSI 稳定性与热力矩阵、校准度、IV / 单变量分析、特征 PSI、模型图表。

支持评分卡与机器学习模型，支持批量（多模型、多期、多标签）与自定义模板。

## 技术栈

- **语言**：Python 3.8+
- **核心依赖**：pandas、numpy、scikit-learn、xlsxwriter、matplotlib、seaborn、pydantic
- **可选依赖**：scorecardpy、statsmodels、shap、lightgbm、scipy

## 目录结构

```
scripts/          # 入口与核心模块
  generate_report.py    # CLI + 库入口：出报告
  preflight_check.py    # 输入契约前置校验（E1~E3 致命 / W1~W7 降级提示）
  validate_output.py    # 产出验收
  check_env.py          # 环境自检
  run_validation.py     # 全流程回归验证
  generate_mock_data.py # 生成脱敏样例数据
  riskpkg/              # 12 个分层模块（binning / charts / config / evaluation / excel / features / metrics / sampler / scorecard / splitter / stability / style）
references/       # 8 份参考文档（数据契约、产出规格、指标定义、阈值、模板、批量、排错、变更日志）
assets/
  config.template.json  # 配置模板
  styles/               # 版式模板（default / compact）
  batch.example.json    # 批量配置示例
  sample/               # 脱敏样例数据 + 示例产出
SKILL.md          # Skill 使用说明（含何时用 / 何时不用、快速开始、关键决策点）
requirements.txt  # 依赖清单
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 环境自检（一次即可）
python scripts/check_env.py

# 3. 用自带脱敏样例跑通全流程
python scripts/generate_report.py \
    --config assets/sample/mock_main_config.json \
    --model D05_mock_score \
    --rnt assets/sample/mock_model_result.csv \
    --wide assets/sample/mock_wide.csv \
    --importance assets/sample/mock_feature_importance.csv \
    --out /tmp/demo.xlsx --top 8

# 4. 产出验收
python scripts/validate_output.py --report /tmp/demo.xlsx
```

真实数据流程：先 `preflight_check.py` 校验输入契约，再 `generate_report.py` 出报告，最后 `validate_output.py` 验收。详见 [SKILL.md](SKILL.md)。

## 环境变量

无。所有配置通过 `--config` JSON 文件与 CLI 参数传入。

## 参考文档

按需加载，完整清单见 [SKILL.md](SKILL.md#references按需加载)。
