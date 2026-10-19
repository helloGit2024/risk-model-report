# -*- coding: utf-8 -*-
"""
riskpkg —— 风控模型报告能力包
================================================
分层架构（自底向上，单向依赖，杜绝循环引用）：

    config     配置层        ReportConfig
    style      样式层        StyleConfig（可换模板：配色/字体/空行/阈值/Sheet 编排）
    metrics    原子指标层    RiskMetrics（KS/AUC/Gini/PSI/Lift/IV/WOE/Calibration）
    binning    分箱层        ScorecardBinner
    stability  稳定性层      StabilityMonitor（全库唯一 PSI 入口）
    evaluation 评估层        ModelEvaluator（分组指标 + Lift 表）
    features   特征层        FeatureAnalyzer（特征全景 + 单变量分析）
    scorecard  评分卡层      ScoreConverter
    sampler    样本统计层    DataStatistics
    splitter   数据划分层    DataSplitter
    charts     可视化层      ChartVisualizer
    excel      报告层        ExcelStyler / ReportBuilder

对外入口仍是 `runReport.get_report`，签名保持不变。
"""

from .config import ReportConfig
from .style import StyleConfig, load_style, dump_style
from .metrics import RiskMetrics, safe_div
from .binning import ScorecardBinner
from .stability import StabilityMonitor
from .evaluation import ModelEvaluator
from .features import FeatureAnalyzer
from .scorecard import ScoreConverter, p_to_score, score_to_p
from .sampler import DataStatistics
from .splitter import DataSplitter
from .charts import ChartVisualizer
from .excel import ExcelStyler, ReportBuilder

__version__ = "2.0.0"

__all__ = [
    "ReportConfig",
    "StyleConfig",
    "load_style",
    "dump_style",
    "RiskMetrics",
    "safe_div",
    "ScorecardBinner",
    "StabilityMonitor",
    "ModelEvaluator",
    "FeatureAnalyzer",
    "ScoreConverter",
    "p_to_score",
    "score_to_p",
    "DataStatistics",
    "DataSplitter",
    "ChartVisualizer",
    "ExcelStyler",
    "ReportBuilder",
]
