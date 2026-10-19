# -*- coding: utf-8 -*-
"""
风控模型报告 · 主编排入口（CLI + 库调用）
================================================================================
两种用法（企业级交付要求两者都留）
------------------------------------------------------------------
1. CLI（Agent / 调度系统首选，最稳）
   python generate_report.py --config cfg.json --model SCORE --rnt rnt.csv \
          --wide wide.csv --importance imp.csv --out out/report.xlsx --top 10
   python generate_report.py --batch batch.json          # 批量（多模型 / 多期 / 多标签）

2. 库调用（供系统集成）
   from generate_report import get_report
   get_report(config, select_modelname, data_path, rnt_path, report_path, importance_path)

对外唯一入口 `get_report`，签名向后兼容（新增参数均带默认值）。

报告 Sheet 清单（16 个，可通过模板 sheet_order / sheet_include 重排或裁剪）
------------------------------------------------------------------
总览 / 样本统计 / 逐月样本统计 / 建模样本统计 / 建模效果统计 / 逐月模型效果 /
模型分10箱 / 模型分20箱 / 分箱坏账趋势 / PSI热力矩阵 / 校准度 / 特征全景 /
特征PSI / 单变量分析 / 单变量分箱按月分析 / 模型图表

历史缺陷修复（勿回退）
------------------------------------------------------------------
P0-1  补全自定义模块 import（原版未 import Evaluation_indicators/Sub_box/…）
P0-3  `univar_analysis_df` 恢复（原在 Utils.py，因 import 被注释而断链）
P0-4  修正 zt_metrics/month_metrics/qudao_metrics 的参数错位与缺失
P0-5  GetKS 支持 ylim（原版调用传了 ylim 但函数无此参数）
P0-8  to_excel 必须指定 sheet_name（原版所有表堆进 Sheet1、标题写空 sheet）
P0-9  从日期派生 month 列（原版全流程依赖 month 但从不生成）
P1-3  lift 头尾方向修正（原版把头部累计 lift 赋给"尾部"）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# ---- 自举：保证从任意工作目录运行都能 import riskpkg ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---- 依赖注入：全部能力来自 riskpkg 分层模块 ----
from riskpkg.config import ReportConfig  # noqa: F401  (对外保留同名导出)
from riskpkg.style import StyleConfig, load_style, dump_style  # noqa: F401
from riskpkg.evaluation import ModelEvaluator
from riskpkg.stability import StabilityMonitor
from riskpkg.features import FeatureAnalyzer
from riskpkg.sampler import DataStatistics
from riskpkg.splitter import DataSplitter
from riskpkg.charts import ChartVisualizer
from riskpkg.excel import ReportBuilder
from riskpkg.metrics import RiskMetrics

logger = logging.getLogger(__name__)

# 数据集标准顺序
DEFAULT_DATA_FLAGS = ["train", "test", "oot"]

# Sheet 默认编排顺序（模板可覆盖）
DEFAULT_SHEET_ORDER: List[str] = [
    "总览", "样本统计", "逐月样本统计", "建模样本统计", "建模效果统计",
    "逐月模型效果", "模型分10箱", "模型分20箱", "分箱坏账趋势", "PSI热力矩阵",
    "校准度", "特征全景", "特征PSI", "单变量分析", "单变量分箱按月分析", "模型图表",
]

# CLI 阈值参数 -> (config 字段, style.thresholds 键) 映射
THRESHOLD_MAP: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "psi_ok": ("psi_threshold", "psi_ok"),
    "psi_warn": ("psi_alert_threshold", "psi_warn"),
    "iv": ("iv_threshold", "iv"),
    "ks_ok": (None, "ks_ok"),
    "ks_warn": (None, "ks_warn"),
    "ks_overfit": ("ks_overfit_threshold", "ks_overfit"),
}


# =============================================================================
# 日志（幂等：批量场景下不重复挂 handler）
# =============================================================================
def _setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s - %(message)s",
        )
    root.setLevel(level)


# =============================================================================
# 数据准备
# =============================================================================
def _prepare_data(rnt_path: str, config: ReportConfig) -> pd.DataFrame:
    """
    读取模型结果表并做标准化处理。

    修复：原版从不生成 `month` 列，但 month_metrics / get_psiheat / score_fb
    等全流程都依赖它，导致逐月分析全线崩溃。
    """
    if rnt_path.endswith(".pkl"):
        df = pd.read_pickle(rnt_path)
    else:
        df = pd.read_csv(rnt_path)

    # 日期列改名（保留原列）
    date_col = config.date
    if date_col in df.columns:
        df = df.rename(columns={date_col: "dateBack"})
        # 派生 month（缺失时自动补全）
        if "month" not in df.columns:
            df = DataSplitter.ensure_month(df, "dateBack")
    elif "month" not in df.columns:
        logger.warning("数据中既无日期列 %s 也无 month 列，逐月分析将不可用", date_col)

    # 标签列标准化
    if config.target in df.columns and config.target != "label":
        df["label"] = df[config.target]

    logger.info("数据准备完成：%d 行 × %d 列", df.shape[0], df.shape[1])
    return df


# =============================================================================
# 计算：所有报告表
# =============================================================================
def _compute_all_tables(df: pd.DataFrame, config: ReportConfig, select_modelname: str,
                        importance_path: Optional[str], top: int,
                        eval_univariate: bool, data_flag: List[str],
                        data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """计算报告所需的全部数据表。"""
    target = config.target
    channel = config.channle_name
    other_label_value = getattr(config, "other_label_value", None) or []

    ev = ModelEvaluator(y_col=target, date_col="dateBack", channel_col=channel,
                        data_flag_col=config.data_flag, month_col="month")
    st = StabilityMonitor(psi_threshold=getattr(config, "psi_threshold", 0.1),
                          psi_alert_threshold=getattr(config, "psi_alert_threshold", 0.25))
    fa = FeatureAnalyzer(y_col=target, data_flag_col=config.data_flag,
                         psi_threshold=st.psi_threshold, psi_alert=st.psi_alert_threshold)
    ds = DataStatistics(df, target, "dateBack", channel,
                        other_label_value=list(other_label_value),
                        model_name=select_modelname)

    tables: Dict[str, Any] = {}

    # 1. 样本统计
    tables["sample"] = {
        "dataset": ds.dataset_sample(),
        "month": ds.month_sample(),
        "channel": ds.channel_sample() if channel and channel in df.columns else pd.DataFrame(),
        "channel_month": ds.channel_month_sample() if channel and channel in df.columns else pd.DataFrame(),
        "overview": ds.overview(),
    }

    # 2. 建模效果（含过拟合诊断）
    tables["model_metrics"] = {
        "dataset": ev.dataset_summary(df, select_modelname),
        "month": ev.evaluate_by_month(df, select_modelname),
        "channel": ev.evaluate_by_channel(df, select_modelname),
        "channel_month": ev.evaluate_by_channel_month(df, select_modelname),
        "month_rate": ev.month_rate(df),
    }

    # 3. 分箱表（10 箱 / 20 箱）
    tables["bin_data"] = {
        "10": ev.lift_table(df, select_modelname, bins=10, data_flag_lst=data_flag),
        "20": ev.lift_table(df, select_modelname, bins=20, data_flag_lst=data_flag),
    }

    # 4. 稳定性
    tables["stability"] = {
        "month_psi": st.score_psi_by_month(df, select_modelname, compare_train=True,
                                           data_flag_col=config.data_flag),
        "psi_heat": st.psi_heat_matrix(df, select_modelname),
        "dataset_psi": st.score_psi_by_dataset(df, select_modelname,
                                               data_flag_col=config.data_flag),
        "trend": st.bin_trend_blocks(df, select_modelname, target, bins=10,
                                     channel_col=channel),
    }

    # 5. 校准度
    cal_rows = []
    for flag in [f for f in data_flag if f in set(df[config.data_flag].dropna())]:
        sub = df[df[config.data_flag] == flag]
        c = RiskMetrics.calibration_table(sub[target], sub[select_modelname], bins=10)
        if not c.empty:
            c.insert(0, "data_flag", flag)
            cal_rows.append(c)
    tables["calibration"] = pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()

    # 6. 特征分析
    importance = None
    if importance_path and os.path.exists(importance_path):
        try:
            importance = FeatureAnalyzer.load_importance(importance_path)
        except Exception as exc:
            logger.warning("特征重要性读取失败：%s", exc)

    if importance is not None and data is not None:
        feats = [f for f in importance["feature"].tolist() if f in data.columns]
        flags = set(df[config.data_flag].dropna()) if config.data_flag in df.columns else set()
        tr = data[data[config.data_flag] == "train"] if "train" in flags else data
        te = data[data[config.data_flag] == "test"] if "test" in flags else None
        oo = data[data[config.data_flag] == "oot"] if "oot" in flags else None

        # 特征全景（train/test/oot 三套）
        try:
            tables["feature_summary"] = fa.build_summary(
                tr, te if te is not None else pd.DataFrame(), oo,
                feats[: max(top * 3, 30)], importance=importance, bins=10)
        except Exception as exc:
            logger.warning("特征全景生成失败：%s", exc)
            tables["feature_summary"] = pd.DataFrame()

        # 特征 PSI
        try:
            tables["feature_psi"] = st.feature_psi_multi(data, feats[: max(top * 3, 30)],
                                                         data_flag_col=config.data_flag)
        except Exception as exc:
            logger.warning("特征 PSI 生成失败：%s", exc)
            tables["feature_psi"] = pd.DataFrame()

        # 单变量分析（train 分箱 → test/oot 复用边界）
        if eval_univariate:
            try:
                tables["univar"] = fa.univar_analysis_df(
                    data, importance, top_n=top, method="tree", max_bin=5)
            except Exception as exc:
                logger.warning("单变量分析失败：%s", exc)
                tables["univar"] = pd.DataFrame()

            # 单变量分箱按月分析（定版新增：特征 × 分箱 × 月份 趋势）
            try:
                tables["univar_month"] = fa.univar_bin_month_table(
                    data, importance, top_n=top, max_bin=5)
            except Exception as exc:
                logger.warning("单变量分箱按月分析失败：%s", exc)
                tables["univar_month"] = pd.DataFrame()
        else:
            tables["univar"] = pd.DataFrame()
            tables["univar_month"] = pd.DataFrame()
    else:
        tables["feature_summary"] = pd.DataFrame()
        tables["feature_psi"] = pd.DataFrame()
        tables["univar"] = pd.DataFrame()
        tables["univar_month"] = pd.DataFrame()

    return tables


# =============================================================================
# 绘图
# =============================================================================
def _draw_all_charts(df: pd.DataFrame, config: ReportConfig, select_modelname: str,
                     img_dir: str, psi_heat: Optional[pd.DataFrame],
                     lift_tables: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
    """生成报告图表，返回图片路径列表。"""
    viz = ChartVisualizer()
    os.makedirs(img_dir, exist_ok=True)
    paths: List[str] = []

    flags = [f for f in DEFAULT_DATA_FLAGS if f in set(df[config.data_flag].dropna())]
    for flag in flags:
        sub = df[df[config.data_flag] == flag]
        sub = sub[sub[config.target].isin([0, 1]) & sub[select_modelname].notna()]
        if sub.empty:
            continue
        p = viz.plot_roc(sub[config.target], sub[select_modelname],
                         name=f"{select_modelname}_{flag}", path=img_dir)
        if p:
            paths.append(p)
        # ylim 参数现已支持（原版调用传 ylim 但函数无此参数 → TypeError）
        p = viz.plot_ks(sub[config.target], sub[select_modelname],
                        name=f"{select_modelname}_{flag}", path=img_dir, ylim=[0.0, 0.6])
        if p:
            paths.append(p)

    # 分数分布（按月）
    if "month" in df.columns:
        p = viz.plot_score_dist(df, select_modelname, hue="month",
                                name=select_modelname, path=img_dir)
        if p:
            paths.append(p)

    # PSI 热力图
    if psi_heat is not None and not psi_heat.empty:
        p = viz.plot_psi_heat(psi_heat, name=select_modelname, path=img_dir)
        if p:
            paths.append(p)

    # Lift 图
    if "10" in lift_tables and "train" in lift_tables["10"]:
        p = viz.plot_lift(lift_tables["10"]["train"], name=select_modelname, path=img_dir)
        if p:
            paths.append(p)

    return paths


# =============================================================================
# Sheet 编排（注册表式：支持模板重排 / 裁剪）
# =============================================================================
def _sheet_specs(tables: Dict[str, Any], select_modelname: str, images: List[str],
                 style: StyleConfig) -> List[Tuple[str, Callable[[ReportBuilder], None]]]:
    """
    把每个 sheet 的写入逻辑登记为 (sheet名, 写入函数)。

    拆成注册表的目的：模板可声明 `sheet_order`（重排）与 `sheet_include`（裁剪），
    企业可按需只出"样本 + 效果 + 稳定性"的精简版报告。
    """
    sample = tables["sample"]
    mm = tables["model_metrics"]
    bins_t = tables["bin_data"]
    stab = tables["stability"]
    gg = style.group_gap  # 分组间空行（定版=1，compact=0）

    specs: List[Tuple[str, Callable[[ReportBuilder], None]]] = []

    # 1. 总览（含渠道数 / 模型名称 / 各数据集时间区间 / 跨度校验）
    def _overview(b: ReportBuilder) -> None:
        if not sample["overview"].empty:
            b.add_sheet("总览", sample["overview"], preset="none", index=True,
                        pct_index_values=["整体坏账率"],
                        number_formats={"值": "General"})
    specs.append(("总览", _overview))

    # 2. 样本统计（渠道维度）
    def _sample_channel(b: ReportBuilder) -> None:
        if not sample["channel"].empty:
            b.add_sheet("样本统计", sample["channel"], title="渠道样本统计",
                        preset="sample", index=True)
    specs.append(("样本统计", _sample_channel))

    # 3. 逐月样本统计（逐月 + 渠道月，上下两块）
    def _sample_month(b: ReportBuilder) -> None:
        blocks = []
        if not sample["month"].empty:
            blocks.append({"title": "逐月样本统计", "df": sample["month"],
                           "preset": "sample", "index": True})
        if not sample["channel_month"].empty:
            blocks.append({"title": "渠道月样本统计", "df": sample["channel_month"],
                           "preset": "sample", "index": True, "group_gap": gg})
        if blocks:
            b.add_multi_block_sheet("逐月样本统计", blocks, freeze=True)
    specs.append(("逐月样本统计", _sample_month))

    # 4. 建模样本统计
    def _sample_dataset(b: ReportBuilder) -> None:
        if not sample["dataset"].empty:
            b.add_sheet("建模样本统计", sample["dataset"], preset="sample", index=True)
    specs.append(("建模样本统计", _sample_dataset))

    # 5. 建模效果统计（数据集 + 渠道）
    def _effect(b: ReportBuilder) -> None:
        blocks = []
        if not mm["dataset"].empty:
            blocks.append({"title": "建模效果统计", "df": mm["dataset"],
                           "preset": "metric", "index": True})
        if not mm["channel"].empty:
            blocks.append({"title": "渠道模型效果", "df": mm["channel"],
                           "preset": "metric", "index": True})
        if blocks:
            b.add_multi_block_sheet("建模效果统计", blocks, freeze=True)
    specs.append(("建模效果统计", _effect))

    # 6. 逐月模型效果（逐月 + 渠道月）
    def _effect_month(b: ReportBuilder) -> None:
        blocks = []
        if not mm["month"].empty:
            blocks.append({"title": "逐月模型效果", "df": mm["month"],
                           "preset": "metric", "index": True})
        if not mm["channel_month"].empty:
            blocks.append({"title": "渠道月模型效果", "df": mm["channel_month"],
                           "preset": "metric", "index": True, "group_gap": gg})
        if blocks:
            b.add_multi_block_sheet("逐月模型效果", blocks, freeze=True)
    specs.append(("逐月模型效果", _effect_month))

    # 7/8. 模型分 N 箱（data_flag 列纵向合并）
    def _make_bin_sheet(bins_key: str) -> Callable[[ReportBuilder], None]:
        def _fn(b: ReportBuilder) -> None:
            per_flag = bins_t.get(bins_key, {})
            concat = []
            for flag, t in per_flag.items():
                if t is None or t.empty:
                    continue
                t = t.copy()
                t.insert(0, "data_flag", flag)
                concat.append(t)
            if concat:
                b.add_sheet(f"模型分{bins_key}箱", pd.concat(concat, ignore_index=True),
                            preset="bin", merge_col="data_flag")
        return _fn
    specs.append(("模型分10箱", _make_bin_sheet("10")))
    specs.append(("模型分20箱", _make_bin_sheet("20")))

    # 9. 分箱坏账趋势（整体 + 各渠道，两级表头）
    def _trend(b: ReportBuilder) -> None:
        trend_blocks = stab.get("trend") or []
        if trend_blocks:
            b.add_multi_block_sheet("分箱坏账趋势", trend_blocks)
    specs.append(("分箱坏账趋势", _trend))

    # 10. PSI 热力矩阵
    def _heat(b: ReportBuilder) -> None:
        if not stab["psi_heat"].empty:
            # 期望版：值用 General 保留原精度，整块共用一个 PSI 三色阶刻度
            b.add_sheet("PSI热力矩阵", stab["psi_heat"], preset="heat", index=True,
                        number_formats={c: "General" for c in stab["psi_heat"].columns})
    specs.append(("PSI热力矩阵", _heat))

    # 11. 校准度（按 data_flag 分组空 gg 行）
    def _calib(b: ReportBuilder) -> None:
        cal = tables.get("calibration", pd.DataFrame())
        if not cal.empty:
            b.add_sheet("校准度", cal, preset="bin", group_gap=gg)
    specs.append(("校准度", _calib))

    # 12. 特征全景
    def _feat_summary(b: ReportBuilder) -> None:
        fs = tables.get("feature_summary", pd.DataFrame())
        if not fs.empty:
            b.add_sheet("特征全景", fs, preset="univar")
    specs.append(("特征全景", _feat_summary))

    # 13. 特征 PSI
    def _feat_psi(b: ReportBuilder) -> None:
        fp = tables.get("feature_psi", pd.DataFrame())
        if not fp.empty:
            b.add_sheet("特征PSI", fp, preset="psi")
    specs.append(("特征PSI", _feat_psi))

    # 14. 单变量分析（两级表头：train/test/oot/psi 分组）
    def _univar(b: ReportBuilder) -> None:
        uv = tables.get("univar", pd.DataFrame())
        if not uv.empty:
            b.add_sheet("单变量分析", FeatureAnalyzer.univar_multi_header(uv),
                        header_rows=2, preset="univar", index=False, group_gap=gg)
    specs.append(("单变量分析", _univar))

    # 15. 单变量分箱按月分析（LIFT 块加数据条）
    def _univar_month(b: ReportBuilder) -> None:
        uvm = tables.get("univar_month", pd.DataFrame())
        if not uvm.empty:
            b.add_sheet("单变量分箱按月分析", uvm, header_rows=2, preset="none",
                        index=False, group_gap=gg, bar_blocks=("LIFT",))
    specs.append(("单变量分箱按月分析", _univar_month))

    # 16. 模型图表
    def _charts(b: ReportBuilder) -> None:
        if images:
            b.add_image_sheet("模型图表", images)
    specs.append(("模型图表", _charts))

    return specs


def _build_excel_sheets(builder: ReportBuilder, tables: Dict[str, Any],
                        select_modelname: str, images: List[str],
                        style: Optional[StyleConfig] = None) -> List[str]:
    """
    按模板编排写入全部 sheet 并统一套用美化预设。

    修复：原版 `to_excel(writer, startrow=1)` 未指定 sheet_name，
    导致所有表堆叠进默认 Sheet1，而标题写在空的手动 sheet 上。
    """
    style = style or builder.style
    specs = _sheet_specs(tables, select_modelname, images, style)

    order = style.sheet_order or DEFAULT_SHEET_ORDER
    include = style.sheet_include

    def _rank(item: Tuple[str, Any]) -> int:
        return order.index(item[0]) if item[0] in order else len(order)

    specs = sorted(specs, key=_rank)
    if include:
        specs = [s for s in specs if s[0] in include]

    written: List[str] = []
    for name, fn in specs:
        try:
            fn(builder)
            written.append(name)
        except Exception as exc:
            # 单个 sheet 失败不影响整份报告（企业级：宁可缺页不可全盘失败）
            logger.warning("sheet[%s] 写入失败，已跳过：%s", name, exc)
    return written


# =============================================================================
# 主入口（库调用）
# =============================================================================
def get_report(config: Dict[str, Any], select_modelname: str, data_path: str,
               rnt_path: str, report_path: str, importance_path: str,
               other_y_lst: List[str] = [], is_zt_score: str = None,
               eval_univariate: bool = True,
               data_flag: List[str] = DEFAULT_DATA_FLAGS, top: int = 10,
               style: Optional[Any] = None,
               img_dir: Optional[str] = None) -> str:
    """
    自动化建模报告主函数（一站式：样本统计 + 模型效果 + 分箱 + PSI + 特征分析 + Excel）。

    Parameters
    ----------
    config           : 配置字典（target/date/data_flag/channle_name/阈值 等）
    select_modelname : 模型分数列名
    data_path        : 原始宽表路径（特征分析、单变量分析用）
    rnt_path         : 模型结果表路径（含分数、标签、时间、数据集标识）
    report_path      : Excel 报告输出路径
    importance_path  : 特征重要性文件路径
    other_y_lst      : 其他业务标签（多标签场景）
    is_zt_score      : 是否为征信分数
    eval_univariate  : 是否开启单变量分析
    data_flag        : 数据集划分
    top              : 展示 Top-N 重要特征
    style            : 版式模板（"default"/"compact"/JSON 路径/StyleConfig）
    img_dir          : 图表输出目录（默认 = 报告同级 {model}_imgs）

    Returns
    -------
    str: 报告文件路径
    """
    _setup_logging()
    run_id = str(uuid.uuid4())[:8]
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    style_obj = load_style(style)

    logging.info("=" * 80)
    logging.info("启动风控模型报告生成 | RUN_ID=%s | %s | 模板=%s",
                 run_id, run_time, style_obj.name)
    logging.info("模型分数列=%s | 目标标签=%s", select_modelname,
                 (config or {}).get("target"))
    logging.info("=" * 80)

    try:
        # ---- 1. 配置校验（修复：原版 ReportConfig(**config, other_y_lst=...) 键冲突）----
        merged = dict(config or {})
        merged.setdefault("other_y_lst", list(other_y_lst or []))
        merged.setdefault("eval_univariate", eval_univariate)
        if is_zt_score is not None:
            merged.setdefault("is_zt_score", is_zt_score)
        cfg = ReportConfig.from_dict(merged)
        logger.info("步骤1/5：参数校验通过")

        # ---- 2. 数据准备 ----
        df = _prepare_data(rnt_path, cfg)
        data = None
        if eval_univariate and data_path and os.path.exists(data_path):
            try:
                data = pd.read_pickle(data_path) if data_path.endswith(".pkl") else pd.read_csv(data_path)
                # 宽表同样需要 month 列（单变量分箱按月分析依赖）
                if "month" not in data.columns and cfg.date in data.columns:
                    data = DataSplitter.ensure_month(data, cfg.date)
                logger.info("宽表加载完成：%s", data.shape)
            except Exception as exc:
                logger.warning("宽表加载失败，跳过特征分析：%s", exc)
        logger.info("步骤2/5：数据准备完成")

        # ---- 3. 计算 ----
        tables = _compute_all_tables(df, cfg, select_modelname, importance_path,
                                     top, eval_univariate, list(data_flag), data)
        logger.info("步骤3/5：指标计算完成")

        # 降级提示：宽表 / 特征重要性 / 单变量开关，三者缺一即跳过 4 个特征类 sheet。
        _reasons: List[str] = []
        if not eval_univariate:
            _reasons.append("eval_univariate=False")
        if data is None:
            _reasons.append(f"宽表缺失或加载失败(data_path={data_path!r})")
        if not importance_path or not os.path.exists(importance_path):
            _reasons.append(f"特征重要性缺失或加载失败(importance_path={importance_path!r})")
        if _reasons:
            logger.info(
                "输入降级（%s）→ 跳过 4 个特征类 sheet："
                "特征全景 / 特征PSI / 单变量分析 / 单变量分箱按月分析",
                "；".join(_reasons),
            )

        # ---- 4. 绘图 ----
        if img_dir is None:
            img_dir = os.path.join(os.path.dirname(os.path.abspath(report_path)),
                                   f"{select_modelname}_imgs")
        images = _draw_all_charts(df, cfg, select_modelname, img_dir,
                                  tables["stability"]["psi_heat"], tables["bin_data"])
        logger.info("步骤4/5：图表生成完成（%d 张）", len(images))

        # ---- 5. 写 Excel ----
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with ReportBuilder(report_path, style=style_obj) as builder:
            _build_excel_sheets(builder, tables, select_modelname, images, style_obj)
        logger.info("步骤5/5：Excel 写入完成 → %s", report_path)

        logging.info("报告生成成功 | RUN_ID=%s | sheets=%d | %s",
                     run_id, len(builder._sheets), report_path)
        return report_path

    except Exception as exc:
        logger.error("报告生成失败：%s", exc, exc_info=True)
        raise


# =============================================================================
# 批量：多模型 / 多期 / 多标签（多标签 = 同一份数据换 target 多跑几次）
# =============================================================================
def _load_job_config(defaults: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    """合并 defaults 与 job 的配置（支持路径 / dict / 覆盖字段）。"""
    base: Dict[str, Any] = {}
    cfg_spec = job.get("config", defaults.get("config"))
    if isinstance(cfg_spec, dict):
        base.update(cfg_spec)
    elif cfg_spec:
        with open(cfg_spec, "r", encoding="utf-8") as f:
            base.update(json.load(f))
    overrides = {**defaults.get("config_overrides", {}), **job.get("config_overrides", {})}
    base.update(overrides)
    # 多标签：job.target 直接覆盖标签列
    if job.get("target"):
        base["target"] = job["target"]
    return base


def run_batch(batch_path: str) -> Dict[str, Any]:
    """
    批量出报告。

    batch.json 结构（job 覆盖 defaults）
    ------------------------------------------------------------------
    {
      "defaults": {"config": "cfg.json", "style": "default", "top": 10,
                   "out_dir": "out", "eval_univariate": true},
      "jobs": [
        {"name": "D05_lgb", "model": "D05_lgb_pkl", "rnt": "rnt.csv",
         "wide": "wide.csv", "importance": "imp.csv"},
        {"name": "D05_lgb_M1", "model": "D05_lgb_pkl", "rnt": "rnt.csv",
         "target": "y_m1", "out": "out/D05_M1.xlsx"}
      ]
    }
    """
    _setup_logging()
    with open(batch_path, "r", encoding="utf-8") as f:
        batch = json.load(f)
    defaults = batch.get("defaults", {}) or {}
    jobs = batch.get("jobs", []) or []
    if not jobs:
        raise ValueError(f"批量配置中没有 jobs：{batch_path}")

    out_dir = defaults.get("out_dir", "out")
    results: List[Dict[str, Any]] = []
    logging.info("批量开始：共 %d 个任务", len(jobs))

    for i, job in enumerate(jobs, 1):
        name = job.get("name") or f"job{i}"
        out = job.get("out") or os.path.join(out_dir, f"{name}.xlsx")
        try:
            cfg = _load_job_config(defaults, job)
            get_report(
                config=cfg,
                select_modelname=job.get("model") or defaults.get("model"),
                data_path=job.get("wide", defaults.get("wide")),
                rnt_path=job.get("rnt", defaults.get("rnt")),
                report_path=out,
                importance_path=job.get("importance", defaults.get("importance")),
                other_y_lst=job.get("other_y_lst", []),
                is_zt_score=job.get("is_zt_score"),
                eval_univariate=job.get("eval_univariate",
                                        defaults.get("eval_univariate", True)),
                data_flag=job.get("data_flag", DEFAULT_DATA_FLAGS),
                top=job.get("top", defaults.get("top", 10)),
                style=job.get("style", defaults.get("style")),
                img_dir=job.get("img_dir"),
            )
            results.append({"name": name, "status": "ok", "out": out})
            logging.info("批量进度 %d/%d：%s 成功", i, len(jobs), name)
        except Exception as exc:
            results.append({"name": name, "status": "fail", "error": str(exc)})
            logging.error("批量进度 %d/%d：%s 失败 - %s", i, len(jobs), name, exc)

    ok = sum(1 for r in results if r["status"] == "ok")
    summary = {"total": len(jobs), "ok": ok, "fail": len(jobs) - ok, "results": results}
    logging.info("批量结束：成功 %d / 失败 %d", ok, len(jobs) - ok)
    return summary


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_report",
        description="风控模型报告自动生成（支持评分卡 / 机器学习模型，批量出报告）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 单份报告（最简：只给结果表，不出特征类 sheet）
  python generate_report.py --config cfg.json --model SCORE --rnt rnt.csv --out out/r.xlsx

  # 完整报告（含特征分析）
  python generate_report.py --config cfg.json --model SCORE --rnt rnt.csv \\
      --wide wide.csv --importance imp.csv --out out/r.xlsx --top 10

  # 换模板 / 裁剪 sheet
  python generate_report.py --config cfg.json --model SCORE --rnt rnt.csv \\
      --out out/r.xlsx --style compact

  # 批量（多模型 / 多期 / 多标签）
  python generate_report.py --batch batch.json
""")
    # 批量模式
    p.add_argument("--batch", help="批量配置文件路径（与单报告参数互斥）")

    # 必需输入
    p.add_argument("--config", help="配置 JSON 路径（target/date/data_flag/channle_name 等）")
    p.add_argument("--model", help="模型分数列名（select_modelname）")
    p.add_argument("--rnt", help="模型结果表路径（csv/pkl，含分数+标签+日期+数据集标识）")
    p.add_argument("--out", help="报告输出路径（.xlsx）")

    # 可选输入
    p.add_argument("--wide", default=None, help="原始宽表路径（特征分析用）")
    p.add_argument("--importance", default=None, help="特征重要性文件路径")
    p.add_argument("--top", type=int, default=10, help="Top-N 重要特征（默认 10）")
    p.add_argument("--data-flag", default="train,test,oot",
                   help="数据集划分，逗号分隔（默认 train,test,oot）")
    p.add_argument("--no-univariate", action="store_true", help="关闭单变量分析（加速）")
    p.add_argument("--img-dir", default=None, help="图表输出目录")

    # 版式模板
    p.add_argument("--style", default=None,
                   help="版式模板：default / compact / 自定义 JSON 路径")
    p.add_argument("--dump-style", default=None,
                   help="把当前模板导出为 JSON（便于企业内复用），随后退出")

    # 阈值覆盖（同时作用于指标告警与 Excel 色阶刻度）
    p.add_argument("--psi-ok", type=float, default=None, help="PSI 观察阈值（默认 0.1）")
    p.add_argument("--psi-warn", type=float, default=None, help="PSI 告警阈值（默认 0.25）")
    p.add_argument("--iv", type=float, default=None, help="IV 入模阈值（默认 0.02）")
    p.add_argument("--ks-ok", type=float, default=None, help="KS 优秀线（默认 0.3，仅影响配色）")
    p.add_argument("--ks-warn", type=float, default=None, help="KS 可用线（默认 0.2，仅影响配色）")
    p.add_argument("--ks-overfit", type=float, default=None,
                   help="train→test/oot 的 KS 过拟合衰减阈值（默认 0.05）")

    p.add_argument("--log-level", default="INFO", help="日志级别（DEBUG/INFO/WARNING）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))

    # 模板导出（不跑数据）
    if args.dump_style:
        style = load_style(args.style)
        path = dump_style(style, args.dump_style)
        logging.info("模板已导出：%s", path)
        return 0

    # 批量模式
    if args.batch:
        summary = run_batch(args.batch)
        return 0 if summary["fail"] == 0 else 1

    # 单报告模式：必需参数校验（前置失败要早、要清楚）
    missing = [k for k, v in [("--config", args.config), ("--model", args.model),
                              ("--rnt", args.rnt), ("--out", args.out)] if not v]
    if missing:
        logging.error("缺少必需参数：%s（或用 --batch 走批量模式）", "、".join(missing))
        return 2

    # 配置：JSON + CLI 阈值覆盖
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    style = load_style(args.style)
    cli_thr = {
        "psi_ok": args.psi_ok, "psi_warn": args.psi_warn, "iv": args.iv,
        "ks_ok": args.ks_ok, "ks_warn": args.ks_warn, "ks_overfit": args.ks_overfit,
    }
    for key, val in cli_thr.items():
        if val is None:
            continue
        cfg_field, style_key = THRESHOLD_MAP[key]
        if cfg_field:
            cfg[cfg_field] = val
        if style_key:
            style.thresholds[style_key] = val
    if any(v is not None for v in cli_thr.values()):
        logging.info("阈值覆盖已生效：%s",
                     {k: v for k, v in cli_thr.items() if v is not None})

    get_report(
        config=cfg,
        select_modelname=args.model,
        data_path=args.wide,
        rnt_path=args.rnt,
        report_path=args.out,
        importance_path=args.importance,
        eval_univariate=not args.no_univariate,
        data_flag=[s.strip() for s in str(args.data_flag).split(",") if s.strip()],
        top=args.top,
        style=style,
        img_dir=args.img_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
