# -*- coding: utf-8 -*-
"""
可视化层（ChartVisualizer）
================================================
修复的旧缺陷
------------------------------------------------------------------
1. `GetKS` 无 `ylim` 参数，但调用方传了 `ylim=` —— TypeError（已支持）
2. 模块顶层执行 `plt.rcParams[...] = ...` —— import 即产生全局副作用（改为实例化时显式设置）
3. 未设置非交互后端 —— 无显示环境（服务器/CI）下绘图崩溃（改为 Agg）
4. shap 为强依赖 —— 未安装时 import 即失败（改为可选依赖）

所有方法统一签名 `plot_*(..., name, path)`，返回图片保存路径，失败返回 None 而不中断报告。
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 非交互后端，保证无显示环境可用
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

logger = logging.getLogger(__name__)

# 中文字体候选（按系统可用性依次尝试）
_CN_FONTS = ["PingFang SC", "Heiti TC", "SimHei", "Microsoft YaHei", "Arial Unicode MS", "STHeiti"]


class ChartVisualizer:
    """报告图表生成器。"""

    def __init__(self, style: str = "whitegrid", figsize: Sequence[int] = (8, 5), dpi: int = 120):
        self.figsize = tuple(figsize)
        self.dpi = dpi
        # 显式设置（不在模块顶层污染全局）
        sns.set_theme(style=style)
        plt.rcParams["axes.unicode_minus"] = False
        self._setup_chinese_font()

    @staticmethod
    def _setup_chinese_font() -> None:
        """按系统可用性设置中文字体，避免中文乱码。"""
        try:
            from matplotlib import font_manager
            available = {f.name for f in font_manager.fontManager.ttflist}
            for f in _CN_FONTS:
                if f in available:
                    plt.rcParams["font.sans-serif"] = [f]
                    return
        except Exception as exc:  # pragma: no cover
            logger.debug("中文字体设置跳过：%s", exc)

    # ------------------------------------------------------------------
    def _save(self, fig, path: Optional[str], name: str) -> Optional[str]:
        if path is None:
            return None
        os.makedirs(path, exist_ok=True)
        fp = os.path.join(path, f"{name}.png")
        try:
            fig.savefig(fp, dpi=self.dpi, bbox_inches="tight")
            logger.info("图表已保存：%s", fp)
            return fp
        except Exception as exc:
            logger.warning("图表保存失败 %s：%s", fp, exc)
            return None
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    def plot_roc(self, y_true, y_score, name: str = "ROC", path: Optional[str] = None,
                 color: str = "darkorange") -> Optional[str]:
        """ROC 曲线（含 AUC 标注）。"""
        try:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
        except Exception as exc:
            logger.warning("ROC 计算失败：%s", exc)
            return None

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} ROC Curve")
        ax.legend(loc="lower right")
        return self._save(fig, path, f"{name}_ROC")

    def plot_ks(self, y_true, y_score, name: str = "KS", path: Optional[str] = None,
                ylim: Optional[Sequence[float]] = None) -> Optional[str]:
        """
        KS 曲线：各阈值下 TPR/FPR/KS。

        Parameters
        ----------
        ylim : Y 轴范围，如 [0, 0.5]（旧版调用方传了该参数但函数不支持 → TypeError）
        """
        try:
            fpr, tpr, thr = roc_curve(y_true, y_score)
        except Exception as exc:
            logger.warning("KS 计算失败：%s", exc)
            return None

        ks = tpr - fpr
        best = int(np.argmax(ks))
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(thr, tpr, label="TPR (坏样本覆盖率)", color="red")
        ax.plot(thr, fpr, label="FPR (好样本误杀率)", color="blue")
        ax.plot(thr, ks, label=f"KS (max={ks[best]:.4f})", color="green")
        ax.axvline(thr[best], ls="--", c="gray",
                   label=f"最优切点={thr[best]:.4f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Rate")
        if ylim is not None:
            ax.set_ylim(list(ylim))
        ax.set_title(f"{name} KS Curve")
        ax.legend(loc="upper right", fontsize=8)
        return self._save(fig, path, f"{name}_KS")

    def plot_psi_heat(self, matrix: pd.DataFrame, name: str = "PSI",
                      path: Optional[str] = None, annot: bool = True) -> Optional[str]:
        """PSI 月份热力矩阵。"""
        if matrix is None or matrix.empty:
            return None
        fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.8), max(6, len(matrix) * 0.6)))
        sns.heatmap(matrix.astype(float), annot=annot, fmt=".4f", cmap="RdYlGn_r",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "PSI"})
        ax.set_title(f"{name} PSI Heatmap")
        return self._save(fig, path, f"{name}_PSI_Heat")

    def plot_score_dist(self, df: pd.DataFrame, score_col: str, hue: Optional[str] = None,
                        name: str = "ScoreDist", path: Optional[str] = None) -> Optional[str]:
        """模型分分布（按 hue 分组的核密度图）。"""
        if score_col not in df.columns:
            return None
        fig, ax = plt.subplots(figsize=self.figsize)
        try:
            if hue and hue in df.columns:
                for v, g in df.groupby(hue):
                    sns.kdeplot(x=g[score_col].dropna(), label=str(v), ax=ax, warn_singular=False)
                ax.legend(fontsize=8)
            else:
                sns.kdeplot(x=df[score_col].dropna(), ax=ax, warn_singular=False)
        except Exception as exc:
            logger.warning("分布图绘制失败：%s", exc)
            plt.close(fig)
            return None
        ax.set_xlabel(score_col); ax.set_title(f"{name} Score Distribution")
        return self._save(fig, path, f"{name}_Dist")

    def plot_bad_rate_trend(self, series: pd.Series, name: str = "BadRateTrend",
                            path: Optional[str] = None) -> Optional[str]:
        """逐月坏账率趋势。"""
        if series is None or series.empty:
            return None
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(series.index.astype(str), series.values, marker="o", color="crimson")
        ax.set_ylabel("Bad Rate"); ax.set_xlabel("Month")
        ax.set_title(f"{name} Monthly Bad Rate")
        plt.xticks(rotation=45)
        return self._save(fig, path, f"{name}_BadRateTrend")

    def plot_lift(self, lift_df: pd.DataFrame, name: str = "Lift",
                  path: Optional[str] = None) -> Optional[str]:
        """Lift 柱状图（区间 lift 与累计 lift）。"""
        if lift_df is None or lift_df.empty or "区间lift" not in lift_df.columns:
            return None
        d = lift_df[lift_df["分箱"] != "Total"]
        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["区间lift"], width=0.4, label="区间lift", color="steelblue")
        if "累计lift" in d.columns:
            ax.bar(x + 0.2, d["累计lift"], width=0.4, label="累计lift", color="darkorange")
        ax.set_xticks(x); ax.set_xticklabels(d["分箱"].astype(str), rotation=45, fontsize=7)
        ax.set_ylabel("Lift"); ax.set_title(f"{name} Lift by Bin")
        ax.legend(fontsize=8)
        return self._save(fig, path, f"{name}_Lift")

    def plot_shap(self, model, X: pd.DataFrame, name: str = "SHAP",
                  path: Optional[str] = None, max_display: int = 20) -> Optional[str]:
        """SHAP 特征贡献图（shap 为可选依赖，未安装时跳过）。"""
        try:
            import shap
        except ImportError:
            logger.info("未安装 shap，跳过 SHAP 图")
            return None
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            fig, ax = plt.subplots(figsize=(8, max(4, min(max_display, X.shape[1]) * 0.35)))
            shap.summary_plot(sv, X, max_display=max_display, show=False)
            return self._save(fig, path, f"{name}_SHAP")
        except Exception as exc:
            logger.warning("SHAP 图绘制失败：%s", exc)
            return None


__all__ = ["ChartVisualizer"]
