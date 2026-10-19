# -*- coding: utf-8 -*-
"""
模型评估层（ModelEvaluator）
================================================
按数据集 / 月份 / 渠道 / 客群 多维分组计算模型表现，并生成 Lift 分箱表。

设计要点
------------------------------------------------------------------
* **单一出口**：所有分组的指标均来自 `RiskMetrics.summarize`，杜绝各分组口径漂移
  （旧版 zt_metrics / qudao_metrics / month_metrics 各自实现，口径互不一致）
* **Lift 方向**：head=高分=高风险，tail=低分=低风险
  （旧版 `qudao_modelscore_metrics` 把头部累计 lift 赋给"尾部"，方向相反）
* **extended 生效**：旧版 `prob_table` 的 extended 参数形同虚设，此处真正输出扩展列
* **无副作用**：不修改传入的 DataFrame
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .metrics import RiskMetrics

logger = logging.getLogger(__name__)

# Lift 表中文列名（报告展示用）
_BIN_COLS_CN = ["分箱", "好样本数", "坏样本数", "总样本数", "样本占比", "箱内坏率", "区间lift", "累计lift"]
_BIN_COLS_EXT = ["累计好样本数", "累计坏样本数", "累计好样本占比", "累计坏样本占比", "KS", "WOE"]


class ModelEvaluator:
    """模型评估器：分组指标 + Lift 分箱表。"""

    def __init__(self, y_col: str, date_col: Optional[str] = None,
                 channel_col: Optional[str] = None, data_flag_col: str = "data_flag",
                 month_col: str = "month"):
        self.y_col = y_col
        self.date_col = date_col
        self.channel_col = channel_col
        self.data_flag_col = data_flag_col
        self.month_col = month_col

    # ------------------------------------------------------------------
    # 基础：过滤有效样本
    # ------------------------------------------------------------------
    def _valid(self, df: pd.DataFrame, prob: str) -> pd.DataFrame:
        cols = [self.y_col, prob]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"数据中缺少字段：{missing}")
        return df[df[self.y_col].isin([0, 1]) & df[prob].notna()]

    def _base_array(self, df: pd.DataFrame, prob: str) -> Optional[np.ndarray]:
        """PSI 基准分布：优先训练集，否则取全量。"""
        if self.data_flag_col in df.columns:
            tr = df.loc[df[self.data_flag_col] == "train", prob]
            if not tr.empty:
                return tr.dropna().values
        return df[prob].dropna().values

    # ------------------------------------------------------------------
    # 一、分组指标（统一出口）
    # ------------------------------------------------------------------
    def evaluate_group(self, df: pd.DataFrame, prob: str, group_col: Optional[str] = None,
                       psi_base: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        按任意字段分组计算全量指标 —— 数据集/月份/渠道/客群 均复用本方法。

        Parameters
        ----------
        group_col : 分组列名；None 表示整体计算
        psi_base  : PSI 基准分布（None 则自动取训练集分布）
        """
        sub = self._valid(df, prob)
        if sub.empty:
            return pd.DataFrame()
        if psi_base is None:
            psi_base = self._base_array(df, prob)

        groups: List = [None] if group_col is None else sorted(sub[group_col].dropna().unique())
        rows = []
        for g in groups:
            g_df = sub if g is None else sub[sub[group_col] == g]
            if g_df.empty:
                continue
            m = RiskMetrics.summarize(g_df[self.y_col], g_df[prob], psi_base=psi_base)
            m[group_col if group_col else "all"] = "all" if g is None else g
            rows.append(m)

        out = pd.DataFrame(rows)
        key = group_col if group_col else "all"
        if out.empty:
            return out
        out = out.set_index(key)
        return self._rename_metrics(out)

    @staticmethod
    def _rename_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """统一指标列命名（与人读报告对齐）。"""
        rename = {
            "lift_head_1pct": "lift-1%",
            "lift_head_5pct": "lift-5%",
            "lift_head_10pct": "lift-10%",
            "lift_head_20pct": "lift-20%",
            "lift_tail_10pct": "lift-尾部10%",
            "capture_10pct": "捕获率-10%",
            "capture_20pct": "捕获率-20%",
        }
        out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        preferred = ["CntRec", "CntBad", "bad_rate", "ks", "auc", "gini",
                     "lift-1%", "lift-5%", "lift-10%", "lift-20%", "lift-尾部10%",
                     "捕获率-10%", "捕获率-20%", "psi", "brier",
                     "ks_threshold", "precision", "recall", "fpr"]
        rest = [c for c in out.columns if c not in preferred]
        return out[[c for c in preferred if c in out.columns] + rest]

    # ------------------------------------------------------------------
    def evaluate_by_dataset(self, df: pd.DataFrame, prob: str,
                            data_flag_lst: Optional[List[str]] = None) -> pd.DataFrame:
        """按数据集（train/test/oot）评估。"""
        if self.data_flag_col not in df.columns:
            return self.evaluate_group(df, prob)
        flags = data_flag_lst or ["train", "test", "oot"]
        avail = [f for f in flags if f in set(df[self.data_flag_col].dropna())]
        out = self.evaluate_group(df, prob, group_col=self.data_flag_col)
        if not out.empty:
            out = out.reindex([f for f in flags if f in out.index])
            if "psi" in out.columns and "train" in out.index:
                out.loc["train", "psi"] = np.nan  # 基准行不参与比较
        return out

    def evaluate_by_month(self, df: pd.DataFrame, prob: str) -> pd.DataFrame:
        """按月份评估（PSI 以训练集为基准，缺失自动跳过）。"""
        if self.month_col not in df.columns:
            logger.warning("数据中无 %s 列，无法做逐月评估", self.month_col)
            return pd.DataFrame()
        return self.evaluate_group(df, prob, group_col=self.month_col)

    def evaluate_by_channel(self, df: pd.DataFrame, prob: str) -> pd.DataFrame:
        """按渠道评估。"""
        if not self.channel_col or self.channel_col not in df.columns:
            return pd.DataFrame()
        return self.evaluate_group(df, prob, group_col=self.channel_col)

    def evaluate_by_channel_month(self, df: pd.DataFrame, prob: str) -> pd.DataFrame:
        """渠道 × 月份 交叉评估（PSI 按渠道内环比）。"""
        if not self.channel_col or self.channel_col not in df.columns or self.month_col not in df.columns:
            return pd.DataFrame()
        sub = self._valid(df, prob)
        rows = []
        for ch, g in sub.groupby(self.channel_col):
            months = sorted(g[self.month_col].dropna().unique())
            prev = None
            for m in months:
                m_df = g[g[self.month_col] == m]
                rec = RiskMetrics.summarize(m_df[self.y_col], m_df[prob], psi_base=prev)
                rec["psi"] = 0.0 if prev is None else RiskMetrics.calc_psi(prev, m_df[prob].dropna().values)
                rec[self.channel_col] = ch
                rec[self.month_col] = m
                rows.append(rec)
                prev = m_df[prob].dropna().values
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows).set_index([self.channel_col, self.month_col])
        return self._rename_metrics(out)

    # ------------------------------------------------------------------
    # 二、整体概览（含过拟合诊断）
    # ------------------------------------------------------------------
    def dataset_summary(self, df: pd.DataFrame, prob: str) -> pd.DataFrame:
        """
        数据集指标 + 过拟合诊断行。

        过拟合判定：KS 绝对衰减 > 阈值（默认 0.05）即告警。
        """
        out = self.evaluate_by_dataset(df, prob)
        if out.empty:
            return out

        extra = {}
        if "train" in out.index and "test" in out.index:
            gap = RiskMetrics.overfit_gap(
                out.loc["train", "ks"], out.loc["test", "ks"],
                out.loc["train", "auc"], out.loc["test", "auc"],
            )
            extra = {
                "ks衰减": gap["ks_gap_abs"],
                "auc衰减": gap["auc_gap"],
                "过拟合告警": "是" if gap["is_overfit"] else "否",
            }
        for k, v in extra.items():
            # 按取值类型建列：字符串列必须是 object，否则 pandas 2.x 静默 upcast、
            # pandas 3.x 直接抛 LossySetitemError（Invalid value '否' for dtype 'float64'）
            dtype = "object" if isinstance(v, str) else "float64"
            out[k] = pd.Series(np.nan, index=out.index, dtype=dtype)
            out.loc["train", k] = v
        return out

    # ------------------------------------------------------------------
    # 三、Lift 分箱表（报告主体表）
    # ------------------------------------------------------------------
    def lift_table(self, df: pd.DataFrame, prob: str, bins: int = 10,
                   data_flag_lst: Optional[List[str]] = None,
                   extended: bool = False) -> Dict[str, pd.DataFrame]:
        """
        以训练集分位数为统一边界，生成各数据集的 Lift 分箱表。

        Returns
        -------
        dict: {data_flag: DataFrame}
            列（基础）：分箱 / 好样本数 / 坏样本数 / 总样本数 / 样本占比 /
                      箱内坏率 / 区间lift / 累计lift
            列（extended=True 追加）：累计好样本数 / 累计坏样本数 /
                      累计好样本占比 / 累计坏样本占比 / KS / WOE

        修复：旧版 extended 参数未生效（select_cols 写死 8 列）。
        """
        sub = self._valid(df, prob)
        if sub.empty:
            return {}

        # 统一边界：优先训练集，否则全量
        if self.data_flag_col in df.columns and "train" in set(df[self.data_flag_col].dropna()):
            base_arr = df.loc[df[self.data_flag_col] == "train", prob].dropna().values
        else:
            base_arr = sub[prob].dropna().values
        edges = RiskMetrics.psi_edges(base_arr, bins=bins)
        if edges.size < 2:
            return {}

        flags = data_flag_lst or (
            sorted(df[self.data_flag_col].dropna().unique()) if self.data_flag_col in df.columns else ["all"]
        )
        result: Dict[str, pd.DataFrame] = {}
        for flag in flags:
            if self.data_flag_col in df.columns:
                g = sub[sub[self.data_flag_col] == flag] if flag != "all" else sub
            else:
                g = sub
            if g.empty:
                continue
            result[flag] = self._build_lift(g, prob, self.y_col, edges, extended)
        return result

    @staticmethod
    def _build_lift(df: pd.DataFrame, prob: str, y_col: str, edges: np.ndarray,
                    extended: bool) -> pd.DataFrame:
        """构建单数据集 Lift 表（高分箱在前）。"""
        x = pd.to_numeric(df[prob], errors="coerce")
        cuts = pd.cut(x, bins=edges, include_lowest=True, right=True)
        tab = pd.DataFrame({"bin": cuts, "y": df[y_col].values})
        agg = tab.groupby("bin", observed=True)["y"].agg(CntRec="count", CntBad="sum").reset_index()
        agg["CntGood"] = agg["CntRec"] - agg["CntBad"]
        agg = agg.sort_values("bin", ascending=False).reset_index(drop=True)  # 高分在前

        tot_rec = float(agg["CntRec"].sum())
        tot_bad = float(agg["CntBad"].sum())
        tot_good = float(agg["CntGood"].sum())
        overall_bad = tot_bad / tot_rec if tot_rec else np.nan

        agg["PctRec"] = agg["CntRec"] / tot_rec if tot_rec else np.nan
        agg["bad_rate"] = agg["CntBad"] / agg["CntRec"].clip(lower=1)
        agg["lift"] = agg["bad_rate"] / overall_bad if overall_bad else np.nan

        cum_bad = agg["CntBad"].cumsum()
        cum_good = agg["CntGood"].cumsum()
        cum_rec = agg["CntRec"].cumsum()
        agg["cum_bad"] = cum_bad
        agg["cum_good"] = cum_good
        agg["cum_bad_rate"] = cum_bad / cum_rec.clip(lower=1)
        agg["cum_lift"] = agg["cum_bad_rate"] / overall_bad if overall_bad else np.nan
        agg["PctGood"] = agg["CntGood"] / tot_good if tot_good else np.nan
        agg["PctBad"] = agg["CntBad"] / tot_bad if tot_bad else np.nan
        agg["cum_PctGood"] = cum_good / tot_good if tot_good else np.nan
        agg["cum_PctBad"] = cum_bad / tot_bad if tot_bad else np.nan
        agg["KS"] = (agg["cum_PctBad"] - agg["cum_PctGood"]).abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            agg["WOE"] = np.log(agg["PctBad"].clip(lower=1e-6) / agg["PctGood"].clip(lower=1e-6))

        out = pd.DataFrame({
            "分箱": agg["bin"].astype(str),
            "好样本数": agg["CntGood"].astype(int),
            "坏样本数": agg["CntBad"].astype(int),
            "总样本数": agg["CntRec"].astype(int),
            "样本占比": agg["PctRec"].round(6),
            "箱内坏率": agg["bad_rate"].round(6),
            "区间lift": agg["lift"].round(4),
            "累计lift": agg["cum_lift"].round(4),
        })
        if extended:
            out["累计好样本数"] = agg["cum_good"].astype(int)
            out["累计坏样本数"] = agg["cum_bad"].astype(int)
            out["累计好样本占比"] = agg["cum_PctGood"].round(6)
            out["累计坏样本占比"] = agg["cum_PctBad"].round(6)
            out["KS"] = agg["KS"].round(4)
            out["WOE"] = agg["WOE"].round(4)

        total_row = {
            "分箱": "Total",
            "好样本数": int(tot_good),
            "坏样本数": int(tot_bad),
            "总样本数": int(tot_rec),
            "样本占比": 1.0,
            "箱内坏率": round(overall_bad, 6) if overall_bad == overall_bad else np.nan,
            "区间lift": 1.0,
            "累计lift": 1.0,
        }
        if extended:
            total_row.update({
                "累计好样本数": int(tot_good), "累计坏样本数": int(tot_bad),
                "累计好样本占比": 1.0, "累计坏样本占比": 1.0,
                "KS": 0.0, "WOE": np.nan,
            })
        return pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

    def monthly_lift(self, df: pd.DataFrame, prob: str, bins: int = 20) -> pd.DataFrame:
        """逐月 Lift 表（每月一张，纵向拼接，带 month 标识）。"""
        if self.month_col not in df.columns:
            return pd.DataFrame()
        sub = self._valid(df, prob)
        rows = []
        for m in sorted(sub[self.month_col].dropna().unique()):
            g = sub[sub[self.month_col] == m]
            tbl = self.lift_table(g, prob, bins=bins, data_flag_lst=["all"])
            if "all" in tbl:
                t = tbl["all"].copy()
                t.insert(0, "month", m)
                rows.append(t)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # 四、逐月好坏样本数（修复 crosstab 缺列崩溃）
    # ------------------------------------------------------------------
    def month_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        逐月样本量与坏账率。

        修复：旧版用 `crosstab` 后直接取列，某月无坏样本时缺列导致 KeyError。
        """
        if self.month_col not in df.columns:
            return pd.DataFrame()
        sub = df[df[self.y_col].isin([0, 1])]
        grp = sub.groupby(self.month_col)[self.y_col].agg(CntRec="count", CntBad="sum").reset_index()
        grp["CntGood"] = grp["CntRec"] - grp["CntBad"]
        grp["bad_rate"] = grp["CntBad"] / grp["CntRec"].clip(lower=1)
        return grp.set_index(self.month_col).sort_index()


__all__ = ["ModelEvaluator"]
