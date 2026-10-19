# -*- coding: utf-8 -*-
"""
稳定性层（StabilityMonitor）
================================================
全库唯一的 PSI 入口，合并旧代码中的四套重复实现
（get_fd_stability / feature_get_psi / get_psiheat / Utils.Cal_Psi），
并修复 `Cal_Psi` 因 inner-join 丢空箱导致 PSI 系统性低估的严重缺陷。

PSI 分档：< 0.1 稳定 | 0.1~0.25 需观察 | > 0.25 显著漂移
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import RiskMetrics

logger = logging.getLogger(__name__)


class StabilityMonitor:
    """模型分与特征的稳定性监控。"""

    def __init__(self, psi_threshold: float = 0.1, psi_alert_threshold: float = 0.25,
                 bins: int = 10):
        self.psi_threshold = psi_threshold
        self.psi_alert_threshold = psi_alert_threshold
        self.bins = bins

    # ------------------------------------------------------------------
    def classify(self, psi: float) -> str:
        """PSI 分档标签。"""
        if psi is None or pd.isna(psi):
            return "无法计算"
        if psi >= self.psi_alert_threshold:
            return "显著漂移"
        if psi >= self.psi_threshold:
            return "需观察"
        return "稳定"

    # ------------------------------------------------------------------
    # 一、模型分 PSI
    # ------------------------------------------------------------------
    def score_psi_by_dataset(self, df: pd.DataFrame, prob: str,
                             data_flag_col: str = "data_flag",
                             base_flag: str = "train") -> pd.DataFrame:
        """各数据集模型分相对训练集（或首个数据集）的 PSI。"""
        flags = [f for f in [base_flag, "test", "oot", "盲测"] if f in set(df[data_flag_col].dropna())]
        if not flags:
            flags = sorted(df[data_flag_col].dropna().unique())

        base_arr = df.loc[df[data_flag_col] == flags[0], prob].dropna().values
        edges = RiskMetrics.psi_edges(base_arr, bins=self.bins)

        rows = []
        for f in flags:
            arr = df.loc[df[data_flag_col] == f, prob].dropna().values
            if f == flags[0]:
                psi = 0.0
            else:
                psi = RiskMetrics.calc_psi(base_arr, arr, edges=edges)
            rows.append({"data_flag": f, "CntRec": int(arr.size), "psi": psi,
                         "grade": "base" if f == flags[0] else self.classify(psi)})
        return pd.DataFrame(rows).set_index("data_flag")

    def score_psi_by_month(self, df: pd.DataFrame, prob: str, month_col: str = "month",
                           compare_train: bool = True,
                           data_flag_col: str = "data_flag") -> pd.DataFrame:
        """
        逐月模型分 PSI。

        compare_train=True  → 每个月都与训练集整体比较（推荐，最常用）
        compare_train=False → 逐月环比（以上一月为基准）
        """
        months = sorted(df[month_col].dropna().unique())
        rows = []
        prev_arr = None
        base_arr = None
        base_edges = None
        if compare_train and data_flag_col in df.columns and "train" in set(df[data_flag_col].dropna()):
            base_arr = df.loc[df[data_flag_col] == "train", prob].dropna().values
            base_edges = RiskMetrics.psi_edges(base_arr, bins=self.bins)

        for m in months:
            arr = df.loc[df[month_col] == m, prob].dropna().values
            if compare_train and base_arr is not None:
                psi = 0.0 if arr.size == 0 else RiskMetrics.calc_psi(base_arr, arr, edges=base_edges)
                ref = "train"
            else:
                if prev_arr is None:
                    psi = 0.0
                    ref = "-"
                else:
                    psi = RiskMetrics.calc_psi(prev_arr, arr, bins=self.bins)
                    ref = "上月"
            rows.append({"month": m, "CntRec": int(arr.size), "psi": psi,
                         "ref": ref, "grade": self.classify(psi)})
            prev_arr = arr
        return pd.DataFrame(rows).set_index("month")

    def psi_heat_matrix(self, df: pd.DataFrame, prob: str,
                        month_col: str = "month") -> pd.DataFrame:
        """月份 × 月份 PSI 对称热力矩阵（对角线为 0）。"""
        return RiskMetrics.calc_psi_matrix(df, prob, month_col=month_col, bins=self.bins)

    # ------------------------------------------------------------------
    # 二、特征稳定性 CSI
    # ------------------------------------------------------------------
    def feature_psi(self, base_df: pd.DataFrame, curr_df: pd.DataFrame,
                    feats: Sequence[str], bins: int = None) -> pd.DataFrame:
        """
        特征级 PSI（CSI）：逐特征比较 base 与 curr 的分布漂移。

        支持数值型与类别型（旧版遇类别型会崩溃）。
        """
        bins = bins or self.bins
        rows = []
        for col in feats:
            if col not in base_df.columns or col not in curr_df.columns:
                continue
            b = base_df[col]
            c = curr_df[col]
            try:
                psi = RiskMetrics.calc_psi(b, c, bins=bins)
            except Exception as exc:
                logger.debug("特征 %s PSI 计算失败：%s", col, exc)
                psi = np.nan
            rows.append({
                "feature": col,
                "psi": psi,
                "grade": self.classify(psi),
                "missing_rate_base": RiskMetrics.calc_missing_rate(b),
                "missing_rate_curr": RiskMetrics.calc_missing_rate(c),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
        return out

    def feature_psi_multi(self, df: pd.DataFrame, feats: Sequence[str],
                          data_flag_col: str = "data_flag",
                          pairs: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        多数据集特征 PSI 对比。默认比较 (train→test) 与 (train→oot)。
        """
        pairs = pairs or {"psi_test": ("train", "test"), "psi_oot": ("train", "oot")}
        base_df = df[df[data_flag_col] == "train"]
        res = pd.DataFrame({"feature": list(feats)}).set_index("feature")
        for name, (src, dst) in pairs.items():
            if src not in set(df[data_flag_col].dropna()) or dst not in set(df[data_flag_col].dropna()):
                continue
            sub = self.feature_psi(base_df, df[df[data_flag_col] == dst], feats)
            res[name] = sub.set_index("feature")["psi"].reindex(res.index)
        return res.reset_index()

    # ------------------------------------------------------------------
    # 三、逐月坏账/分数趋势（供图表使用）
    # ------------------------------------------------------------------
    def bin_trend_blocks(self, df: pd.DataFrame, prob: str, y: str,
                         bins: int = 10, month_col: str = "month",
                         channel_col: Optional[str] = None) -> List[Dict]:
        """
        分箱坏账趋势（定版版式）：**整体 + 各渠道** 各为一个块，自上而下排列。

        行：分组值（纵向合并） + score_bin（高分箱在前） + total 汇总行
        列：两级表头 —— score_bin 之后接 5 个指标块 × 月份
            用户数 / 用户数占比 / 逾期用户数 / 逾期率 / LIFT

        口径说明
        --------
        * 用户数占比：该分箱用户数 / **当月**总用户数（按月归一化）
        * 逾期率    ：该分箱逾期数 / 该分箱用户数
        * LIFT      ：该分箱逾期率 / 当月整体逾期率
        * total 行  ：当月汇总（占比恒为 1，LIFT 恒为 1）

        Returns
        -------
        List[Dict]: 可直接交给 `ReportBuilder.add_multi_block_sheet` 的块列表
        """
        keep = [prob, month_col, y]
        if channel_col and channel_col in df.columns:
            keep.append(channel_col)
        sub = df[keep].copy()
        sub = sub[sub[y].isin([0, 1]) & sub[prob].notna()]
        if sub.empty:
            return []

        months = [str(m) for m in sorted(sub[month_col].dropna().unique())]
        edges = RiskMetrics.psi_edges(sub[prob], bins=bins)
        cuts = pd.cut(sub[prob], bins=edges, include_lowest=True)
        sub = sub.assign(_bin=cuts.astype(str))
        # 高分箱在前（与模型分10箱/20箱版式一致）
        order = [str(c) for c in reversed(list(cuts.cat.categories))]

        metrics = ["用户数", "用户数占比", "逾期用户数", "逾期率", "LIFT"]

        def build(name: str, g: pd.DataFrame) -> Dict:
            grp = (g.groupby(["_bin", month_col], observed=True)[y]
                     .agg(CntRec="count", CntBad="sum").reset_index())
            cnt = (grp.pivot(index="_bin", columns=month_col, values="CntRec")
                      .reindex(index=order, columns=months))
            bad = (grp.pivot(index="_bin", columns=month_col, values="CntBad")
                      .reindex(index=order, columns=months))

            cnt_sum = cnt.sum(axis=0)
            bad_sum = bad.sum(axis=0)
            safe_cnt = cnt_sum.replace(0, np.nan)
            rate_all = (bad_sum / safe_cnt).round(6)
            share = (cnt / safe_cnt).round(6)
            rate = (bad / cnt.replace(0, np.nan)).round(6)
            lift = (rate / rate_all.replace(0, np.nan)).round(4)

            cols = [("", "score"), ("score_bin", "")]
            for metric in metrics:
                for m in months:
                    cols.append((metric, m))

            rows = []
            for b in order:
                rows.append([name, b]
                            + [cnt.loc[b, m] for m in months]
                            + [share.loc[b, m] for m in months]
                            + [bad.loc[b, m] for m in months]
                            + [rate.loc[b, m] for m in months]
                            + [lift.loc[b, m] for m in months])
            rows.append([name, "total"]
                        + [cnt_sum[m] for m in months]
                        + [1.0 for _ in months]
                        + [bad_sum[m] for m in months]
                        + [rate_all[m] for m in months]
                        + [1.0 for _ in months])

            out = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(cols))
            title = "模型分（整体）" if name == "整体" else f"模型分（渠道{name}）"
            return {"title": title, "df": out, "header_rows": 2, "preset": "trend",
                    "merge_col": ("", "score"),
                    # 期望版：LIFT 用数据条、两位小数（负数标红）
                    "bar_blocks": ("LIFT",),
                    "block_formats": {"LIFT": "0.00_);[Red]\\(0.00\\)"}}

        blocks = [build("整体", sub)]
        if channel_col and channel_col in df.columns:
            for ch, g in sub.groupby(channel_col):
                blocks.append(build(str(ch), g))
        return blocks


__all__ = ["StabilityMonitor"]
