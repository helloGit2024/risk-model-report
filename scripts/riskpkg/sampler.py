# -*- coding: utf-8 -*-
"""
样本统计层（DataStatistics）
================================================
替代原 `dataMerge.Data_statistics`（该模块在工程中缺失，导致链路断裂）。

坏账率口径：**坏账率 = 坏样本数 / (好样本数 + 坏样本数)**
（旧版 `zt_bad_rate` 用 `iloc[-1,-3]/iloc[-1,-2]` 取"好/坏"之比，语义可疑且强依赖
外部模块列序，此处明确定义，杜绝歧义）

灰样本：标签取值不在 {0,1} 中的样本（如 M2/M3/未知），单独统计不计入坏率分母。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataStatistics:
    """建模样本统计：数据集 / 逐月 / 渠道 / 渠道×月。"""

    def __init__(self, df: pd.DataFrame, y_col: str, date_col: Optional[str] = None,
                 channel_col: Optional[str] = None, month_col: str = "month",
                 data_flag_col: str = "data_flag",
                 other_label_value: Optional[List] = None,
                 model_name: Optional[str] = None):
        self.df = df
        self.y = y_col
        self.date_col = date_col
        self.channel_col = channel_col
        self.month_col = month_col
        self.data_flag_col = data_flag_col
        self.other_label_value = other_label_value or []
        self.model_name = model_name

    # ------------------------------------------------------------------
    def _agg(self, sub: pd.DataFrame, index_col: Optional[str] = None,
             index_name: str = "分组") -> pd.DataFrame:
        """核心聚合：好/坏/灰/合计 + 坏账率。"""
        if sub is None or sub.empty:
            return pd.DataFrame()

        y = sub[self.y]
        is_grey = (y.isin(self.other_label_value).astype(int)
                   if self.other_label_value else pd.Series(0, index=sub.index))

        # 逐行打标后聚合：保证 index_col 分组与不分组两种场景行为一致
        marks = pd.DataFrame({
            "好样本数": (y == 0).astype(int),
            "坏样本数": (y == 1).astype(int),
            "灰样本数": is_grey,
            "总样本数": 1,
        }, index=sub.index)

        if index_col is not None:
            marks[index_col] = sub[index_col].values
            agg = marks.groupby(index_col, dropna=False).sum()
            agg.index.name = index_col
        else:
            agg = marks.sum().to_frame().T
            agg.index = [index_name]
            agg.index.name = index_name

        agg["总样本数"] = agg["总样本数"].astype(int)
        agg["坏账率"] = (agg["坏样本数"]
                        / (agg["好样本数"] + agg["坏样本数"]).clip(lower=1)).round(6)
        return agg

    # ------------------------------------------------------------------
    def dataset_sample(self) -> pd.DataFrame:
        """各数据集（train/test/oot/…）样本与坏账率，末行 total。"""
        if self.data_flag_col not in self.df.columns:
            return self._agg(self.df, index_name="total")
        rows = []
        for flag, sub in self.df.groupby(self.data_flag_col):
            r = self._agg(sub)
            r.index = [flag]
            r.index.name = "data_flag"
            rows.append(r)
        out = pd.concat(rows) if rows else pd.DataFrame()
        if not out.empty:
            total = self._agg(self.df)
            total.index = ["total"]
            total.index.name = "data_flag"
            out = pd.concat([out, total])
        return out

    def month_sample(self) -> pd.DataFrame:
        """逐月样本与坏账率，末行 total。"""
        if self.month_col not in self.df.columns:
            logger.warning("数据中无 %s 列，跳过逐月统计", self.month_col)
            return pd.DataFrame()
        out = self._agg(self.df, index_col=self.month_col).sort_index()
        if not out.empty:
            total = self._agg(self.df)
            total.index = ["total"]
            total.index.name = self.month_col
            out = pd.concat([out, total])
        return out

    def channel_sample(self) -> pd.DataFrame:
        """各渠道样本与坏账率，末行 total。"""
        if not self.channel_col or self.channel_col not in self.df.columns:
            return pd.DataFrame()
        out = self._agg(self.df, index_col=self.channel_col).sort_index()
        if not out.empty:
            total = self._agg(self.df)
            total.index = ["total"]
            total.index.name = self.channel_col
            out = pd.concat([out, total])
        return out

    def channel_month_sample(self) -> pd.DataFrame:
        """渠道 × 月份 交叉样本与坏账率。"""
        if not self.channel_col or self.channel_col not in self.df.columns or self.month_col not in self.df.columns:
            return pd.DataFrame()
        rows = []
        for ch, sub in self.df.groupby(self.channel_col):
            r = self._agg(sub, index_col=self.month_col)
            r.insert(0, self.channel_col, ch)
            rows.append(r.reset_index())
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not out.empty:
            out = out.set_index([self.channel_col, self.month_col]).sort_index()
        return out

    # ------------------------------------------------------------------
    def _flag_date_range(self, flag: str) -> Optional[str]:
        """单个数据集的时间区间（YYYY-MM-DD ~ YYYY-MM-DD）。"""
        if self.data_flag_col not in self.df.columns or not self.date_col:
            return None
        if self.date_col not in self.df.columns:
            return None
        sub = self.df[self.df[self.data_flag_col] == flag]
        if sub.empty:
            return None
        d = pd.to_datetime(sub[self.date_col], errors="coerce").dropna()
        if d.empty:
            return None
        return f"{d.min().strftime('%Y-%m-%d')} ~ {d.max().strftime('%Y-%m-%d')}"

    def _span_check(self) -> Optional[str]:
        """
        样本时间跨度校验：起止日期的自然月跨度 vs 实际出现的月份数。

        不一致时列出缺失月份 —— 数据漏月会让逐月趋势出现断点，必须显式提示。
        """
        if not self.date_col or self.date_col not in self.df.columns:
            return None
        if self.month_col not in self.df.columns:
            return None
        d = pd.to_datetime(self.df[self.date_col], errors="coerce").dropna()
        if d.empty:
            return None

        start, end = d.min(), d.max()
        span = (end.year - start.year) * 12 + (end.month - start.month) + 1
        actual = int(self.df[self.month_col].nunique())
        if span == actual:
            return f"一致（{actual}个月）"

        have = {str(m)[:7] for m in self.df[self.month_col].dropna().unique()}
        full = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")
        missing = [str(p) for p in full if str(p) not in have]
        tip = ("，缺失：" + "、".join(missing[:6]) + ("等" if len(missing) > 6 else "")
               if missing else "")
        return f"不一致：起止跨度{span}个月，实际{actual}个月{tip}"

    def overview(self) -> pd.DataFrame:
        """
        样本总览：样本量、时间范围、渠道数、模型名称、各数据集时间区间、跨度校验。
        """
        info = {
            "总样本数": len(self.df),
            "好样本数": int((self.df[self.y] == 0).sum()),
            "坏样本数": int((self.df[self.y] == 1).sum()),
            "灰样本数": int(self.df[self.y].isin(self.other_label_value).sum()) if self.other_label_value else 0,
        }
        info["整体坏账率"] = round(
            info["坏样本数"] / max(info["好样本数"] + info["坏样本数"], 1), 6
        )
        if self.date_col and self.date_col in self.df.columns:
            d = pd.to_datetime(self.df[self.date_col], errors="coerce").dropna()
            if not d.empty:
                info["起始日期"] = d.min().strftime("%Y-%m-%d")
                info["结束日期"] = d.max().strftime("%Y-%m-%d")
        if self.month_col in self.df.columns:
            info["月份数"] = int(self.df[self.month_col].nunique())
        if self.channel_col and self.channel_col in self.df.columns:
            info["渠道数"] = int(self.df[self.channel_col].nunique())
        if self.model_name:
            info["模型名称"] = self.model_name

        # 各数据集时间区间
        if self.data_flag_col in self.df.columns:
            for flag in [f for f in ["train", "test", "oot"]
                         if f in set(self.df[self.data_flag_col].dropna())]:
                rng = self._flag_date_range(flag)
                if rng:
                    info[f"{flag}时间区间"] = rng

        span = self._span_check()
        if span:
            info["时间跨度校验"] = span

        return pd.DataFrame([info]).T.rename(columns={0: "值"})


__all__ = ["DataStatistics"]
