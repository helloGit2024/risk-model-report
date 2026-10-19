# -*- coding: utf-8 -*-
"""
数据划分层（DataSplitter）
================================================
从 Utils.data_flag_split / data_flag_channle_split 迁移而来，修复以下问题：

1. **隐式依赖 main_config.json**：旧版在函数内部 `getConfigJson()` 读固定相对路径，
   脱离指定目录即崩溃 → 改为配置注入（传 dict，可选）
2. **回写配置文件**：旧版会修改 main_config.json 的 cat 字段（副作用）→ 已移除
3. **两处配置来源不一致**：`data_flag_channle_split` 第 218 行用参数、第 247 行用
   config，行为不一致 → 统一为参数优先

划分逻辑（符合信贷规范）
------------------------------------------------------------------
* OOT / 盲测：按**时间区间**切分（时间外验证，不参与训练）
* train / test：在剩余样本上按 **分层随机** 切分（stratify 保持坏率一致）
* 灰样本：标签不在 {0,1} 的样本单独标记，不参与训练
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """建模样本划分器。"""

    def __init__(self, target: str, date_col: str, channel_col: Optional[str] = None,
                 other_label_value: Optional[List] = None):
        self.target = target
        self.date_col = date_col
        self.channel_col = channel_col
        self.other_label_value = other_label_value or []

    # ------------------------------------------------------------------
    def split(self, df: pd.DataFrame, oot_split: Optional[List[str]] = None,
              oot_split2: Optional[List[str]] = None, test_size: float = 0.15,
              channel_split: Optional[Dict] = None) -> pd.DataFrame:
        """
        划分数据集，返回带 `data_flag` 列的副本（**不修改原 df**）。

        Parameters
        ----------
        oot_split   : OOT 日期区间 [start, end]
        oot_split2  : 盲测日期区间 [start, end]（可选）
        test_size   : 测试集比例，0=无 test，1=无 train
        channel_split : {渠道: [start, end]}，按渠道各自指定 OOT 区间
        """
        out = df.copy()
        oot_split = oot_split or []
        oot_split2 = oot_split2 or []

        out["data_flag"] = np.nan
        date = out[self.date_col]

        # 1. 盲测
        if len(oot_split2) == 2:
            out.loc[date.between(oot_split2[0], oot_split2[1], inclusive="both"), "data_flag"] = "盲测"

        # 2. oot
        if channel_split:
            for channel, dates in channel_split.items():
                if len(dates) != 2 or self.channel_col is None:
                    continue
                ch = int(channel) if str(channel).isdigit() else channel
                m = ((out[self.channel_col] == ch)
                     & date.between(dates[0], dates[1], inclusive="both")
                     & out[self.target].isin([0, 1]))
                out.loc[m, "data_flag"] = "oot"
        elif len(oot_split) == 2:
            m = date.between(oot_split[0], oot_split[1], inclusive="both") & out[self.target].isin([0, 1])
            out.loc[m, "data_flag"] = "oot"

        # 3. 灰样本
        if self.other_label_value:
            m = out[self.target].isin(self.other_label_value) & out["data_flag"].isna()
            out.loc[m, "data_flag"] = "灰样本"

        # 4. train + test
        pool = out[out[self.target].isin([0, 1]) & out["data_flag"].isna()]
        if test_size <= 0:
            tr_idx, te_idx = pool.index, pd.Index([])
            logger.info("注意：全部作为训练集，无测试集")
        elif test_size >= 1:
            tr_idx, te_idx = pd.Index([]), pool.index
            logger.info("注意：全部作为测试集，无训练集")
        else:
            try:
                tr, te = train_test_split(pool, test_size=test_size, random_state=42,
                                          stratify=pool[self.target])
                tr_idx, te_idx = tr.index, te.index
            except Exception as exc:
                logger.warning("分层切分失败(%s)，退化为普通随机切分", exc)
                tr, te = train_test_split(pool, test_size=test_size, random_state=42)
                tr_idx, te_idx = tr.index, te.index

        out.loc[tr_idx, "data_flag"] = "train"
        out.loc[te_idx, "data_flag"] = "test"
        out["data_flag"] = out["data_flag"].fillna("other")

        logger.info("数据集划分完成：%s", out["data_flag"].value_counts().to_dict())
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def ensure_month(df: pd.DataFrame, date_col: str, month_col: str = "month") -> pd.DataFrame:
        """
        从日期列派生月份列（YYYY-MM）。若已存在则不覆盖。

        修复：runReport 全流程依赖 month 列，但旧版从不生成，导致逐月分析全线崩溃。
        """
        if month_col in df.columns:
            return df
        out = df.copy()
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out[month_col] = dt.dt.strftime("%Y-%m")
        return out


__all__ = ["DataSplitter"]
