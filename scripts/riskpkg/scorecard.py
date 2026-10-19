# -*- coding: utf-8 -*-
"""
评分卡转换层（ScoreConverter）
================================================
全库唯一的 概率 <-> 分数 转换实现（旧代码此处有 3 份重复实现，已合并）。

标准评分卡公式
------------------------------------------------------------------
    B     = PDO / ln(2)
    A     = base_score - B × ln(base_odds)
    Score = A - B × ln(p / (1 - p))

其中 odds = (1-p)/p 为好坏比；当好坏比 = base_odds 时分数恰为 base_score。
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd


class ScoreConverter:
    """概率与评分卡分数的双向转换。"""

    def __init__(self, pdo: int = 60, base_score: int = 600, base_odds: float = 20.0,
                 min_score: float = 0.0, max_score: float = 1000.0):
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.min_score = min_score
        self.max_score = max_score
        self.B = pdo / math.log(2)
        self.A = base_score - self.B * math.log(base_odds)

    # ------------------------------------------------------------------
    def p_to_score(self, p: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """逾期概率 → 模型分数（分数越低风险越高，符合评分卡惯例）。"""
        arr = np.asarray(p, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = np.where((arr <= 0) | (arr >= 1), np.nan, arr / (1 - arr))
            score = self.A - self.B * np.log(odds)
        score = np.clip(score, self.min_score, self.max_score)
        score = np.where(np.isnan(arr), np.nan, score)

        if isinstance(p, pd.Series):
            return pd.Series(score, index=p.index).round(0)
        if np.isscalar(p):
            return float(np.round(score, 0)) if not np.isnan(score) else np.nan
        return np.round(score, 0)

    def score_to_p(self, score: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        模型分数 → 逾期概率（p_to_score 的严格逆运算）。

        由 Score = A - B·ln(p/(1-p)) 反解：
            odds = p/(1-p) = exp((A - Score)/B)  →  p = odds / (1 + odds)

        注意 odds 是"坏好比"（不是好坏比），否则反解会得到 p 与 1-p 互换的错误结果。
        """
        arr = np.asarray(score, dtype=float)
        odds = np.exp((self.A - arr) / self.B)   # odds = p/(1-p)
        p = odds / (1 + odds)
        if isinstance(score, pd.Series):
            return pd.Series(p, index=score.index)
        if np.isscalar(score):
            return float(p)
        return p

    # ------------------------------------------------------------------
    @staticmethod
    def get_score(prob: Union[float, np.ndarray, pd.Series], digits: int = 0) -> Union[float, np.ndarray, pd.Series]:
        """概率 × 1000 取整（简单线性映射，非标准评分卡，保留旧行为）。"""
        arr = np.asarray(prob, dtype=float)
        out = np.floor(arr * 1000)
        if isinstance(prob, pd.Series):
            return pd.Series(out, index=prob.index)
        if np.isscalar(prob):
            return float(out)
        return out

    # ------------------------------------------------------------------
    def score_distribution(self, prob, name: str = "score") -> pd.Series:
        """分数分布统计（均值/标准差/分位数），用于分数稳定性报告。"""
        s = self.p_to_score(prob)
        s = pd.Series(s).dropna()
        if s.empty:
            return pd.Series(dtype=float, name=name)
        return pd.Series(
            {
                "count": int(s.size),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "p5": round(float(s.quantile(0.05)), 2),
                "p25": round(float(s.quantile(0.25)), 2),
                "p50": round(float(s.quantile(0.50)), 2),
                "p75": round(float(s.quantile(0.75)), 2),
                "p95": round(float(s.quantile(0.95)), 2),
                "max": round(float(s.max()), 2),
            },
            name=name,
        )


# 模块级便捷函数（等价于默认参数的转换器）
_DEFAULT = ScoreConverter()


def p_to_score(p, pdo: int = 60, base_score: int = 600, base_odds: float = 20.0):
    """便捷函数：概率转分数。"""
    if (pdo, base_score, base_odds) == (60, 600, 20.0):
        return _DEFAULT.p_to_score(p)
    return ScoreConverter(pdo, base_score, base_odds).p_to_score(p)


def score_to_p(score, pdo: int = 60, base_score: int = 600, base_odds: float = 20.0):
    """便捷函数：分数转概率。"""
    if (pdo, base_score, base_odds) == (60, 600, 20.0):
        return _DEFAULT.score_to_p(score)
    return ScoreConverter(pdo, base_score, base_odds).score_to_p(score)


__all__ = ["ScoreConverter", "p_to_score", "score_to_p"]
