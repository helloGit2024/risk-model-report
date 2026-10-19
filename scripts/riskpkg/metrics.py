# -*- coding: utf-8 -*-
"""
风控原子指标层（RiskMetrics）
================================================
职责：提供无状态、纯函数、可独立测试的风控指标原子能力。
所有指标口径在此文件内锁死，全库唯一实现，禁止在别处重复造轮子。

【全局口径约定】（与 scorecardpy.woebin 实测一致，行业 P1 口径）
------------------------------------------------------------------
* 标签：y=1 为坏样本（bad），y=0 为好样本（good）
* 分数：模型分/prob 越大 → 风险越高（若业务反向，请调用方自行取负）
* WOE  = ln( Bad占比 / Good占比 )       —— 分数越高风险越高时 WOE 越大
* IV   = Σ( Bad占比 - Good占比 ) × WOE  —— 值域 [0, +inf)
* PSI  = Σ( 实际占比 - 基准占比 ) × ln( 实际占比 / 基准占比 )
* KS   = max( TPR - FPR )
* Gini = 2 × AUC - 1
* Lift = 分片坏率 / 整体坏率

IV 分档（Siddiqi 标准）
------------------------------------------------------------------
< 0.02 无预测力 | 0.02~0.1 弱 | 0.1~0.3 中等 | 0.3~0.5 强 | > 0.5 强到可疑（查泄漏/过拟合）

PSI 分档（行业惯例）
------------------------------------------------------------------
< 0.1 稳定 | 0.1~0.25 需观察 | > 0.25 显著漂移
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

# 常量：平滑系数（避免 0 占比导致 ln 发散）
_EPS = 1e-6
# PSI 默认分箱数
_DEFAULT_BINS = 10
# 样本量过小时拒绝计算（占比无统计意义）
_MIN_SAMPLE = 30

# IV 分档边界
IV_GRADES: List[Tuple[float, float, str]] = [
    (0.5, float("inf"), "强-可疑"),
    (0.3, 0.5, "强"),
    (0.1, 0.3, "中"),
    (0.02, 0.1, "弱"),
    (0.0, 0.02, "无预测力"),
]

# PSI 分档边界
PSI_GRADES: List[Tuple[float, float, str]] = [
    (0.25, float("inf"), "显著漂移"),
    (0.1, 0.25, "需观察"),
    (0.0, 0.1, "稳定"),
]

# KS 分档（区分度）
KS_GRADES: List[Tuple[float, float, str]] = [
    (0.4, float("inf"), "优秀"),
    (0.3, 0.4, "良好"),
    (0.2, 0.3, "可用"),
    (0.0, 0.2, "偏弱"),
]


def safe_div(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default: float = np.nan,
    eps: float = _EPS,
) -> Union[float, np.ndarray, pd.Series]:
    """
    安全除法：分母为 0 / NaN / inf 时返回 default。

    统一处理标量、ndarray、Series 三种形态（替代原 safe_div 的 ~80 行广播逻辑）。
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            (den == 0) | np.isnan(den) | np.isinf(den) | np.isnan(num),
            np.nan,
            num / np.where(den == 0, eps, den),
        )
    out = np.where(np.isinf(out), np.nan, out)

    if np.isscalar(numerator) and np.isscalar(denominator):
        val = float(out)
        return default if math.isnan(val) else val
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        return pd.Series(out, index=getattr(numerator, "index", getattr(denominator, "index", None))).fillna(default)
    return np.where(np.isnan(out), default, out)


class RiskMetrics:
    """风控指标原子层：全静态方法，无内部状态，可独立单元测试。"""

    # ------------------------------------------------------------------
    # 一、区分度指标
    # ------------------------------------------------------------------
    @staticmethod
    def calc_ks(y_true, y_score) -> float:
        """
        KS = max(TPR - FPR)。

        单类别标签（全好/全坏）无法计算，返回 nan 并记录告警。
        """
        y = np.asarray(y_true)
        s = np.asarray(y_score, dtype=float)
        mask = ~(pd.isna(y) | pd.isna(s))
        y, s = y[mask], s[mask]
        if y.size == 0 or np.unique(y).size < 2:
            logger.warning("calc_ks：标签仅单一类别或样本为空，返回 NaN")
            return np.nan
        try:
            fpr, tpr, _ = roc_curve(y, s)
            return float(round(max(tpr - fpr), 4))
        except Exception as exc:  # pragma: no cover
            logger.warning("calc_ks 异常：%s", exc)
            return np.nan

    @staticmethod
    def calc_auc(y_true, y_score) -> float:
        """AUC（ROC 曲线下面积）。"""
        y = np.asarray(y_true)
        s = np.asarray(y_score, dtype=float)
        mask = ~(pd.isna(y) | pd.isna(s))
        y, s = y[mask], s[mask]
        if y.size == 0 or np.unique(y).size < 2:
            logger.warning("calc_auc：标签仅单一类别或样本为空，返回 NaN")
            return np.nan
        try:
            return float(round(roc_auc_score(y, s), 4))
        except Exception as exc:  # pragma: no cover
            logger.warning("calc_auc 异常：%s", exc)
            return np.nan

    @staticmethod
    def calc_gini(y_true, y_score) -> float:
        """Gini = 2 × AUC - 1（银行必报指标；等于 AR 准确率比率）。"""
        auc = RiskMetrics.calc_auc(y_true, y_score)
        return np.nan if pd.isna(auc) else float(round(2 * auc - 1, 4))

    @staticmethod
    def ks_cutoff(y_true, y_score) -> Dict[str, float]:
        """
        KS 最优切点及该点完整混淆指标（策略阈值的直接依据）。

        Returns
        -------
        dict: threshold / ks / tpr / fpr / precision / recall / youden_j
        """
        y = np.asarray(y_true)
        s = np.asarray(y_score, dtype=float)
        mask = ~(pd.isna(y) | pd.isna(s))
        y, s = y[mask], s[mask]
        empty = {
            "threshold": np.nan, "ks": np.nan, "tpr": np.nan, "fpr": np.nan,
            "precision": np.nan, "recall": np.nan, "youden_j": np.nan,
        }
        if y.size == 0 or np.unique(y).size < 2:
            return empty

        try:
            fpr, tpr, thr = roc_curve(y, s)
        except Exception as exc:  # pragma: no cover
            logger.warning("ks_cutoff 异常：%s", exc)
            return empty

        ks_arr = tpr - fpr
        i = int(np.argmax(ks_arr))
        threshold = float(thr[i])

        pred_pos = s >= threshold
        tp = float(np.sum(pred_pos & (y == 1)))
        fp = float(np.sum(pred_pos & (y == 0)))
        fn = float(np.sum(~pred_pos & (y == 1)))
        tn = float(np.sum(~pred_pos & (y == 0)))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        return {
            "threshold": round(threshold, 6),
            "ks": round(float(ks_arr[i]), 4),
            "tpr": round(float(tpr[i]), 4),
            "fpr": round(float(fpr[i]), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "youden_j": round(float(tpr[i] - fpr[i]), 4),
        }

    @staticmethod
    def overfit_gap(ks_train: float, ks_test: float, auc_train: float, auc_test: float) -> Dict[str, float]:
        """
        过拟合缺口：训练集与测试集区分度衰减（产线红线：KS 衰减 < 0.05）。

        Returns
        -------
        dict: ks_gap / auc_gap / is_overfit(bool 0|1)
        """
        ks_gap = safe_div(ks_train - ks_test, ks_train) if not pd.isna(ks_train) else np.nan
        auc_gap = auc_train - auc_test if not (pd.isna(auc_train) or pd.isna(auc_test)) else np.nan
        abs_ks_gap = ks_train - ks_test if not (pd.isna(ks_train) or pd.isna(ks_test)) else np.nan
        return {
            "ks_gap_abs": round(float(abs_ks_gap), 4) if not pd.isna(abs_ks_gap) else np.nan,
            "ks_gap_rate": round(float(ks_gap), 4) if not pd.isna(ks_gap) else np.nan,
            "auc_gap": round(float(auc_gap), 4) if not pd.isna(auc_gap) else np.nan,
            "is_overfit": int((not pd.isna(abs_ks_gap)) and abs_ks_gap > 0.05),
        }

    # ------------------------------------------------------------------
    # 二、Lift / 捕获率（策略裁量核心）
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_index(y_score: np.ndarray, slice_pct: float, tail: bool) -> np.ndarray:
        """按分数排序取头/尾切片下标（向量化，不构建分箱表，避免无谓开销）。"""
        n = y_score.size
        k = max(1, int(round(n * slice_pct)))
        order = np.argsort(y_score, kind="mergesort")  # 升序
        return order[:k] if tail else order[::-1][:k]

    @staticmethod
    def calc_lift_slice(y_true, y_score, slice_pct: float = 0.1, tail: bool = False) -> float:
        """
        分片 Lift = 分片坏率 / 整体坏率。

        Parameters
        ----------
        slice_pct : 切片比例，0.1 表示取分数最高（或最低）10% 的样本
        tail      : False=头部（最高分，高风险）；True=尾部（最低分，低风险）
        """
        y = np.asarray(y_true)
        s = np.asarray(y_score, dtype=float)
        mask = ~(pd.isna(y) | pd.isna(s))
        y, s = y[mask], s[mask]
        if y.size == 0:
            return np.nan
        overall = float(np.mean(y))
        if overall <= 0:
            return np.nan
        idx = RiskMetrics._slice_index(s, slice_pct, tail)
        return float(round(float(np.mean(y[idx])) / overall, 4))

    @staticmethod
    def calc_capture_rate(y_true, y_score, top_pct: float = 0.1) -> float:
        """
        坏客户捕获率 = 头部 top_pct 高风险样本中命中的坏样本数 / 全部坏样本数。

        策略含义：拦截 10% 的申请能挡住多少比例的坏账。
        """
        y = np.asarray(y_true)
        s = np.asarray(y_score, dtype=float)
        mask = ~(pd.isna(y) | pd.isna(s))
        y, s = y[mask], s[mask]
        if y.size == 0:
            return np.nan
        total_bad = float(np.sum(y == 1))
        if total_bad <= 0:
            return np.nan
        idx = RiskMetrics._slice_index(s, top_pct, tail=False)
        return float(round(float(np.sum(y[idx] == 1)) / total_bad, 4))

    # ------------------------------------------------------------------
    # 三、稳定性指标 PSI（全库唯一实现）
    # ------------------------------------------------------------------
    @staticmethod
    def psi_edges(base: Union[np.ndarray, pd.Series], bins: int = _DEFAULT_BINS,
                  method: str = "quantile") -> np.ndarray:
        """
        以基准样本生成分箱边界（供多次复用，避免重复 qcut，提升性能）。

        method: 'quantile' 等频（默认，行业惯例）| 'equal' 等距 | 'category' 类别型
        """
        arr = pd.Series(base).dropna()
        if arr.empty:
            return np.array([-np.inf, np.inf])

        is_numeric = pd.api.types.is_numeric_dtype(arr)
        if (not is_numeric) or method == "category" or arr.nunique() <= bins:
            # 类别型 / 取值数少于箱数：按唯一值分箱
            vals = np.sort(arr.unique())
            if is_numeric:
                edges = np.unique(np.concatenate([[-np.inf], vals.astype(float)]))
            else:
                return vals  # 类别型直接返回取值列表，交由调用方用 isin 处理
            return edges if edges.size >= 2 else np.array([-np.inf, np.inf])

        try:
            if method == "equal":
                _, edges = pd.cut(arr, bins=bins, retbins=True, duplicates="drop")
            else:
                _, edges = pd.qcut(arr, q=bins, retbins=True, duplicates="drop")
        except Exception:
            _, edges = pd.qcut(arr.rank(method="first"), q=bins, retbins=True, duplicates="drop")

        edges = np.unique(np.asarray(edges, dtype=float))
        if edges.size < 2:
            return np.array([-np.inf, np.inf])
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

    @staticmethod
    def _dist(arr: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """按边界计算占比分布；空箱以 0 保留（关键修复：原实现 inner join 会静默丢箱）。"""
        if isinstance(edges, np.ndarray) and edges.dtype.kind in ("O", "U", "S"):
            # 类别型
            counts = np.array([np.sum(arr == v) for v in edges], dtype=float)
        else:
            idx = np.searchsorted(edges, arr, side="right") - 1
            idx = np.clip(idx, 0, len(edges) - 2)
            counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        total = counts.sum()
        if total <= 0:
            return np.full(counts.shape, np.nan)
        return counts / total

    @staticmethod
    def calc_psi(base, curr, bins: int = _DEFAULT_BINS, method: str = "quantile",
                 edges: Optional[np.ndarray] = None, min_sample: int = _MIN_SAMPLE) -> float:
        """
        PSI = Σ(实际占比 - 基准占比) × ln(实际占比 / 基准占比)。

        修复要点（相对旧实现）
        ----------------------
        1. 空箱以 eps 保留参与计算 —— 旧版 `Cal_Psi` 用 inner join 丢箱，导致 PSI 系统性低估
        2. 支持类别型特征 —— 旧版遇字符串特征直接崩溃
        3. 支持外部传入 edges 复用边界 —— 逐月 PSI 场景性能提升显著
        """
        b = pd.Series(base).dropna()
        c = pd.Series(curr).dropna()
        if b.empty or c.empty or len(b) < min_sample or len(c) < min_sample:
            logger.debug("calc_psi：样本不足（base=%d, curr=%d），返回 NaN", len(b), len(c))
            return np.nan

        if edges is None:
            edges = RiskMetrics.psi_edges(b, bins=bins, method=method)

        b_arr = b.values if not isinstance(edges, np.ndarray) or edges.dtype.kind not in ("O", "U", "S") else b.values
        c_arr = c.values

        p = RiskMetrics._dist(b_arr, edges)
        q = RiskMetrics._dist(c_arr, edges)
        if np.any(np.isnan(p)) or np.any(np.isnan(q)):
            return np.nan

        # 平滑：避免出现 0 占比导致 ln 发散
        p = np.where(p <= 0, _EPS, p)
        q = np.where(q <= 0, _EPS, q)
        psi = float(np.sum((q - p) * np.log(q / p)))
        return float(round(max(psi, 0.0), 6))

    @staticmethod
    def calc_psi_matrix(df: pd.DataFrame, prob: str, month_col: str = "month",
                        bins: int = _DEFAULT_BINS) -> pd.DataFrame:
        """
        月份 × 月份 PSI 热力矩阵（对称阵，对角线为 0）。

        边界复用：每个月只做一次 qcut，两两组合仅做 O(n) 的 searchsorted。
        """
        months = sorted(df[month_col].dropna().unique())
        edges_map, dist_map = {}, {}
        for m in months:
            arr = df.loc[df[month_col] == m, prob].dropna().values
            if arr.size == 0:
                continue
            e = RiskMetrics.psi_edges(arr, bins=bins)
            edges_map[m] = e
            dist_map[m] = arr

        mat = pd.DataFrame(np.nan, index=months, columns=months, dtype=float)
        for i, m1 in enumerate(months):
            for m2 in months[i:]:
                if m1 not in edges_map or m2 not in dist_map:
                    continue
                p = RiskMetrics._dist(dist_map[m1], edges_map[m1])
                q = RiskMetrics._dist(dist_map[m2], edges_map[m1])
                if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                    val = np.nan
                else:
                    p = np.where(p <= 0, _EPS, p)
                    q = np.where(q <= 0, _EPS, q)
                    val = round(max(float(np.sum((q - p) * np.log(q / p))), 0.0), 6)
                mat.loc[m1, m2] = val
                mat.loc[m2, m1] = val
        for m in months:
            mat.loc[m, m] = 0.0
        return mat

    # ------------------------------------------------------------------
    # 四、IV / WOE（统一口径：WOE = ln(Bad%/Good%)）
    # ------------------------------------------------------------------
    @staticmethod
    def calc_iv_woe(y_true, x, bins: int = _DEFAULT_BINS, method: str = "quantile",
                    edges: Optional[np.ndarray] = None) -> Tuple[float, pd.DataFrame]:
        """
        计算单特征 IV 与 WOE 明细表。

        Returns
        -------
        (iv, woe_table)
            woe_table 列：bin / CntRec / CntGood / CntBad / PctRec / PctGood / PctBad /
                          bad_rate / woe / iv
        """
        y = pd.Series(np.asarray(y_true)).reset_index(drop=True)
        xv = pd.Series(np.asarray(x)).reset_index(drop=True)
        mask = y.isin([0, 1]) & xv.notna()
        y, xv = y[mask].astype(int), xv[mask]

        if y.empty or y.nunique() < 2:
            return 0.0, pd.DataFrame()

        if edges is None:
            edges = RiskMetrics.psi_edges(xv, bins=bins, method=method)

        if isinstance(edges, np.ndarray) and edges.dtype.kind in ("O", "U", "S"):
            grp = pd.Series(pd.Categorical(xv.values, categories=edges))
        else:
            grp = pd.cut(xv, bins=edges, include_lowest=True)

        tab = pd.DataFrame({"bin": grp.astype(str), "y": y.values})
        agg = tab.groupby("bin", observed=True)["y"].agg(CntRec="count", CntBad="sum").reset_index()
        agg["CntGood"] = agg["CntRec"] - agg["CntBad"]

        total_bad = float(agg["CntBad"].sum())
        total_good = float(agg["CntGood"].sum())
        if total_bad <= 0 or total_good <= 0:
            return 0.0, agg

        agg["PctRec"] = agg["CntRec"] / agg["CntRec"].sum()
        agg["PctBad"] = agg["CntBad"] / total_bad
        agg["PctGood"] = agg["CntGood"] / total_good
        agg["bad_rate"] = agg["CntBad"] / agg["CntRec"].clip(lower=1)

        bad_p = agg["PctBad"].clip(lower=_EPS)
        good_p = agg["PctGood"].clip(lower=_EPS)
        agg["woe"] = np.log(bad_p / good_p)                      # WOE = ln(Bad%/Good%)
        agg["iv"] = (agg["PctBad"] - agg["PctGood"]) * agg["woe"]  # IV  = Σ(Bad%-Good%)×WOE

        return float(round(agg["iv"].sum(), 6)), agg

    @staticmethod
    def calc_iv(y_true, x, bins: int = _DEFAULT_BINS, method: str = "quantile",
                edges: Optional[np.ndarray] = None) -> float:
        """仅返回 IV 值。"""
        return RiskMetrics.calc_iv_woe(y_true, x, bins=bins, method=method, edges=edges)[0]

    @staticmethod
    def calc_feature_ks(y_true, x, bins: int = _DEFAULT_BINS,
                        edges: Optional[np.ndarray] = None) -> float:
        """
        特征 KS：各箱累计坏样本占比与累计好样本占比之差的最大值。
        """
        y = pd.Series(np.asarray(y_true)).reset_index(drop=True)
        xv = pd.Series(np.asarray(x)).reset_index(drop=True)
        mask = y.isin([0, 1]) & xv.notna()
        y, xv = y[mask].astype(int), xv[mask]
        if y.empty or y.nunique() < 2:
            return np.nan

        if edges is None:
            edges = RiskMetrics.psi_edges(xv, bins=bins)
        if isinstance(edges, np.ndarray) and edges.dtype.kind in ("O", "U", "S"):
            grp = pd.Series(pd.Categorical(xv.values, categories=edges))
        else:
            grp = pd.cut(xv, bins=edges, include_lowest=True)

        tab = pd.DataFrame({"bin": grp.astype(str), "y": y.values})
        agg = tab.groupby("bin", observed=True)["y"].agg(CntRec="count", CntBad="sum").reset_index()
        agg["CntGood"] = agg["CntRec"] - agg["CntBad"]
        tb, tg = agg["CntBad"].sum(), agg["CntGood"].sum()
        if tb <= 0 or tg <= 0:
            return np.nan
        agg = agg.sort_values("bin")
        cum_bad = (agg["CntBad"] / tb).cumsum()
        cum_good = (agg["CntGood"] / tg).cumsum()
        return float(round(float((cum_bad - cum_good).abs().max()), 4))

    @staticmethod
    def calc_monotonicity(bad_rates: Sequence[float]) -> Dict[str, Union[bool, str, float]]:
        """
        分箱坏率单调性检验（评分卡可解释性要求）。

        Returns
        -------
        dict: is_monotonic / direction(递增|递减|非单调) / violations(违反点数)
        """
        arr = pd.Series(list(bad_rates)).dropna()
        if arr.size < 2:
            return {"is_monotonic": False, "direction": "样本不足", "violations": np.nan}
        diff = np.diff(arr.values)
        inc = int(np.sum(diff > 0))
        dec = int(np.sum(diff < 0))
        total = int(diff.size)
        if inc == total:
            return {"is_monotonic": True, "direction": "递增", "violations": 0}
        if dec == total:
            return {"is_monotonic": True, "direction": "递减", "violations": 0}
        return {"is_monotonic": False, "direction": "非单调", "violations": int(min(inc, dec))}

    # ------------------------------------------------------------------
    # 五、校准度（Calibration）
    # ------------------------------------------------------------------
    @staticmethod
    def calc_brier(y_true, y_score) -> float:
        """Brier Score = mean((p - y)^2)，越小校准越好。"""
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_score, dtype=float)
        mask = ~(np.isnan(y) | np.isnan(p))
        y, p = y[mask], p[mask]
        if y.size == 0:
            return np.nan
        return float(round(float(np.mean((p - y) ** 2)), 6))

    @staticmethod
    def hosmer_lemeshow(y_true, y_score, groups: int = 10) -> Tuple[float, float]:
        """
        Hosmer-Lemeshow 拟合优度检验（按预测概率分位的校准检验）。

        Returns
        -------
        (chi2, p_value)  p_value 越大校准越好（>0.05 通常认为校准良好）
        """
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_score, dtype=float)
        mask = ~(np.isnan(y) | np.isnan(p))
        y, p = y[mask], p[mask]
        if y.size == 0 or np.unique(y).size < 2:
            return np.nan, np.nan

        try:
            qq = pd.qcut(pd.Series(p).rank(method="first"), q=groups, labels=False, duplicates="drop")
        except Exception:
            return np.nan, np.nan

        df = pd.DataFrame({"y": y, "p": p, "g": qq.values})
        grp = df.groupby("g").agg(obs=("y", "sum"), n=("y", "size"), exp=("p", "sum"))
        grp["obs0"] = grp["n"] - grp["obs"]
        grp["exp0"] = grp["n"] - grp["exp"]
        chi2 = float(
            (((grp["obs"] - grp["exp"]) ** 2) / grp["exp"].clip(lower=_EPS)
             + ((grp["obs0"] - grp["exp0"]) ** 2) / grp["exp0"].clip(lower=_EPS)).sum()
        )
        try:
            from scipy import stats
            p_value = float(1 - stats.chi2.cdf(chi2, df=max(groups - 2, 1)))
        except Exception:
            p_value = np.nan
        return round(chi2, 4), round(p_value, 4) if not pd.isna(p_value) else np.nan

    @staticmethod
    def calibration_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
        """
        校准表：各分箱的 预期坏率(预测均值) vs 实际坏率，用于绘制校准曲线。
        """
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_score, dtype=float)
        mask = ~(np.isnan(y) | np.isnan(p))
        y, p = y[mask], p[mask]
        if y.size == 0:
            return pd.DataFrame()
        edges = RiskMetrics.psi_edges(p, bins=bins)
        grp = pd.cut(pd.Series(p), bins=edges, include_lowest=True)
        out = pd.DataFrame({"bin": grp.astype(str), "y": y, "p": p})
        res = out.groupby("bin", observed=True).agg(
            CntRec=("y", "size"), CntBad=("y", "sum"),
            expected_pd=("p", "mean"),
        ).reset_index()
        res["actual_pd"] = res["CntBad"] / res["CntRec"].clip(lower=1)
        res["gap"] = res["actual_pd"] - res["expected_pd"]
        return res

    # ------------------------------------------------------------------
    # 六、覆盖率 / 缺失率
    # ------------------------------------------------------------------
    @staticmethod
    def calc_coverage(series: pd.Series, invalid_values: Optional[List] = None) -> float:
        """
        覆盖率 = 1 - 缺失率。

        修复：同时处理字符串型与数值型的无效值（旧实现 isin([-1,-999]) 对 "-1" 不命中）。
        """
        if series is None or len(series) == 0:
            return np.nan
        s = series
        na = s.isna()
        if invalid_values:
            str_vals = {str(v) for v in invalid_values}
            num_vals = [v for v in invalid_values if isinstance(v, (int, float, np.number))]
            as_str = s.astype(str)
            na = na | as_str.isin(str_vals)
            if num_vals and pd.api.types.is_numeric_dtype(s):
                na = na | s.isin(num_vals)
        return float(round(1 - float(na.mean()), 4))

    @staticmethod
    def calc_missing_rate(series: pd.Series) -> float:
        """缺失率（含 NaN）。"""
        if series is None or len(series) == 0:
            return np.nan
        return float(round(float(series.isna().mean()), 4))

    # ------------------------------------------------------------------
    # 七、分档评级
    # ------------------------------------------------------------------
    @staticmethod
    def grade(value: float, rules: List[Tuple[float, float, str]], default: str = "未知") -> str:
        """按分档规则给指标打标签。rules 须由大到小排列。"""
        if value is None or pd.isna(value):
            return default
        for lo, hi, label in rules:
            if lo <= value < hi:
                return label
        return default

    @staticmethod
    def iv_grade(iv: float) -> str:
        return RiskMetrics.grade(iv, IV_GRADES)

    @staticmethod
    def psi_grade(psi: float) -> str:
        return RiskMetrics.grade(psi, PSI_GRADES)

    @staticmethod
    def ks_grade(ks: float) -> str:
        return RiskMetrics.grade(ks, KS_GRADES)

    # ------------------------------------------------------------------
    # 八、一次性汇总（供分组评估复用，保证各分组口径完全一致）
    # ------------------------------------------------------------------
    @staticmethod
    def summarize(y_true, y_score, psi_base: Optional[np.ndarray] = None,
                  lift_points: Optional[List[float]] = None,
                  capture_points: Optional[List[float]] = None,
                  with_cutoff: bool = True) -> Dict[str, float]:
        """
        单组样本的指标全量汇总 —— 分组评估的唯一出口，杜绝各分组口径漂移。

        Parameters
        ----------
        psi_base    : PSI 基准分布（通常为 train 的模型分）；None 则不计算 PSI
        lift_points : 需要计算的头部 lift 切片比例，默认 [0.01, 0.05, 0.1, 0.2]
        capture_points : 捕获率切片比例，默认 [0.1, 0.2]
        """
        lift_points = lift_points or [0.01, 0.05, 0.1, 0.2]
        capture_points = capture_points or [0.1, 0.2]

        y = np.asarray(y_true)
        s = np.asarray(y_score, dtype=float)
        mask = ~(pd.isna(y) | pd.isna(s))
        y, s = y[mask], s[mask]

        res: Dict[str, float] = {
            "CntRec": int(y.size),
            "CntBad": int(np.sum(y == 1)) if y.size else 0,
            "bad_rate": float(round(float(np.mean(y)), 6)) if y.size else np.nan,
        }
        res["ks"] = RiskMetrics.calc_ks(y, s)
        res["auc"] = RiskMetrics.calc_auc(y, s)
        res["gini"] = RiskMetrics.calc_gini(y, s)
        res["brier"] = RiskMetrics.calc_brier(y, s)

        for pt in lift_points:
            res["lift_head_{:.0f}pct".format(pt * 100)] = RiskMetrics.calc_lift_slice(y, s, pt, tail=False)
        res["lift_tail_10pct"] = RiskMetrics.calc_lift_slice(y, s, 0.1, tail=True)

        for pt in capture_points:
            res["capture_{:.0f}pct".format(pt * 100)] = RiskMetrics.calc_capture_rate(y, s, pt)

        if psi_base is not None:
            res["psi"] = RiskMetrics.calc_psi(psi_base, s)
        else:
            res["psi"] = np.nan

        if with_cutoff:
            cut = RiskMetrics.ks_cutoff(y, s)
            res["ks_threshold"] = cut["threshold"]
            res["precision"] = cut["precision"]
            res["recall"] = cut["recall"]
            res["fpr"] = cut["fpr"]

        return res


__all__ = ["RiskMetrics", "safe_div", "IV_GRADES", "PSI_GRADES", "KS_GRADES"]
