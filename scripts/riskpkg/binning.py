# -*- coding: utf-8 -*-
"""
分箱层（ScorecardBinner）
================================================
全库唯一的分箱实现，合并旧代码中的两套重复实现（Sub_box / FeatureBinner），
并修复以下缺陷：

1. `smbin_lx`      —— StratifiedKFold 未导入（NameError）
2. `smbin_lx_qcut` —— num_bins 参数被忽略，内部写死 q=10
3. `smbin_cu`      —— NaN 样本计入 CntRec 但不计入好/坏，导致箱内坏率分母虚高
4. `FeatureBinner.transform` —— 硬编码 "y" 列名，标签列改名即崩
5. `smbin_lx_qcut(cutpoints=...)` —— 参数不存在（TypeError）

设计原则
------------------------------------------------------------------
* `fit`   只在训练样本上学习边界，绝不修改传入的 DataFrame（无副作用）
* `transform` 复用 `fit` 学到的边界 —— 保证 train/test/oot 同口径、无信息泄露
* 缺失值单独成箱（缺失本身有预测力，且可计算其 IV 贡献）
* WOE 口径：ln(Bad% / Good%)，与 scorecardpy 一致
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .metrics import RiskMetrics, _EPS

logger = logging.getLogger(__name__)

try:  # scorecardpy 为可选依赖，未安装时自动降级
    import scorecardpy as sc
    _HAS_SC = True
except ImportError:  # pragma: no cover
    sc = None
    _HAS_SC = False


class ScorecardBinner:
    """
    评分卡分箱器（fit/transform 模式）。

    Examples
    --------
    >>> b = ScorecardBinner(method="tree", max_bin=5)
    >>> b.fit(train_df, "blzVar43568", "y")
    >>> table = b.transform(oot_df)          # 同边界的分布对比表
    >>> woe    = b.transform_woe(oot_df)     # WOE 编码列
    """

    MISSING_LABEL = "missing"

    def __init__(self, method: str = "tree", max_bin: int = 5,
                 min_samples_leaf: float = 0.05, custom_cuts: Optional[Sequence[float]] = None,
                 missing_as_bin: bool = True):
        """
        Parameters
        ----------
        method : 'tree'(决策树) | 'chimerge'(卡方,需scorecardpy) | 'qcut'(等频) | 'cut'(等距)
        max_bin : 最大箱数
        min_samples_leaf : 每箱最小样本占比（仅 tree 方法）
        custom_cuts : 自定义切点，优先级最高
        missing_as_bin : 缺失值是否单独成箱
        """
        self.method = method
        self.max_bin = max_bin
        self.min_samples_leaf = min_samples_leaf
        self.custom_cuts = list(custom_cuts) if custom_cuts else None
        self.missing_as_bin = missing_as_bin

        # fit 产物
        self.feat: Optional[str] = None
        self.y_col: Optional[str] = None
        self.cutpoints: Optional[List[float]] = None
        self.woe_map: Dict[str, float] = {}
        self.iv: float = 0.0
        self.edges_: Optional[np.ndarray] = None
        self.is_categorical_: bool = False

    # ------------------------------------------------------------------
    # 边界学习
    # ------------------------------------------------------------------
    def _learn_cutpoints(self, x: pd.Series, y: pd.Series) -> List[float]:
        """学习切点（不含首尾 ±inf）。"""
        if self.custom_cuts:
            return sorted(float(c) for c in self.custom_cuts)

        method = self.method
        if method in ("tree", "chimerge") and _HAS_SC:
            try:
                return self._learn_by_scorecardpy(x, y, method)
            except Exception as exc:
                logger.warning("%s 分箱失败(%s)，降级为等频分箱", method, exc)

        if method == "tree":
            return self._learn_by_decision_tree(x, y)

        # qcut / cut / 降级分支
        try:
            if method == "cut":
                _, edges = pd.cut(x, bins=self.max_bin, retbins=True, duplicates="drop")
            else:
                _, edges = pd.qcut(x, q=self.max_bin, retbins=True, duplicates="drop")
            edges = np.unique(np.asarray(edges, dtype=float))
            return list(edges[1:-1])  # 去掉首尾边界，保留中间切点
        except Exception:
            ranked = x.rank(method="first")
            _, edges = pd.qcut(ranked, q=self.max_bin, retbins=True, duplicates="drop")
            return [float(x.loc[ranked <= e].max()) for e in np.asarray(edges)[1:-1]]

    def _learn_by_scorecardpy(self, x: pd.Series, y: pd.Series, method: str) -> List[float]:
        """调用 scorecardpy.woebin 学习边界。"""
        df = pd.DataFrame({"x": x.values, "y": y.values})
        res = sc.woebin(df, y="y", x=["x"], method=method,
                        stop_limit=0.05, count_distr_limit=self.min_samples_leaf)
        tbl = res["x"]
        breaks = pd.to_numeric(tbl["breaks"], errors="coerce").dropna().tolist()
        return sorted(float(b) for b in breaks)

    def _learn_by_decision_tree(self, x: pd.Series, y: pd.Series) -> List[float]:
        """
        决策树最优分箱（修复旧版 StratifiedKFold 缺失与 n_splits<2 崩溃）。
        """
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.tree import DecisionTreeClassifier

        n = len(y)
        if n < 50 or y.nunique() < 2:
            return self._fallback_qcut(x)

        leaf_cnt = max(10, int(n * self.min_samples_leaf))
        n_splits = max(2, min(5, n // max(leaf_cnt, 1)))
        n_splits = min(n_splits, int(y.sum()), int((y == 0).sum()))
        if n_splits < 2:
            return self._fallback_qcut(x)

        X = x.values.reshape(-1, 1)
        param_grid = {
            "max_leaf_nodes": list(range(2, self.max_bin + 1)),
            "min_samples_leaf": [leaf_cnt],
        }
        try:
            grid = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                scoring="roc_auc",
                n_jobs=1,
            )
            grid.fit(X, y.values)
            clf = grid.best_estimator_
            thresholds = clf.tree_.threshold
            cuts = sorted({float(t) for t in thresholds if t != -2})
            cuts = [c for c in cuts if float(x.min()) < c < float(x.max())]
            return cuts[: self.max_bin - 1] if cuts else self._fallback_qcut(x)
        except Exception as exc:
            logger.warning("决策树分箱失败(%s)，降级为等频分箱", exc)
            return self._fallback_qcut(x)

    def _fallback_qcut(self, x: pd.Series) -> List[float]:
        try:
            _, edges = pd.qcut(x, q=self.max_bin, retbins=True, duplicates="drop")
            return list(np.unique(np.asarray(edges, dtype=float))[1:-1])
        except Exception:
            return []

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, feat: str, y: str) -> "ScorecardBinner":
        """
        在给定数据上学习分箱边界与 WOE 映射。

        注意：**不会修改传入的 df**（旧版 get_fx 会写入 'cut' 列造成污染）。
        """
        self.feat, self.y_col = feat, y
        sub = df[[feat, y]].copy()
        sub = sub[sub[y].isin([0, 1])]
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce") if not self._is_cat(sub[feat]) else sub[feat]

        self.is_categorical_ = self._is_cat(sub[feat])

        if sub.empty or sub[y].nunique() < 2:
            logger.warning("fit：特征 %s 有效样本不足或标签单一", feat)
            self.cutpoints, self.woe_map, self.iv = [], {}, 0.0
            return self

        if self.is_categorical_:
            self.cutpoints = []
            self.edges_ = np.array(sorted(sub[feat].dropna().unique().astype(str)))
        else:
            valid = sub[sub[feat].notna()]
            self.cutpoints = self._learn_cutpoints(valid[feat], valid[y]) if not valid.empty else []
            edges = [-np.inf] + list(self.cutpoints) + [np.inf]
            self.edges_ = np.unique(np.asarray(edges, dtype=float))
            if self.edges_.size < 2:
                self.edges_ = np.array([-np.inf, np.inf])

        # 以训练样本为准计算 WOE 与 IV
        tbl = self._build_table(sub, feat, y)
        if not tbl.empty:
            self.iv = float(round(tbl["iv"].sum(), 6))
            self.woe_map = dict(zip(tbl["bin"].astype(str), tbl["woe"]))
        return self

    @staticmethod
    def _is_cat(s: pd.Series) -> bool:
        return (not pd.api.types.is_numeric_dtype(s)) or s.nunique(dropna=True) <= 2 and not pd.api.types.is_float_dtype(s)

    def _assign_bin(self, series: pd.Series) -> pd.Series:
        """按已学到的边界分配分箱标签（含缺失单独成箱）。"""
        s = series
        if self.is_categorical_:
            out = s.astype(str)
            out = out.where(s.notna(), self.MISSING_LABEL)
            return out

        s_num = pd.to_numeric(s, errors="coerce")
        labels = pd.cut(s_num, bins=self.edges_, include_lowest=True, right=True)
        out = pd.Series(labels.astype(str), index=s.index)
        if self.missing_as_bin:
            out = out.where(s_num.notna(), self.MISSING_LABEL)
        return out

    def _build_table(self, df: pd.DataFrame, feat: str, y: str) -> pd.DataFrame:
        """构建单数据集的分布/IV/WOE 明细表（含 Total 行）。"""
        sub = df[[feat, y]].copy()
        sub = sub[sub[y].isin([0, 1])]
        if sub.empty:
            return pd.DataFrame()

        bins = self._assign_bin(sub[feat])
        tab = pd.DataFrame({"bin": bins.values, "y": sub[y].values})
        agg = tab.groupby("bin", observed=True)["y"].agg(CntRec="count", CntBad="sum").reset_index()
        agg["CntGood"] = agg["CntRec"] - agg["CntBad"]

        total_bad = float(agg["CntBad"].sum())
        total_good = float(agg["CntGood"].sum())
        total_rec = float(agg["CntRec"].sum())
        if total_bad <= 0 or total_good <= 0:
            agg["PctRec"] = agg["CntRec"] / max(total_rec, 1)
            agg["PctBad"] = agg["PctGood"] = np.nan
            agg["bad_rate"] = agg["CntBad"] / agg["CntRec"].clip(lower=1)
            agg["woe"] = np.nan
            agg["iv"] = 0.0
            agg["lift"] = np.nan
        else:
            agg["PctRec"] = agg["CntRec"] / total_rec
            agg["PctGood"] = agg["CntGood"] / total_good
            agg["PctBad"] = agg["CntBad"] / total_bad
            agg["bad_rate"] = agg["CntBad"] / agg["CntRec"].clip(lower=1)
            bad_p = agg["PctBad"].clip(lower=_EPS)
            good_p = agg["PctGood"].clip(lower=_EPS)
            agg["woe"] = np.log(bad_p / good_p)
            agg["iv"] = (agg["PctBad"] - agg["PctGood"]) * agg["woe"]
            agg["lift"] = agg["bad_rate"] / (total_bad / total_rec)

        agg["total_iv"] = round(agg["iv"].sum(), 6)
        agg = agg.sort_values("bin").reset_index(drop=True)

        total_row = {
            "bin": "Total",
            "CntRec": int(agg["CntRec"].sum()),
            "CntBad": int(agg["CntBad"].sum()),
            "CntGood": int(agg["CntGood"].sum()),
            "PctRec": 1.0,
            "PctGood": 1.0,
            "PctBad": 1.0,
            "bad_rate": agg["CntBad"].sum() / max(agg["CntRec"].sum(), 1),
            "woe": np.nan,
            "iv": agg["iv"].sum(),
            "lift": 1.0,
            "total_iv": round(agg["iv"].sum(), 6),
        }
        return pd.concat([agg, pd.DataFrame([total_row])], ignore_index=True)

    def transform(self, df: pd.DataFrame, feat: Optional[str] = None,
                  y: Optional[str] = None) -> pd.DataFrame:
        """
        用 fit 学到的边界，在任意数据集（test/oot）上生成分布对比表。

        修复：不再硬编码 "y" 列名（旧 FeatureBinner.transform 的 bug）。
        """
        feat = feat or self.feat
        y = y or self.y_col
        if feat is None or y is None or y not in df.columns:
            raise ValueError("transform 需要指定 feat 与 y（y 列必须存在于 df 中）")
        return self._build_table(df, feat, y)

    def transform_woe(self, df: pd.DataFrame, feat: Optional[str] = None) -> pd.Series:
        """输出 WOE 编码列（用 fit 学到的 WOE 映射，保证跨数据集一致）。"""
        feat = feat or self.feat
        bins = self._assign_bin(df[feat])
        woe = bins.astype(str).map(self.woe_map)
        return pd.Series(woe.values, index=df.index, name=f"{feat}_woe")

    def fit_transform(self, df: pd.DataFrame, feat: str, y: str) -> pd.DataFrame:
        return self.fit(df, feat, y).transform(df, feat, y)

    # ------------------------------------------------------------------
    @classmethod
    def batch(cls, train: pd.DataFrame, test: pd.DataFrame, feats: Sequence[str],
              y: str, method: str = "tree", max_bin: int = 5,
              min_samples_leaf: float = 0.05) -> Dict[str, pd.DataFrame]:
        """
        批量分箱：train 学边界 → test 复用边界（无信息泄露）。

        Returns
        -------
        dict: {feat: 合并后的 train/test 分布对比表}
        """
        out: Dict[str, pd.DataFrame] = {}
        for f in feats:
            if f not in train.columns or f not in test.columns:
                logger.warning("batch：特征 %s 缺失，跳过", f)
                continue
            try:
                b = cls(method=method, max_bin=max_bin, min_samples_leaf=min_samples_leaf)
                tr = b.fit(train, f, y).transform(train, f, y)
                te = b.transform(test, f, y)
                tr = tr.add_suffix("_train")
                te = te.add_suffix("_test")
                out[f] = pd.concat([tr, te], axis=1)
            except Exception as exc:
                logger.warning("batch：特征 %s 分箱失败：%s", f, exc)
        return out


__all__ = ["ScorecardBinner"]
