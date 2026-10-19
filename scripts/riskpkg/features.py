# -*- coding: utf-8 -*-
"""
特征层（FeatureAnalyzer）
================================================
特征全景分析 + 单变量分析，对齐《特征全景分析表》的 train/test/oot 三套结构。

修复的旧缺陷
------------------------------------------------------------------
1. `build_feature_summary` 只有 train/oot 两套，缺 test —— 已补齐三套
2. `calc_psi(base_arr=..., curr_arr=...)` 参数名错误（应为 test_arr）导致 TypeError
3. `get_fx` 直接给传入的 df 写 'cut' 列 —— 污染调用方数据，已消除副作用
4. `_calc_feature_statsV1` 引用未定义的 psi 变量 —— UnboundLocalError
5. `univar_analysis_df` 的 `df.replace(-1, np.nan, inplace=True)` 就地污染原始数据，
   且 -1 常是有效业务编码（如"无卡"），已改为可配置的无效值处理
6. `univar_analysis_df` 按**位置**拼接 psi 列表，train/test 箱数不一致时错位 —— 改为按 bin 值对齐
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .binning import ScorecardBinner
from .metrics import RiskMetrics
from .stability import StabilityMonitor

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """特征分析器：特征全景 + 单变量分析 + 共线性。"""

    def __init__(self, y_col: str, data_flag_col: str = "data_flag",
                 month_col: str = "month", psi_threshold: float = 0.1,
                 psi_alert: float = 0.25, invalid_values: Optional[List] = None):
        self.y_col = y_col
        self.data_flag_col = data_flag_col
        self.month_col = month_col
        self.invalid_values = invalid_values  # 视为缺失的业务编码，如 [-1, -999]
        self.monitor = StabilityMonitor(psi_threshold=psi_threshold, psi_alert_threshold=psi_alert)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def load_importance(path: str, top_n: Optional[int] = None) -> pd.DataFrame:
        """读取特征重要性表（兼容 feature/importance_gain/importance_split 等列名）。"""
        df = pd.read_csv(path)
        if "feature" not in df.columns:
            first = df.columns[0]
            df = df.rename(columns={first: "feature"})
        if "importance" not in df.columns:
            cand = [c for c in df.columns if "gain" in str(c).lower()]
            if cand:
                df = df.rename(columns={cand[0]: "importance"})
            else:
                num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num:
                    df = df.rename(columns={num[0]: "importance"})
        df = df[["feature", "importance"]].copy()
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n) if top_n else df

    def _clean(self, df: pd.DataFrame, feats: Sequence[str]) -> pd.DataFrame:
        """
        复制数据并把业务无效值转为 NaN（**不修改原 df**）。

        修复：旧版 `univar_analysis_df` 用 `df.replace(-1, np.nan, inplace=True)`
        就地污染调用方数据，且 -1 可能是"无记录"这类有效业务编码。
        """
        sub = df[list(feats) + [self.y_col]].copy()
        if self.invalid_values:
            sub[list(feats)] = sub[list(feats)].replace(self.invalid_values, np.nan)
        return sub

    # ------------------------------------------------------------------
    # 一、特征全景（train / test / oot 三套）
    # ------------------------------------------------------------------
    def build_summary(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                      df_oot: Optional[pd.DataFrame], feats: Sequence[str],
                      model=None, importance: Optional[pd.DataFrame] = None,
                      bins: int = 10) -> pd.DataFrame:
        """
        特征全景表，列结构与《特征全景分析表》对齐：

        变量名称 | 入模重要性 | 覆盖率_train | IV_train | KS_train |
        覆盖率_test | IV_test | KS_test | 覆盖率_oot | IV_oot | KS_oot | PSI_oot

        要点：**以 train 的分箱边界统一计算三套 IV/KS**，保证跨数据集可比
        （旧版各数据集各自分箱，IV 不可比）。
        """
        imp_map: Dict[str, float] = {}
        if importance is not None and "feature" in importance.columns:
            col = "importance" if "importance" in importance.columns else importance.columns[-1]
            imp_map = dict(zip(importance["feature"], importance[col]))
        elif model is not None and hasattr(model, "feature_name_"):
            try:
                vals = model.feature_importances_
                imp_map = dict(zip(model.feature_name_, vals))
            except Exception as exc:
                logger.warning("从模型读取重要性失败：%s", exc)

        feats = [f for f in feats if f in df_train.columns]
        datasets = {"train": df_train, "test": df_test, "oot": df_oot}

        # 边界统一以 train 为准；直接用原始 series，由 psi_edges 自行判别
        # 数值型（等频/等距）与类别型（取值分箱）
        # 修复：旧写法先 to_numeric 会把字符串特征全转成 NaN，导致混合类型数据崩溃
        edges_map: Dict[str, np.ndarray] = {}
        for f in feats:
            vals = df_train[f].dropna()
            edges_map[f] = RiskMetrics.psi_edges(vals, bins=bins) if not vals.empty else np.array([-np.inf, np.inf])

        rows = []
        for f in feats:
            row: Dict[str, object] = {"变量名称": f, "入模重要性": imp_map.get(f, np.nan)}
            for name, d in datasets.items():
                if d is None or f not in d.columns:
                    for suffix in ("覆盖率", "IV", "KS"):
                        row[f"{suffix}_{name}"] = np.nan
                    continue
                clean = self._clean(d, [f])
                row[f"覆盖率_{name}"] = RiskMetrics.calc_coverage(clean[f], self.invalid_values)
                row[f"IV_{name}"] = RiskMetrics.calc_iv(clean[self.y_col], clean[f], edges=edges_map[f])
                row[f"KS_{name}"] = RiskMetrics.calc_feature_ks(clean[self.y_col], clean[f], edges=edges_map[f])
            # 特征 PSI（train → oot / test）
            if df_oot is not None and f in df_oot.columns:
                row["PSI_oot"] = RiskMetrics.calc_psi(df_train[f], df_oot[f], edges=edges_map[f])
            else:
                row["PSI_oot"] = np.nan
            if df_test is not None and f in df_test.columns:
                row["PSI_test"] = RiskMetrics.calc_psi(df_train[f], df_test[f], edges=edges_map[f])
            rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty and "IV_train" in out.columns:
            out["IV评级"] = out["IV_train"].apply(RiskMetrics.iv_grade)
            out["PSI评级"] = out["PSI_oot"].apply(self.monitor.classify)
            out = out.sort_values("IV_train", ascending=False, na_position="last").reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # 二、单变量分析（train 分箱 → test/oot 复用边界）
    # ------------------------------------------------------------------
    def univar_analysis_df(self, df: pd.DataFrame, var_imp: pd.DataFrame,
                           top_n: int = 30, method: str = "tree",
                           max_bin: int = 5,
                           data_flag_lst: Optional[List[str]] = None) -> pd.DataFrame:
        """
        单变量分析：train 上分箱，test/oot 复用同一边界，输出对比表。

        输出列
        ------
        variable | bin | count_distr_train | badprob_train | total_iv_train |
        count_distr_test | badprob_test | total_iv_test |
        count_distr_oot  | badprob_oot  | total_iv_oot  | psi_train_test | psi_train_oot

        修复：按 **bin 值** 对齐各数据集（旧版按位置拼接，箱数不一致时错位）。
        """
        flags = data_flag_lst or [f for f in ["train", "test", "oot"] if f in set(df[self.data_flag_col].dropna())]
        parts = {f: df[df[self.data_flag_col] == f] for f in flags}
        parts = {k: v for k, v in parts.items() if not v.empty}
        if "train" not in parts:
            logger.warning("univar_analysis_df：缺少 train 数据集，无法分箱")
            return pd.DataFrame()

        feats = [f for f in var_imp["feature"].tolist()[:top_n] if f in df.columns]
        rows = []

        for f in feats:
            try:
                binner = ScorecardBinner(method=method, max_bin=max_bin)
                binner.fit(parts["train"], f, self.y_col)
            except Exception as exc:
                logger.warning("特征 %s 分箱失败：%s", f, exc)
                continue

            tables = {}
            for flag, sub in parts.items():
                try:
                    t = binner.transform(sub, f, self.y_col)
                    t = t[t["bin"] != "Total"]
                except Exception as exc:
                    logger.debug("特征 %s 在 %s 转换失败：%s", f, flag, exc)
                    continue
                if not t.empty:
                    tables[flag] = t.set_index("bin")

            if "train" not in tables:
                continue

            # 以 train 的箱为主干，按 bin 值对齐（彻底消除错位）
            base_bins = tables["train"].index.tolist()
            for bin_val in base_bins:
                row: Dict[str, object] = {"variable": f, "bin": str(bin_val)}
                for flag in flags:
                    if flag in tables and bin_val in tables[flag].index:
                        r = tables[flag].loc[bin_val]
                        row[f"count_distr_{flag}"] = round(float(r["PctRec"]), 6)
                        row[f"badprob_{flag}"] = round(float(r["bad_rate"]), 6)
                        row[f"total_iv_{flag}"] = round(float(r["total_iv"]), 6)
                    else:
                        row[f"count_distr_{flag}"] = np.nan
                        row[f"badprob_{flag}"] = np.nan
                        row[f"total_iv_{flag}"] = np.nan
                for flag in flags:
                    if flag == "train":
                        continue
                    try:
                        row[f"psi_train_{flag}"] = RiskMetrics.calc_psi(
                            parts["train"][f], parts[flag][f]
                        )
                    except Exception:
                        row[f"psi_train_{flag}"] = np.nan
                rows.append(row)

        out = pd.DataFrame(rows)
        ordered = ["variable", "bin"]
        for flag in flags:
            ordered += [c for c in [f"count_distr_{flag}", f"badprob_{flag}", f"total_iv_{flag}"] if c in out.columns]
        ordered += [c for c in out.columns if c.startswith("psi_")]
        return out[[c for c in ordered if c in out.columns]]

    @staticmethod
    def univar_multi_header(df: pd.DataFrame) -> pd.DataFrame:
        """
        把单变量分析的扁平列名转成两级表头（定版版式）。

        上层：variable / bin **沿用同名**（期望版 A2/B2 也写 variable/bin），
              其余按 `train` / `test` / `oot` / `psi` 分组
        下层：原始列名
        """
        if df is None or df.empty:
            return df
        pairs = []
        for c in df.columns:
            name = str(c)
            if name in ("variable", "bin"):
                pairs.append((name, name))
            elif name.startswith("psi_"):
                pairs.append(("psi", name))
            else:
                pairs.append((name.split("_")[-1], name))
        out = df.copy()
        out.columns = pd.MultiIndex.from_tuples(pairs)
        return out

    def univar_bin_month_table(self, df: pd.DataFrame, var_imp: pd.DataFrame,
                               top_n: int = 10, max_bin: int = 5,
                               month_col: Optional[str] = None) -> pd.DataFrame:
        """
        单变量分箱按月分析（定版新增）。

        行：Top-N 特征 × 分箱（高分箱在前）
        列：两级表头 —— 指标英文 / 分箱 之后接 5 个指标块 × 月份
            用户数 / 用户数占比 / 逾期用户数 / 逾期率 / LIFT

        分箱在**全量**上统一切分（保证各月口径可比），口径与分箱坏账趋势一致。
        """
        month_col = month_col or self.month_col
        if month_col not in df.columns:
            logger.warning("数据中无 %s 列，跳过单变量分箱按月分析", month_col)
            return pd.DataFrame()
        if self.y_col not in df.columns:
            return pd.DataFrame()

        months = [str(m) for m in sorted(df[month_col].dropna().unique())]
        feats = [f for f in var_imp["feature"].tolist()[:top_n] if f in df.columns]
        if not feats or not months:
            return pd.DataFrame()

        metrics = ["用户数", "用户数占比", "逾期用户数", "逾期率", "LIFT"]
        rows = []

        for f in feats:
            sub = df[[f, month_col, self.y_col]].copy()
            sub = sub[sub[self.y_col].isin([0, 1])]
            if self.invalid_values:
                sub[f] = sub[f].replace(self.invalid_values, np.nan)
            sub = sub[sub[f].notna()]
            if sub.empty:
                continue
            try:
                edges = RiskMetrics.psi_edges(sub[f], bins=max_bin)
                cats = pd.cut(sub[f], bins=edges, include_lowest=True)
                sub = sub.assign(_bin=cats.astype(str))
                order = [str(c) for c in reversed(list(cats.cat.categories))]
            except Exception as exc:
                logger.warning("特征 %s 分箱失败，跳过：%s", f, exc)
                continue

            grp = (sub.groupby(["_bin", month_col], observed=True)[self.y_col]
                     .agg(CntRec="count", CntBad="sum").reset_index())
            cnt = (grp.pivot(index="_bin", columns=month_col, values="CntRec")
                      .reindex(index=order, columns=months))
            bad = (grp.pivot(index="_bin", columns=month_col, values="CntBad")
                      .reindex(index=order, columns=months))

            cnt_sum, bad_sum = cnt.sum(axis=0), bad.sum(axis=0)
            safe = cnt_sum.replace(0, np.nan)
            rate_all = (bad_sum / safe).round(6)
            share = (cnt / safe).round(6)
            rate = (bad / cnt.replace(0, np.nan)).round(6)
            lift = (rate / rate_all.replace(0, np.nan)).round(4)

            for b in order:
                rows.append([f, b]
                            + [cnt.loc[b, m] for m in months]
                            + [share.loc[b, m] for m in months]
                            + [bad.loc[b, m] for m in months]
                            + [rate.loc[b, m] for m in months]
                            + [lift.loc[b, m] for m in months])

        if not rows:
            return pd.DataFrame()

        cols = [("", "指标英文"), ("", "分箱")]
        for metric in metrics:
            for m in months:
                cols.append((metric, m))
        return pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(cols))

    # ------------------------------------------------------------------
    # 三、特征逐月趋势
    # ------------------------------------------------------------------
    def feature_trend(self, data: pd.DataFrame, importance: pd.DataFrame,
                      top: int = 20, by_month: bool = True) -> pd.DataFrame:
        """
        特征分布逐月趋势（均值/标准差/分位数/缺失率）。

        修复：旧版 `feature_fb` 用 `describe` 剔除非数值列后与 `nunique()` 行数不齐导致崩。
        """
        feats = [f for f in importance["feature"].tolist()[:top] if f in data.columns]
        num_feats = [f for f in feats if pd.api.types.is_numeric_dtype(data[f])]
        skipped = [f for f in feats if f not in num_feats]
        if skipped:
            logger.info("特征趋势跳过非数值特征：%s", skipped)

        if not num_feats:
            return pd.DataFrame()

        if by_month and self.month_col in data.columns:
            grp = data.groupby(self.month_col)[num_feats]
            res = grp.agg(["mean", "std"]).T
            res.columns = [f"{m}_{a}" for m, a in res.columns]
        else:
            res = data[num_feats].agg(["mean", "std"]).T
            res.columns = [f"all_{c}" for c in res.columns]

        res = res.reset_index().rename(columns={"level_0": "feature", "index": "feature"})
        res["缺失率"] = [RiskMetrics.calc_missing_rate(data[f]) for f in res["feature"]]
        res["nUnique"] = [int(data[f].nunique(dropna=True)) for f in res["feature"]]
        return res

    # ------------------------------------------------------------------
    # 四、共线性 VIF
    # ------------------------------------------------------------------
    @staticmethod
    def vif(df: pd.DataFrame, feats: Sequence[str], max_missing_rate: float = 0.5) -> pd.DataFrame:
        """
        方差膨胀因子（VIF > 10 通常认为存在严重共线性）。

        缺失率过高的特征自动跳过，避免填充引入伪相关。
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logger.warning("未安装 statsmodels，跳过 VIF 计算")
            return pd.DataFrame()

        use = []
        for f in feats:
            if f not in df.columns:
                continue
            s = df[f]
            if not pd.api.types.is_numeric_dtype(s):
                continue
            if s.isna().mean() > max_missing_rate:
                continue
            use.append(f)
        if len(use) < 2:
            return pd.DataFrame()

        sub = df[use].apply(pd.to_numeric, errors="coerce")
        sub = sub.fillna(sub.median())
        rows = []
        for i, f in enumerate(use):
            try:
                v = variance_inflation_factor(sub.values, i)
            except Exception:
                v = np.nan
            rows.append({"feature": f, "VIF": round(float(v), 4) if v == v else np.nan})
        out = pd.DataFrame(rows)
        out["共线性"] = out["VIF"].apply(lambda x: "严重(>10)" if (x == x and x > 10) else ("中等(5~10)" if (x == x and x > 5) else "正常"))
        return out.sort_values("VIF", ascending=False, na_position="last").reset_index(drop=True)


__all__ = ["FeatureAnalyzer"]
