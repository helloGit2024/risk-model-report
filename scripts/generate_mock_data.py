# -*- coding: utf-8 -*-
"""
Mock 数据生成器
================================================================================
用于覆盖**真实数据无法触达的异常分支**，与真实数据验证形成互补：

异常场景
------------------------------------------------------------------
1. 单类别月       某个月只有好样本（旧版 crosstab 缺列 → KeyError）
2. 极端缺失特征   缺失率 > 30%（验证覆盖率/缺失率口径）
3. 类别型特征     字符串特征（旧版 Cal_Psi 遇字符串 → 崩溃）
4. 常量特征       取值唯一（分箱/IV 退化场景）
5. 漂移特征       OOT 分布显著偏移（PSI > 0.25，验证告警生效）
6. 空分组         某渠道在某月无样本（验证分组健壮性）
7. 极小分组       某渠道样本量 < 30（PSI 拒绝计算，返回 NaN 而非报错）

正常场景
------------------------------------------------------------------
* 强/中/弱三档特征（IV 分档可验证）
* 模型分 KS 控制在 0.35 ~ 0.45（典型可用模型）
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from riskpkg.metrics import RiskMetrics


def generate_mock_data(n: int = 50000, seed: int = 42, out_dir: str = "mock",
                       target_ks: tuple = (0.35, 0.45)) -> Dict[str, str]:
    """
    生成 mock 数据集（模型结果表 + 宽表 + 特征重要性）。

    输出文件（统一 `mock_` 前缀，与 data/ 真实数据一一对应）
    ------------------------------------------------------------------
    mock_model_result.csv        模型结果表（分数+标签+维度）  ← data/lgbdoc.csv
    mock_wide.csv                宽表（特征+标签+划分）        ← data/D05_newdf.csv
    mock_feature_importance.csv  特征重要性                    ← data/D05_lgb_importance.csv

    Returns
    -------
    dict: {'model_result': 模型结果表, 'wide': 宽表, 'importance': 重要性}
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 基础维度 ----------------
    months = [f"2024-{m:02d}" for m in range(1, 9)]
    # 月份权重：最后一个月样本很少（用于触发小样本/单类别场景）
    month_w = np.array([0.10, 0.12, 0.13, 0.14, 0.13, 0.12, 0.11, 0.02])
    month_w = month_w / month_w.sum()
    month = rng.choice(months, size=n, p=month_w)

    channels = np.array([101, 102, 103])
    ch_w = np.array([0.5, 0.3, 0.2])
    channel = rng.choice(channels, size=n, p=ch_w)

    # 数据集：oot 为最后两个月，train/test 随机分层
    is_oot = np.isin(month, months[-2:])
    data_flag = np.where(is_oot, "oot",
                         rng.choice(["train", "test"], size=n, p=[0.8, 0.2]))

    # ---------------- 特征 ----------------
    feats: Dict[str, np.ndarray] = {}

    # 3 个强特征（IV > 0.3）
    for i in range(3):
        feats[f"strong_{i}"] = rng.normal(0, 1, n)
    # 5 个中等特征（0.1 < IV < 0.3）
    for i in range(5):
        feats[f"mid_{i}"] = rng.normal(0, 1, n)
    # 14 个弱/噪声特征
    for i in range(14):
        feats[f"noise_{i}"] = rng.normal(0, 1, n)
    # 常量特征（退化场景）
    feats["const_feat"] = np.ones(n)
    # 类别型特征（字符串，旧版 PSI 会崩）
    feats["cat_feat"] = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])

    # ---------------- 标签与分数 ----------------
    logit = (-2.2
             + 1.6 * feats["strong_0"] + 1.3 * feats["strong_1"] + 1.1 * feats["strong_2"]
             + 0.7 * feats["mid_0"] + 0.6 * feats["mid_1"]
             + 0.5 * feats["mid_2"] + 0.4 * feats["mid_3"] + 0.3 * feats["mid_4"])
    p_true = 1 / (1 + np.exp(-logit))

    # 调整噪声使 KS 落入目标区间
    sigma = 0.6
    y = rng.binomial(1, p_true)
    for _ in range(24):
        score = np.clip(p_true + rng.normal(0, sigma, n), 1e-6, 1 - 1e-6)
        ks = RiskMetrics.calc_ks(y[~is_oot], score[~is_oot])
        if ks is None or np.isnan(ks):
            sigma *= 0.8
            continue
        if ks < target_ks[0]:
            sigma *= 0.75
        elif ks > target_ks[1]:
            sigma *= 1.25
        else:
            break

    # ---------------- 异常分支构造 ----------------
    # 1. 单类别月：最后一个 oot 月全部为好样本（无坏样本）
    single_class_month = months[-1]
    m_mask = month == single_class_month
    y[m_mask] = 0

    # 2. 漂移特征：OOT 上分布显著偏移（PSI > 0.25）
    drift = rng.normal(0, 1, n)
    drift[is_oot] = drift[is_oot] + 2.5
    feats["drift_feat"] = drift

    # 3. 高缺失特征（缺失率 > 30%）
    hi_miss = rng.normal(0, 1, n)
    hi_miss[rng.random(n) < 0.35] = np.nan
    feats["high_missing_feat"] = hi_miss

    # 4. 空分组：渠道 103 在某个 train 月无样本（把该组合样本改到其他月）
    empty_m = months[0]
    idx = np.where((channel == 103) & (month == empty_m))[0]
    if len(idx) > 0:
        month[idx] = months[1]

    # ---------------- 组装 ----------------
    df = pd.DataFrame(feats)
    df["id"] = np.arange(n)
    df["month"] = month
    df["channle"] = channel
    df["y"] = y
    df["D05_mock_score"] = score
    df["data_flag"] = data_flag
    df["dateback"] = pd.to_datetime(df["month"].astype(str) + "-15").dt.strftime("%Y-%m-%d")

    # 宽表（含特征 + 标签 + 划分）
    wide_cols = [c for c in df.columns if c not in ("D05_mock_score",)]
    data_path = os.path.join(out_dir, "mock_wide.csv")
    df[wide_cols].to_csv(data_path, index=False)

    # 模型结果表（分数 + 标签 + 维度）
    rnt_cols = ["id", "dateback", "month", "channle", "y", "data_flag", "D05_mock_score"]
    rnt_path = os.path.join(out_dir, "mock_model_result.csv")
    df[rnt_cols].to_csv(rnt_path, index=False)

    # 特征重要性
    feat_names = [c for c in df.columns
                  if c.startswith(("strong_", "mid_", "noise_", "drift_", "high_missing_",
                                   "cat_feat", "const_feat"))]
    imp = pd.DataFrame({
        "feature": feat_names,
        "importance_gain": rng.uniform(10, 30000, len(feat_names)),
        "importance_split": rng.integers(10, 600, len(feat_names)),
    }).sort_values("importance_gain", ascending=False)
    imp_path = os.path.join(out_dir, "mock_feature_importance.csv")
    imp.to_csv(imp_path, index=False)

    # 报告配置（与生成数据的字段口径严格对应，可直接喂给 runReport.get_report）
    import json
    cfg = {
        "target": "y",                 # 标签列，1=坏 0=好
        "date": "dateback",            # 日期列（每月 15 日）
        "data_flag": "data_flag",      # 数据集划分列：train / test / oot
        "channle_name": "channle",     # 渠道列：101 / 102 / 103
        "is_channle": True,            # 按渠道拆分渠道相关 Sheet
        "eval_univariate": True,       # 生成特征全景 / 单变量分析等 Sheet
        "bins": 10,                    # 模型分分箱数
        "bins_lst": [10, 20],          # 多套分箱（模型分10箱 / 模型分20箱）
        "importance_top": 26,          # 特征分析取前 N 个（mock 全部 26 个特征）
        "model_name": "D05_mock",      # 报告标题中的模型名
        "keys": ["id"],                # 主键列
    }
    cfg_path = os.path.join(out_dir, "mock_main_config.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("Mock 数据生成完成：")
    print(f"  {rnt_path}  {df[rnt_cols].shape}")
    print(f"  {data_path}  {df[wide_cols].shape}")
    print(f"  {imp_path}  {imp.shape}")
    print(f"  {cfg_path}")
    print(f"  坏率={y.mean():.4f}  KS(非oot)={RiskMetrics.calc_ks(y[~is_oot], score[~is_oot]):.4f}"
          f"  AUC={RiskMetrics.calc_auc(y[~is_oot], score[~is_oot]):.4f}")
    return {"model_result": rnt_path, "wide": data_path,
            "importance": imp_path, "config": cfg_path}


if __name__ == "__main__":
    generate_mock_data()
