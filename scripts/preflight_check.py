# -*- coding: utf-8 -*-
"""
输入契约前置校验（出报告之前必跑）
================================================================================
为什么需要：完整报告要跑 ~85 秒，若输入列名不对 / 标签不是 0-1 / 缺 oot，
原来的表现是"跑完才炸"或"静默少几个 sheet"。本脚本在 3 秒内把这些问题拦掉。

校验项
------------------------------------------------------------------
ERROR（必须修，否则报告必失败）
  E1 文件不存在（config / rnt）
  E2 结果表缺必需列：模型分数列 / 标签列 / 日期列(或 month) / 数据集标识列
  E3 标签列不是 0-1 二值（且未声明灰样本取值）

WARN（能跑，但会有降级或结论不可信）
  W1 缺 test / oot，无法做过拟合与跨时间稳定性诊断（**全量判定**，不受抽样影响）
  W2 坏率为 0 或 1（样本全同）
  W3 坏率过低（<0.5%）或过高（>50%），KS/PSI 结论需谨慎
  W4 分数列缺失率 > 5%
  W5 宽表/重要性缺失或特征覆盖率低 → 4 个特征类 sheet 不出
  W6 日期列无法解析为时间 → 逐月分析不可用
  W7 分数方向疑似反了（分高 = 优质客户）→ lift / 捕获率方向全反

用法
------------------------------------------------------------------
python preflight_check.py --config cfg.json --model SCORE --rnt rnt.csv \
        --wide wide.csv --importance imp.csv
python preflight_check.py --config cfg.json --model SCORE --rnt rnt.csv --json
退出码：0=可继续（含 WARN）；1=必须修（含 ERROR）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# 抽样行数（大表只读前 N 行做契约校验）
SAMPLE_ROWS = 200_000


def _load_any(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
        return df.head(nrows) if nrows else df
    return pd.read_csv(path, nrows=nrows)


def _columns_only(path: str) -> List[str]:
    """只取列名（宽表可能极大，避免全量加载）。"""
    try:
        if path.endswith(".pkl"):
            return list(pd.read_pickle(path).columns)
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _load_column(path: str, col: str) -> Optional[List[Any]]:
    """只读取单列的**全量**取值（不抽样）。

    为什么需要：本脚本为提速会把结果表抽样到前 SAMPLE_ROWS 行，
    而 oot 常常排在文件尾部（按时间排序），抽样会把 oot 整段漏掉，
    导致 W1 误报「缺少 oot」。单列全量读取成本很低（50 万行 × 1 列），
    用它能拿到真实的数据集分布。
    """
    try:
        if path.endswith(".pkl"):
            obj = pd.read_pickle(path)
            if col not in getattr(obj, "columns", []):
                return None
            return obj[col].dropna().tolist()
        return pd.read_csv(path, usecols=[col])[col].dropna().tolist()
    except Exception:
        return None


def preflight(config_path: Optional[str], model: str, rnt_path: str,
              wide_path: Optional[str] = None,
              importance_path: Optional[str] = None,
              verbose: bool = True) -> Dict[str, Any]:
    """执行前置校验，返回结构化结果。"""
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    # ---- E1 文件存在性 ----
    if config_path and not os.path.exists(config_path):
        errors.append(f"[E1] 配置文件不存在：{config_path}")
        cfg: Dict[str, Any] = {}
    elif config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            errors.append(f"[E1] 配置文件解析失败：{exc}")
            cfg = {}
    else:
        cfg = {}

    target = cfg.get("target", "y")
    date_col = cfg.get("date", "dateback")
    flag_col = cfg.get("data_flag", "data_flag")
    gray = cfg.get("other_label_value") or []
    info["config"] = {"target": target, "date": date_col, "data_flag": flag_col}

    if not rnt_path or not os.path.exists(rnt_path):
        errors.append(f"[E1] 结果表不存在：{rnt_path}")
        return {"ok": False, "errors": errors, "warnings": warnings, "info": info}

    # ---- 加载结果表（抽样）----
    try:
        df = _load_any(rnt_path, SAMPLE_ROWS)
    except Exception as exc:
        errors.append(f"[E1] 结果表读取失败：{exc}")
        return {"ok": False, "errors": errors, "warnings": warnings, "info": info}

    info["rnt_shape"] = list(df.shape)

    # ---- E2 必需列 ----
    required = {model: "模型分数列", target: "标签列", flag_col: "数据集标识列"}
    for col, desc in required.items():
        if col not in df.columns:
            errors.append(f"[E2] 结果表缺少{desc} `{col}`（现有列：{list(df.columns)[:12]}…）")
    has_date = date_col in df.columns or "month" in df.columns
    if not has_date:
        errors.append(f"[E2] 结果表缺少日期列 `{date_col}`，也没有 month 列 → 逐月分析不可用")

    # ---- E3 标签二值性 ----
    if target in df.columns:
        vals = set(pd.Series(df[target].dropna().unique()).tolist())
        allowed = {0, 1} | {int(v) for v in gray if str(v).lstrip("-").isdigit()}
        if not vals.issubset(allowed):
            errors.append(f"[E3] 标签列 `{target}` 取值异常：{sorted(vals)}（应为 0/1，"
                          f"灰样本需在 config.other_label_value 声明）")
        # 坏率
        good = df[target].isin([0, 1])
        bad_rate = float(df.loc[good, target].mean()) if good.any() else float("nan")
        info["bad_rate"] = None if pd.isna(bad_rate) else round(bad_rate, 6)
        if good.any():
            if bad_rate in (0.0, 1.0):
                warnings.append(f"[W2] 坏率={bad_rate}：样本标签全同，KS/AUC 无意义")
            elif bad_rate < 0.005 or bad_rate > 0.5:
                warnings.append(f"[W3] 坏率={bad_rate:.4f} 偏离常规区间(0.5%~50%)，结论需谨慎")

    # ---- W1 数据集划分（全量判定：抽样会把排在文件尾部的 oot 漏掉）----
    if flag_col in df.columns:
        flags = set(pd.Series(df[flag_col].dropna().unique()).astype(str).tolist())
        full = _load_column(rnt_path, flag_col)
        if full:
            flags = set(pd.Series(full).astype(str).unique().tolist())
            info["flag_counts"] = {str(k): int(v)
                                   for k, v in pd.Series(full).value_counts().to_dict().items()}
            info["flag_counts_source"] = "全量"
        else:  # 单列读取失败（如非标准格式）→ 退回抽样结果，并标明来源
            cnt = df[flag_col].value_counts().to_dict()
            info["flag_counts"] = {str(k): int(v) for k, v in cnt.items()}
            info["flag_counts_source"] = f"抽样前 {len(df)} 行"
        info["data_flags"] = sorted(flags)
        for need in ("test", "oot"):
            if need not in flags:
                warnings.append(f"[W1] 数据集缺少 {need} → 无法做"
                                f"{'过拟合诊断' if need == 'test' else '跨时间稳定性诊断'}")

    # ---- W4 分数列缺失率 ----
    if model in df.columns:
        miss = float(df[model].isna().mean())
        info["score_missing_rate"] = round(miss, 6)
        if miss > 0.05:
            warnings.append(f"[W4] 分数列 `{model}` 缺失率 {miss:.2%} > 5%")

    # ---- W7 分数方向（分高 = 高风险？）----
    try:
        if model in df.columns and target in df.columns:
            d = df[[model, target]].apply(pd.to_numeric, errors="coerce").dropna()
            d = d[d[target].isin([0, 1])]
            if len(d) >= 1000:
                corr = float(d[model].corr(d[target]))
                mono = None
                try:
                    bins = pd.qcut(d[model], 10, duplicates="drop")
                    br = d.groupby(bins, observed=True)[target].mean()
                    up = sum(1 for i in range(len(br) - 1) if br.iloc[i + 1] >= br.iloc[i])
                    mono = round(up / max(len(br) - 1, 1), 4)
                except Exception:
                    pass
                info["score_direction"] = {"corr_with_y": round(corr, 6),
                                           "bad_rate_monotonic": mono}
                if corr < 0 or (mono is not None and mono < 0.4):
                    warnings.append(
                        f"[W7] 分数与坏率{'负相关' if corr < 0 else '非单调递增'}"
                        f"（corr={corr:.4f}，十分位坏率单调比例={mono}）"
                        f" → 疑似「分高 = 优质客户」。先取反：df['risk'] = -df['{model}']，"
                        f"再传 --model risk（见 data-contract.md §5）")
                elif abs(corr) < 0.01:
                    info["score_direction"]["note"] = \
                        "区分度极弱：KS 会很低，校准度 / Brier 不宜解读"
    except Exception:
        pass

    # ---- W6 日期可解析 ----
    if date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        info["date_parse_fail_rate"] = round(float(parsed.isna().mean()), 6)
        if parsed.isna().mean() > 0.05:
            warnings.append(f"[W6] 日期列 `{date_col}` 解析失败率 "
                            f"{parsed.isna().mean():.2%} > 5% → 逐月分析可能为空")

    # ---- W5 宽表 / 重要性 ----
    imp_cols: List[str] = []
    if importance_path:
        if not os.path.exists(importance_path):
            warnings.append(f"[W5] 特征重要性文件不存在：{importance_path} → 4 个特征类 sheet 不出")
        else:
            try:
                imp = _load_any(importance_path)
                cand = [c for c in imp.columns if str(c).lower() in
                        ("feature", "feature_name", "name", "var", "变量")]
                if not cand:
                    warnings.append(f"[W5] 特征重要性缺少特征名列（现有列：{list(imp.columns)}）")
                else:
                    imp_cols = [str(v) for v in imp[cand[0]].dropna().tolist()]
                    info["importance_features"] = len(imp_cols)
            except Exception as exc:
                warnings.append(f"[W5] 特征重要性读取失败：{exc}")
    else:
        warnings.append("[W5] 未提供特征重要性 → 4 个特征类 sheet 不出")

    if wide_path:
        if not os.path.exists(wide_path):
            warnings.append(f"[W5] 宽表不存在：{wide_path} → 4 个特征类 sheet 不出")
        else:
            try:
                wcols = set(_columns_only(wide_path))
                info["wide_columns"] = len(wcols)
                if imp_cols:
                    hit = sum(1 for c in imp_cols if c in wcols)
                    info["feature_coverage"] = round(hit / max(len(imp_cols), 1), 4)
                    if hit == 0:
                        warnings.append("[W5] 重要特征在宽表中 0 命中 → 特征类 sheet 将为空")
                    elif hit / len(imp_cols) < 0.5:
                        warnings.append(
                            f"[W5] 重要特征在宽表命中率仅 {hit}/{len(imp_cols)} → 特征类 sheet 会大量缺失")
                if flag_col not in wcols:
                    warnings.append(f"[W5] 宽表缺少数据集标识列 `{flag_col}` → 特征 PSI 会失败")
            except Exception as exc:
                warnings.append(f"[W5] 宽表读取失败：{exc}")
    else:
        warnings.append("[W5] 未提供宽表 → 4 个特征类 sheet 不出")

    result = {"ok": len(errors) == 0, "errors": errors,
              "warnings": warnings, "info": info}

    if verbose:
        print("=" * 62)
        print(f"输入契约前置校验 | model={model} | rnt={rnt_path}")
        print("=" * 62)
        print(f"  结果表：{info.get('rnt_shape')} | 坏率={info.get('bad_rate')} "
              f"| 数据集={info.get('data_flags')}")
        if info.get("flag_counts"):
            print(f"  数据集分布：{info['flag_counts']}（{info.get('flag_counts_source', '-')}）")
        if info.get("score_direction"):
            sd = info["score_direction"]
            print(f"  分数方向：corr(score,y)={sd.get('corr_with_y')} "
                  f"| 十分位坏率单调比例={sd.get('bad_rate_monotonic')}"
                  f"{' | ' + sd['note'] if sd.get('note') else ''}")
        for w in warnings:
            print(f"  {w}")
        for e in errors:
            print(f"  {e}")
        print("=" * 62)
        print("结论：" + ("可以出报告" if result["ok"] else "存在致命问题，请先修复"))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="风控模型报告 · 输入契约前置校验")
    p.add_argument("--config", help="配置 JSON 路径")
    p.add_argument("--model", required=True, help="模型分数列名")
    p.add_argument("--rnt", required=True, help="结果表路径（csv/pkl）")
    p.add_argument("--wide", default=None, help="宽表路径")
    p.add_argument("--importance", default=None, help="特征重要性路径")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args(argv)
    res = preflight(args.config, args.model, args.rnt, args.wide,
                    args.importance, verbose=not args.json)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
