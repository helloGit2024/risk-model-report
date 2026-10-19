# -*- coding: utf-8 -*-
"""
环境自检（跑报告之前先跑它）
================================================================================
为什么需要：报告依赖 ~10 个三方库，其中 5 个是**可选**的（缺了会静默降级）。
出报告前先自检，可避免"跑 85 秒才发现缺 xlsxwriter"的情况。

用法
------------------------------------------------------------------
python check_env.py              # 人类可读输出
python check_env.py --json       # 机器可读（供调度系统判断）
退出码：0=核心依赖齐全且无版本越界；1=依赖缺失或版本不兼容（必须处理后才能出报告）
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Dict, List, Tuple

# 核心依赖：import 名 -> (pip 包名, 最低版本)
CORE_DEPS: Dict[str, Tuple[str, str]] = {
    "pandas": ("pandas", "1.5.0"),
    "numpy": ("numpy", "1.21.0"),
    "sklearn": ("scikit-learn", "1.0.0"),
    "xlsxwriter": ("xlsxwriter", "3.0.0"),   # 数据条/色阶依赖，openpyxl 不支持
    "matplotlib": ("matplotlib", "3.5.0"),
    "seaborn": ("seaborn", "0.12.0"),
    "openpyxl": ("openpyxl", "3.0.0"),       # 产出验收读取 xlsx 用
}

# 版本不兼容上界（不含）：import 名 -> (上界版本, 说明)
# 为什么需要：旧版只校验下限，pandas 3.0.5 一路绿灯放行，
# 直到 evaluation.py 往 float 列写 "否" 才抛 LossySetitemError（跑了 4.5 秒才炸）。
# requirements.txt 已 pin pandas<2.2，这里必须与它口径一致。
UPPER_BOUNDS: Dict[str, Tuple[str, str]] = {
    "pandas": ("3.0.0", "pandas>=3.0 未适配：往 float 列写字符串会抛 LossySetitemError"
                        "（全库按 1.5~2.x 语义编写）→ pip install 'pandas>=1.5,<2.2'"),
}

# 可选依赖：import 名 -> (pip 包名, 缺失时的降级行为)
OPTIONAL_DEPS: Dict[str, Tuple[str, str]] = {
    "scorecardpy": ("scorecardpy", "chimerge/tree 分箱后端 → 降级为自研等频/决策树分箱"),
    "statsmodels": ("statsmodels", "VIF 共线性 → 跳过 VIF"),
    "shap": ("shap", "SHAP 特征贡献图 → 跳过，不影响报告"),
    "lightgbm": ("lightgbm", "读取模型 feature_name_/importances_ → 跳过"),
    "scipy": ("scipy", "Hosmer-Lemeshow 校准检验 p 值 → 跳过"),
    "pydantic": ("pydantic", "ReportConfig 校验 → 退化为普通对象（行为一致）"),
}


def _version_tuple(v: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for seg in str(v).split(".")[:3]:
        num = "".join(ch for ch in seg if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts) if parts else (0,)


def _installed_version(mod) -> str:
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "unknown"


def check(verbose: bool = True) -> Dict[str, Any]:
    """执行环境自检，返回结构化结果。"""
    core: List[Dict[str, Any]] = []
    optional: List[Dict[str, Any]] = []
    missing_core: List[str] = []
    incompatible: List[str] = []

    for imp, (pip_name, min_v) in CORE_DEPS.items():
        try:
            mod = importlib.import_module(imp)
            ver = _installed_version(mod)
            ok = ver == "unknown" or _version_tuple(ver) >= _version_tuple(min_v)
            note = ""
            # 版本上界：只校验下限会让大版本跳跃（如 pandas 3.x）一路绿灯
            if imp in UPPER_BOUNDS and ver != "unknown":
                hi, msg = UPPER_BOUNDS[imp]
                if _version_tuple(ver) >= _version_tuple(hi):
                    ok = False
                    note = msg
                    incompatible.append(f"{pip_name}<{hi}（当前 {ver}）")
            core.append({"import": imp, "pip": pip_name, "version": ver,
                         "required": min_v, "ok": bool(ok), "upper_note": note})
            if not ok and not note:
                missing_core.append(f"{pip_name}>={min_v}（当前 {ver}）")
        except ImportError:
            core.append({"import": imp, "pip": pip_name, "version": None,
                         "required": min_v, "ok": False, "upper_note": ""})
            missing_core.append(f"{pip_name}>={min_v}（未安装）")

    for imp, (pip_name, degrade) in OPTIONAL_DEPS.items():
        try:
            mod = importlib.import_module(imp)
            optional.append({"import": imp, "pip": pip_name,
                             "version": _installed_version(mod), "installed": True,
                             "degrade": degrade})
        except ImportError:
            optional.append({"import": imp, "pip": pip_name, "version": None,
                             "installed": False, "degrade": degrade})

    # 中文字体（图表乱码的常见根因）
    font_ok = False
    try:
        import matplotlib  # noqa: F401
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        font_ok = any(k in "".join(names) for k in ("SimHei", "Microsoft YaHei", "微软雅黑"))
    except Exception:
        font_ok = False

    result = {
        "ok": len(missing_core) == 0 and len(incompatible) == 0,
        "python": sys.version.split()[0],
        "core": core,
        "optional": optional,
        "missing_core": missing_core,
        "incompatible": incompatible,
        "chinese_font": bool(font_ok),
    }

    if verbose:
        print("=" * 62)
        print(f"环境自检 | Python {result['python']}")
        print("=" * 62)
        for c in core:
            flag = "OK  " if c["ok"] else "MISS"
            print(f"  [{flag}] {c['pip']:<14} {str(c['version']):<12} (需 >={c['required']})")
            if c.get("upper_note"):
                print(f"         ↳ 不兼容：{c['upper_note']}")
        print("-" * 62)
        miss_opt = [o for o in optional if not o["installed"]]
        for o in optional:
            if o["installed"]:
                print(f"  [OPT ] {o['pip']:<14} {o['version']}")
        for o in miss_opt:
            print(f"  [ -- ] {o['pip']:<14} 未安装 → {o['degrade']}")
        print("-" * 62)
        print(f"  中文字体：{'可用' if font_ok else '未发现（图表中文可能显示为方框）'}")
        print("=" * 62)
        if missing_core or incompatible:
            print("结论：环境不可用，请先处理：")
            if missing_core:
                print("  pip install " + " ".join(m.split("（")[0] for m in missing_core))
            for b in incompatible:
                print(f"  版本不兼容：{b} → 需降级")
        else:
            print("结论：核心依赖齐全，可以出报告。")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="风控模型报告 · 环境自检")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args(argv)
    res = check(verbose=not args.json)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
