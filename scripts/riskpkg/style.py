# -*- coding: utf-8 -*-
"""
样式层（StyleConfig · 可换模板）
================================================================================
把「定版」时硬编码在 `excel.py` 里的配色 / 字体 / 空行 / 阈值 / Sheet 编排
抽出来，做成**可替换的模板配置**，满足企业级"换公司模板"的需求。

三种用法
------------------------------------------------------------------
1. 内置模板名：`load_style("default")` / `load_style("compact")`
2. 外部 JSON：`load_style("path/to/corp.json")`（见 `assets/styles/`）
3. 代码注入：`StyleConfig.from_dict({...})`

模板 JSON 结构（缺省字段自动继承 default）
------------------------------------------------------------------
{
  "name": "corp",
  "font": "微软雅黑", "font_size": 10,
  "colors": {"green": "#63BE7B", "header_bg": "#4472C4", ...},
  "thresholds": {"psi_ok": 0.1, "psi_warn": 0.25, "ks_ok": 0.3, "ks_warn": 0.2},
  "gap_rows": 2,        // 同一 sheet 内相邻数据块之间的空行
  "group_gap": 1,       // 同一表内不同分组之间的空行
  "col_width": 14,
  "freeze": true,
  "sheet_order": ["总览", "样本统计", ...],   // 缺省=内置顺序
  "sheet_include": ["总览", ...]              // 缺省=全部；用于裁剪报告
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# 内置默认值（与"定版"版式完全一致）
# -----------------------------------------------------------------------------
DEFAULT_COLORS: Dict[str, str] = {
    "green": "#63BE7B",
    "yellow": "#FFEB84",
    "red": "#F8696B",
    "blue": "#638EC6",
    "light_blue": "#B4C6E7",
    "orange": "#F4B183",
    "header_bg": "#4472C4",
    "header_font": "#FFFFFF",
    "title_bg": "#D9E1F2",
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "psi_ok": 0.1,     # PSI 观察阈值
    "psi_warn": 0.25,  # PSI 漂移告警阈值
    "ks_ok": 0.3,      # KS 优秀线
    "ks_warn": 0.2,    # KS 可用线
}

# 内置模板：name -> 配置片段（缺字段继承 default）
BUILTIN_STYLES: Dict[str, Dict[str, Any]] = {
    "default": {},
    "compact": {
        "name": "compact",
        "font_size": 9,
        "gap_rows": 1,
        "group_gap": 0,
        "col_width": 12,
    },
}


@dataclass
class StyleConfig:
    """报告版式配置。"""

    name: str = "default"
    font: str = "微软雅黑"
    font_size: int = 10
    colors: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COLORS))
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    gap_rows: int = 2
    group_gap: int = 1
    col_width: int = 14
    freeze: bool = True
    sheet_order: Optional[List[str]] = None
    sheet_include: Optional[List[str]] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleConfig":
        data = dict(data or {})
        base = cls()
        colors = dict(DEFAULT_COLORS)
        colors.update(data.pop("colors", {}) or {})
        th = dict(DEFAULT_THRESHOLDS)
        th.update(data.pop("thresholds", {}) or {})
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(colors=colors, thresholds=th, **known)

    @classmethod
    def from_json(cls, path: str) -> "StyleConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "font": self.font,
            "font_size": self.font_size,
            "colors": dict(self.colors),
            "thresholds": dict(self.thresholds),
            "gap_rows": self.gap_rows,
            "group_gap": self.group_gap,
            "col_width": self.col_width,
            "freeze": self.freeze,
            "sheet_order": self.sheet_order,
            "sheet_include": self.sheet_include,
        }

    # 阈值便捷读取（供 evaluation / stability 复用同一套口径）
    def thr(self, key: str, default: float = 0.0) -> float:
        return float(self.thresholds.get(key, default))

    def color(self, key: str, default: str = "#000000") -> str:
        return self.colors.get(key, default)


def load_style(spec: Optional[Any] = None) -> StyleConfig:
    """
    加载模板。

    Parameters
    ----------
    spec : None | "default" | "compact" | StyleConfig | JSON 路径
    """
    if spec is None:
        return StyleConfig()
    if isinstance(spec, StyleConfig):
        return spec
    if isinstance(spec, dict):
        return StyleConfig.from_dict(spec)
    name = str(spec)
    if name in BUILTIN_STYLES:
        return StyleConfig.from_dict(BUILTIN_STYLES[name])
    if os.path.exists(name):
        return StyleConfig.from_json(name)
    raise ValueError(
        f"未知模板 {spec!r}：可选内置 {sorted(BUILTIN_STYLES)}，或传入 JSON 文件路径"
    )


def dump_style(style: StyleConfig, path: str) -> str:
    """把模板落盘，便于企业内复用/版本化管理。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(style.to_dict(), f, ensure_ascii=False, indent=2)
    return path


__all__ = ["StyleConfig", "load_style", "dump_style", "BUILTIN_STYLES",
           "DEFAULT_COLORS", "DEFAULT_THRESHOLDS"]
