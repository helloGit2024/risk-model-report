# -*- coding: utf-8 -*-
"""
报告层（ExcelStyler / ReportBuilder）
================================================
职责
------------------------------------------------------------------
* `ExcelStyler` —— sheet 内数据美化函数库（数据条 / 色阶 / 图标集 / 阈值高亮 / 边框 /
  列宽 / 冻结 / 数字格式），**所有 sheet 复用同一套 API**
* `ReportBuilder` —— Excel 报告构建器，根治旧版 `to_excel` 未指定 `sheet_name`
  导致所有表堆进同一个 Sheet1、标题写在空 sheet 上的严重缺陷

美化预设（preset）
------------------------------------------------------------------
metric  KS/AUC/Gini/Lift 等区分度指标：三色阶 + 阈值高亮
psi     PSI 稳定性：绿(0) → 黄(0.1) → 红(0.25)
bin     分箱表：样本量数据条 + 坏率/Lift 色阶
univar  单变量分析：IV/PSI 数据条 + 阈值高亮
sample  样本统计：坏账率色阶
none    仅基础边框与格式
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .style import StyleConfig, load_style

logger = logging.getLogger(__name__)

# 配色（默认值来自 StyleConfig；ReportBuilder 构造时会用当前模板 apply_style 覆盖，
# 因此这些常量是"上一次生效的模板值"，换模板无需改本文件）
C_GREEN = "#63BE7B"
C_YELLOW = "#FFEB84"
C_RED = "#F8696B"
C_BLUE = "#638EC6"
C_LIGHT_BLUE = "#B4C6E7"
C_ORANGE = "#F4B183"

# PSI 行业阈值
PSI_OK, PSI_WARN = 0.1, 0.25
# KS 阈值
KS_OK, KS_WARN = 0.3, 0.2


def apply_style(style: Optional[StyleConfig]) -> StyleConfig:
    """
    把模板配置应用到本模块的全局常量（配色 / 阈值）。

    设计取舍：样式常量原本硬编码在 30+ 处调用点，逐处改签名成本高；
    改为"模板生效 = 刷新模块全局"，配合"每次构建报告都新建 ReportBuilder"
    （格式缓存随 ExcelStyler 实例重建），可保证不同模板互不串味。
    """
    global C_GREEN, C_YELLOW, C_RED, C_BLUE, C_LIGHT_BLUE, C_ORANGE
    global PSI_OK, PSI_WARN, KS_OK, KS_WARN

    style = load_style(style)
    C_GREEN = style.color("green", C_GREEN)
    C_YELLOW = style.color("yellow", C_YELLOW)
    C_RED = style.color("red", C_RED)
    C_BLUE = style.color("blue", C_BLUE)
    C_LIGHT_BLUE = style.color("light_blue", C_LIGHT_BLUE)
    C_ORANGE = style.color("orange", C_ORANGE)
    PSI_OK = style.thr("psi_ok", PSI_OK)
    PSI_WARN = style.thr("psi_warn", PSI_WARN)
    KS_OK = style.thr("ks_ok", KS_OK)
    KS_WARN = style.thr("ks_warn", KS_WARN)
    return style


class ExcelStyler:
    """
    Sheet 内数据美化器（依赖 xlsxwriter）。

    Usage
    -----
    >>> styler = ExcelStyler(workbook)
    >>> styler.beautify(ws, df, start_row=1, preset="metric")
    """

    def __init__(self, workbook, font: Optional[str] = None,
                 font_size: Optional[int] = None,
                 style: Optional[StyleConfig] = None):
        self.wb = workbook
        self.style = load_style(style)
        # 注意：默认值必须是 None，否则传进来的模板字号会被默认参数"吃掉"
        self.font = font or self.style.font
        self.font_size = font_size if font_size is not None else self.style.font_size
        self._fmt_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # 基础格式
    # ------------------------------------------------------------------
    def _fmt(self, key: str, **kwargs) -> object:
        """带缓存的格式构造（避免重复创建大量 Format 对象）。"""
        if key not in self._fmt_cache:
            base = {"font_name": self.font, "font_size": self.font_size}
            base.update(kwargs)
            self._fmt_cache[key] = self.wb.add_format(base)
        return self._fmt_cache[key]

    def fmt_title(self, ncols: int = 1) -> object:
        return self._fmt("title", bold=True, font_size=13, align="center", valign="vcenter",
                         bg_color=self.style.color("title_bg", "#D9E1F2"), border=1)

    def fmt_header(self) -> object:
        return self._fmt("header", bold=True,
                         bg_color=self.style.color("header_bg", "#4472C4"),
                         font_color=self.style.color("header_font", "#FFFFFF"),
                         align="center", valign="vcenter", border=1, text_wrap=True)

    def fmt_border(self) -> object:
        return self._fmt("border", border=1, valign="vcenter")

    def fmt_num(self, digits: int = 4) -> object:
        return self._fmt(f"num{digits}", num_format="0." + "0" * digits, border=1)

    def fmt_pct(self, digits: int = 2) -> object:
        return self._fmt(f"pct{digits}", num_format="0." + "0" * digits + "%", border=1)

    def fmt_int(self) -> object:
        return self._fmt("int", num_format="#,##0", border=1)

    def fmt_good(self) -> object:
        return self._fmt("good", bg_color=C_GREEN, border=1, num_format="0.0000")

    def fmt_warn(self) -> object:
        return self._fmt("warn", bg_color=C_YELLOW, border=1, num_format="0.0000")

    def fmt_bad(self) -> object:
        return self._fmt("bad", bg_color=C_RED, border=1, num_format="0.0000")

    # ------------------------------------------------------------------
    # 区域定位
    # ------------------------------------------------------------------
    @staticmethod
    def _idx_levels(df: pd.DataFrame, index: bool) -> int:
        """索引占据的列数（MultiIndex 如 channle+month 占 2 列）。"""
        if not index:
            return 0
        return df.index.nlevels if isinstance(df.index, pd.MultiIndex) else 1

    @staticmethod
    def _col_idx(df: pd.DataFrame, col: str, index: bool = False) -> int:
        """列在 Excel 中的 0-based 索引（考虑 index 列占位）。"""
        if col in df.columns:
            return list(df.columns).index(col) + ExcelStyler._idx_levels(df, index)
        if index and col == df.index.name:
            return 0
        if index and col in ("index", "__index__"):
            return 0
        return -1

    def _data_range(self, df: pd.DataFrame, start_row: int, col: str,
                    index: bool = False, header_rows: int = 1,
                    row_map: Optional[Sequence[int]] = None) -> Optional[tuple]:
        """返回 (r1, c, r2, c) 数据区坐标（不含表头）。

        header_rows: 表头行数，两级表头为 2 —— 数据区起点随之后移。
        row_map   : 数据行号 → 真实 Excel 行号的映射（组间插空行时使用）；
                    提供后 r2 取 row_map 末位，使条件格式覆盖插入空行后的真实最后行。
        """
        c = self._col_idx(df, col, index)
        if c < 0:
            return None
        r1 = start_row + header_rows
        r2 = row_map[len(df) - 1] if row_map else r1 + len(df) - 1
        return (r1, c, r2, c)

    def _numeric_cols(self, df: pd.DataFrame, keywords: Sequence[str] = ()) -> List[str]:
        """筛选数值列（可按关键字进一步过滤）。"""
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if keywords:
            cols = [c for c in cols if any(k.lower() in str(c).lower() for k in keywords)]
        return cols

    # ------------------------------------------------------------------
    # 一、标题 / 边框 / 列宽 / 冻结
    # ------------------------------------------------------------------
    def write_title(self, ws, ncols: int, text: str, row: int = 0) -> None:
        """写入跨列居中的大标题。"""
        if ncols <= 0:
            return
        ws.merge_range(row, 0, row, max(ncols - 1, 0), text, self.fmt_title())

    def apply_border(self, ws, df: pd.DataFrame, start_row: int, index: bool = False,
                     header_rows: int = 1,
                     row_map: Optional[Sequence[int]] = None) -> None:
        """为整表（含表头）加边框。

        row_map : 数据行 → 真实 Excel 行的映射（组间插空行时使用）。
                  传入后按每个分组的真实行区间分段加框，**空行不带边框**
                  （对齐模板版式：分隔行是纯粹空白）。
        """
        ncols = len(df.columns) + self._idx_levels(df, index)
        if ncols <= 0:
            return
        opts = {"type": "formula", "criteria": "TRUE", "format": self.fmt_border()}
        last = max(ncols - 1, 0)
        if row_map:
            for (i, j) in self._row_groups(df, index):
                ws.conditional_format(row_map[i], 0, row_map[j], last, opts)
        else:
            ws.conditional_format(start_row, 0, start_row + header_rows + len(df) - 1,
                                  last, opts)

    def apply_header_style(self, ws, df: pd.DataFrame, start_row: int, index: bool = False,
                           header_rows: int = 1) -> None:
        """表头样式（pandas 写入的表头无法直接改格式，用条件格式覆盖）。"""
        ncols = len(df.columns) + self._idx_levels(df, index)
        ws.conditional_format(start_row, 0, start_row + header_rows - 1, max(ncols - 1, 0),
                              {"type": "formula", "criteria": "TRUE", "format": self.fmt_header()})

    def set_width(self, ws, df: pd.DataFrame, min_w: int = 10, max_w: int = 38,
                  index: bool = False) -> None:
        """按内容长度自动设置列宽。"""
        off = self._idx_levels(df, index)
        if index:
            # MultiIndex 不支持 astype(str)，改用逐元素求长度（兼容普通索引与多级索引）
            idx_len = max((len(str(v)) for v in df.index), default=0)
            ws.set_column(0, max(off - 1, 0), max(min_w, min(max_w, idx_len + 4)))
        for i, c in enumerate(df.columns):
            try:
                body = df[c].astype(str).str.len().max()
            except Exception:
                body = 10
            # MultiIndex 列名取最后一级，避免 tuple 字符串把列宽撑爆
            label = str(c[-1] if isinstance(c, tuple) else c)
            w = max(min_w, min(max_w, int(max(body or 0, len(label) * 2, 8)) + 2))
            ws.set_column(i + off, i + off, w)

    def freeze_header(self, ws, start_row: int, col: int = 0) -> None:
        """冻结表头（含上方标题区）。"""
        ws.freeze_panes(start_row + 1, col)

    def set_number_format(self, ws, df: pd.DataFrame, start_row: int,
                          cols: Sequence[str], fmt: str = "0.0000", index: bool = False,
                          header_rows: int = 1) -> None:
        """为指定列设置数字格式。"""
        f = self._fmt(f"custom_{fmt}", num_format=fmt, border=1)
        for c in cols:
            rng = self._data_range(df, start_row, c, index, header_rows)
            if rng:
                ws.set_column(rng[1], rng[3], None, f)

    # ------------------------------------------------------------------
    # 二、数据条 / 色阶 / 图标集（需求4 核心）
    # ------------------------------------------------------------------
    def apply_data_bar(self, ws, df: pd.DataFrame, start_row: int,
                       cols: Optional[Sequence[str]] = None,
                       color: str = C_BLUE, index: bool = False,
                       bar_solid: bool = True, header_rows: int = 1,
                       row_map: Optional[Sequence[int]] = None) -> None:
        """
        数据样条（Data Bar）—— 在单元格内用条形长度直观表达数值大小。

        适用于：样本量、IV、PSI、坏账率等需要看量级的列。
        """
        cols = list(cols) if cols is not None else self._numeric_cols(df)
        opts = {"type": "data_bar", "bar_color": color, "bar_solid": bar_solid}
        for c in cols:
            rng = self._data_range(df, start_row, c, index, header_rows, row_map=row_map)
            if rng:
                for sub in self._split_range_by_row_map(rng, row_map):
                    ws.conditional_format(sub[0], sub[1], sub[2], sub[3], opts)

    def apply_color_scale(self, ws, df: pd.DataFrame, start_row: int,
                          cols: Optional[Sequence[str]] = None,
                          min_color: str = C_GREEN, mid_color: str = C_YELLOW,
                          max_color: str = C_RED, min_type: str = "num",
                          mid_type: str = "num", max_type: str = "num",
                          min_value: float = 0, mid_value: float = 0.1,
                          max_value: float = 0.25, index: bool = False,
                          header_rows: int = 1,
                          row_map: Optional[Sequence[int]] = None) -> None:
        """
        三色阶（Color Scale）—— 用颜色深浅表达数值高低，风控阈值场景首选。

        典型用法：PSI 三色阶（0 绿 → 0.1 黄 → 0.25 红）。
        """
        cols = list(cols) if cols is not None else self._numeric_cols(df)
        opts = {
            "type": "3_color_scale",
            "min_color": min_color, "mid_color": mid_color, "max_color": max_color,
            "min_type": min_type, "mid_type": mid_type, "max_type": max_type,
            "min_value": min_value, "mid_value": mid_value, "max_value": max_value,
        }
        for c in cols:
            rng = self._data_range(df, start_row, c, index, header_rows, row_map=row_map)
            if rng:
                for sub in self._split_range_by_row_map(rng, row_map):
                    ws.conditional_format(sub[0], sub[1], sub[2], sub[3], opts)

    def apply_2color_scale(self, ws, df: pd.DataFrame, start_row: int,
                           cols: Optional[Sequence[str]] = None,
                           min_color: str = "#FFFFFF", max_color: str = C_BLUE,
                           index: bool = False, header_rows: int = 1,
                           row_map: Optional[Sequence[int]] = None) -> None:
        """双色阶（由浅到深），适合"越大越好"型指标。"""
        cols = list(cols) if cols is not None else self._numeric_cols(df)
        opts = {"type": "2_color_scale", "min_color": min_color, "max_color": max_color}
        for c in cols:
            rng = self._data_range(df, start_row, c, index, header_rows, row_map=row_map)
            if rng:
                for sub in self._split_range_by_row_map(rng, row_map):
                    ws.conditional_format(sub[0], sub[1], sub[2], sub[3], opts)

    def apply_icon_set(self, ws, df: pd.DataFrame, start_row: int,
                       cols: Optional[Sequence[str]] = None,
                       icon_style: str = "3_arrows", index: bool = False,
                       reverse: bool = False, header_rows: int = 1,
                       row_map: Optional[Sequence[int]] = None) -> None:
        """
        图标集（Icon Set）—— 用箭头/红绿灯直观标注等级。

        icon_style: 3_arrows / 3_traffic_lights / 4_arrows / 5_quarters 等
        """
        cols = list(cols) if cols is not None else self._numeric_cols(df)
        opts = {"type": "icon_set", "icon_style": icon_style}
        if reverse:
            opts["reverse_icons"] = True
        for c in cols:
            rng = self._data_range(df, start_row, c, index, header_rows, row_map=row_map)
            if rng:
                for sub in self._split_range_by_row_map(rng, row_map):
                    ws.conditional_format(sub[0], sub[1], sub[2], sub[3], opts)

    def highlight_rules(self, ws, df: pd.DataFrame, start_row: int, col: str,
                        rules: Sequence[tuple], index: bool = False,
                        header_rows: int = 1,
                        row_map: Optional[Sequence[int]] = None) -> None:
        """
        阈值高亮：按业务红线给单元格上色。

        Parameters
        ----------
        rules : [(op, value, format_key), ...]，format_key ∈ {good, warn, bad}
                op 支持 '>' '>=' '<' '<=' '==' 'between'
        """
        rng = self._data_range(df, start_row, col, index, header_rows, row_map=row_map)
        if not rng:
            return
        fmt_map = {"good": self.fmt_good(), "warn": self.fmt_warn(), "bad": self.fmt_bad()}
        sub_ranges = self._split_range_by_row_map(rng, row_map)
        for rule in rules:
            op, value, key = rule[0], rule[1], rule[2]
            fmt = fmt_map.get(key, self.fmt_warn())
            if op == "between" and isinstance(value, (list, tuple)):
                opts = {"type": "cell", "criteria": "between",
                        "minimum": value[0], "maximum": value[1], "format": fmt}
            else:
                opts = {"type": "cell", "criteria": op, "value": value, "format": fmt}
            for sub in sub_ranges:
                ws.conditional_format(sub[0], sub[1], sub[2], sub[3], opts)

    # ------------------------------------------------------------------
    # 三、统一入口：预设美化（跨 sheet 复用的关键）
    # ------------------------------------------------------------------
    def beautify(self, ws, df: pd.DataFrame, start_row: int, preset: str = "none",
                 index: bool = False, header_rows: int = 1,
                 row_map: Optional[Sequence[int]] = None) -> None:
        """
        按预设一键美化某个 sheet 的数据区 —— 各 sheet 只需指定 preset 即可复用。

        preset
        ------
        metric : 区分度指标（KS/AUC/Gini/Lift/捕获率）—— 色阶 + KS/PSI 阈值高亮
        psi    : 稳定性指标（PSI/CSI）—— 三色阶 绿→黄→红（0/0.1/0.25）
        bin    : 分箱表 —— 样本量数据条 + 坏率/Lift 色阶
        univar : 单变量分析 —— IV/PSI 数据条 + 阈值高亮
        sample : 样本统计 —— 坏账率色阶 + 样本量数据条
        none   : 仅边框与表头

        row_map : 数据行号 → 真实 Excel 行的映射（组间插空行时使用）。
                  透传给 apply_border / apply_data_bar / apply_color_scale /
                  highlight_rules，使条件格式覆盖到插入空行后的真实最后行。
        """
        if df is None or df.empty:
            return

        self.apply_header_style(ws, df, start_row, index, header_rows)
        self.apply_border(ws, df, start_row, index, header_rows, row_map=row_map)
        self.set_width(ws, df, index=index)

        if preset == "none":
            return

        if preset == "metric":
            # 定版：bad_rate 数据条 + ks 三色阶/阈值 + psi 色阶 + 过拟合告警高亮
            rate = self._numeric_cols(df, ["bad_rate", "坏账率", "坏率"])
            if rate:
                self.apply_data_bar(ws, df, start_row, cols=rate, index=index,
                                    color=C_BLUE, header_rows=header_rows,
                                    row_map=row_map)
            for c in self._numeric_cols(df, ["ks"]):
                self.apply_color_scale(ws, df, start_row, cols=[c], index=index,
                                       min_color=C_RED, mid_color=C_YELLOW, max_color=C_GREEN,
                                       min_value=KS_WARN, mid_value=KS_OK, max_value=0.6,
                                       header_rows=header_rows, row_map=row_map)
                self.highlight_rules(ws, df, start_row, c,
                                     [(">=", KS_OK, "good"), ("between", (KS_WARN, KS_OK), "warn"),
                                      ("<", KS_WARN, "bad")], index=index,
                                     header_rows=header_rows, row_map=row_map)
            for c in self._numeric_cols(df, ["psi"]):
                self.apply_color_scale(ws, df, start_row, cols=[c], index=index,
                                       min_value=0, mid_value=PSI_OK, max_value=PSI_WARN,
                                       header_rows=header_rows, row_map=row_map)
            for c in df.columns:
                if "过拟合" in str(c[-1] if isinstance(c, tuple) else c):
                    self.highlight_rules(ws, df, start_row, c, [("==", '"是"', "bad")],
                                         index=index, header_rows=header_rows,
                                         row_map=row_map)

        elif preset == "heat":
            # 矩阵类（月×月 PSI）：**整块共用一个三色阶刻度**，
            # 逐列单独设刻度会让各月自成一色，无法横向对比漂移
            off = self._idx_levels(df, index)
            r1 = start_row + header_rows
            r2 = (row_map[len(df) - 1] if row_map else r1 + len(df) - 1)
            c2 = off + max(len(df.columns) - 1, 0)
            if row_map:
                for sub in self._split_range_by_row_map((r1, off, r2, c2), row_map):
                    ws.conditional_format(sub[0], sub[1], sub[2], sub[3], {
                        "type": "3_color_scale", "min_color": C_GREEN, "mid_color": C_YELLOW,
                        "max_color": C_RED, "min_type": "num", "mid_type": "num",
                        "max_type": "num", "min_value": 0, "mid_value": PSI_OK,
                        "max_value": PSI_WARN})
            else:
                ws.conditional_format(r1, off, r2, c2, {
                    "type": "3_color_scale", "min_color": C_GREEN, "mid_color": C_YELLOW,
                    "max_color": C_RED, "min_type": "num", "mid_type": "num",
                    "max_type": "num", "min_value": 0, "mid_value": PSI_OK,
                    "max_value": PSI_WARN})

        elif preset == "psi":
            cols = self._numeric_cols(df, ["psi", "PSI"])
            if cols:
                self.apply_color_scale(ws, df, start_row, cols=cols, index=index,
                                       min_color=C_GREEN, mid_color=C_YELLOW, max_color=C_RED,
                                       min_value=0, mid_value=PSI_OK, max_value=PSI_WARN,
                                       header_rows=header_rows, row_map=row_map)
                for c in cols:
                    self.highlight_rules(ws, df, start_row, c,
                                         [(">=", PSI_WARN, "bad"),
                                          ("between", (PSI_OK, PSI_WARN), "warn")],
                                         index=index, header_rows=header_rows,
                                         row_map=row_map)

        elif preset == "bin":
            # 定版：箱内坏率用数据条，区间 lift 用色阶；校准度 gap 用三色阶（负绿/正红）
            rate = self._numeric_cols(df, ["坏率", "bad_rate", "箱内坏率"])
            if rate:
                self.apply_data_bar(ws, df, start_row, cols=rate, index=index,
                                    color=C_BLUE, header_rows=header_rows,
                                    row_map=row_map)
            lift = self._numeric_cols(df, ["lift"])
            if lift:
                self.apply_2color_scale(ws, df, start_row, cols=lift,
                                        min_color="#FFFFFF", max_color=C_ORANGE, index=index,
                                        header_rows=header_rows, row_map=row_map)
            gap = self._numeric_cols(df, ["gap"])
            if gap:
                self.apply_color_scale(ws, df, start_row, cols=gap,
                                       min_color=C_GREEN, mid_color=C_YELLOW,
                                       max_color=C_RED, min_type="min",
                                       mid_type="percentile", max_type="max",
                                       min_value=0, mid_value=50, max_value=0,
                                       index=index, header_rows=header_rows,
                                       row_map=row_map)

        elif preset == "univar":
            for c in self._numeric_cols(df, ["iv", "IV"]):
                self.apply_data_bar(ws, df, start_row, cols=[c], index=index, color=C_BLUE,
                                    header_rows=header_rows, row_map=row_map)
                self.highlight_rules(ws, df, start_row, c,
                                     [(">=", 0.3, "good"), ("between", (0.1, 0.3), "warn"),
                                      ("<", 0.02, "bad")], index=index,
                                     header_rows=header_rows, row_map=row_map)
            for c in self._numeric_cols(df, ["psi", "PSI"]):
                self.apply_color_scale(ws, df, start_row, cols=[c], index=index,
                                       min_color=C_GREEN, mid_color=C_YELLOW, max_color=C_RED,
                                       min_value=0, mid_value=PSI_OK, max_value=PSI_WARN,
                                       header_rows=header_rows, row_map=row_map)
            rate = self._numeric_cols(df, ["badprob", "坏率"])
            if rate:
                self.apply_2color_scale(ws, df, start_row, cols=rate,
                                        min_color="#FFFFFF", max_color=C_RED, index=index,
                                        header_rows=header_rows, row_map=row_map)

        elif preset == "sample":
            # 定版：样本统计只给「坏账率」加数据条（列偏移由列名定位，渠道月表自动落在 G 列）
            rate = self._numeric_cols(df, ["坏账率", "坏率", "bad_rate"])
            if rate:
                self.apply_data_bar(ws, df, start_row, cols=rate, index=index,
                                    color=C_BLUE, header_rows=header_rows,
                                    row_map=row_map)

    # ------------------------------------------------------------------
    # 四、数字格式映射（定版：坏账率类 → 0.0% ；校准度 → 0.000 ；计数 → 整数）
    # ------------------------------------------------------------------
    PCT_EXACT = ("整体坏账率", "坏账率", "坏率", "bad_rate", "箱内坏率",
                 "样本占比", "用户数占比", "占比", "逾期率")
    INT_KEYS = ("样本数", "Cnt", "用户数", "数量", "count")
    FMT3_EXACT = ("expected_pd", "actual_pd", "gap")
    # 单变量分析：占比 / 坏率列 → 0.0%（必须排在 INT_KEYS 之前，
    # 否则 count_distr_* 会被 "count" 命中而错判成整数）
    RATE_PREFIX = ("count_distr", "badprob")
    # PSI 保留原始精度（模板为 General）
    GENERAL_PREFIX = ("psi_",)
    # 两级表头「指标块 → 数字格式」，与期望版一致
    BLOCK_FMT = {"用户数": "General", "逾期用户数": "General",
                 "用户数占比": "0.00%", "逾期率": "0.00%", "LIFT": "General"}

    def _col_label(self, col) -> str:
        """列名取可读标签（两级表头用最后一级）。"""
        return str(col[-1] if isinstance(col, tuple) else col)

    def fmt_for(self, col, dtype=None) -> object:
        """
        按列名（+dtype）推断单元格数字格式。

        优先级：3 位小数 → 百分比 → 单变量占比/坏率前缀 → PSI(General)
                → 整数 → 文本(General) → 4 位小数
        """
        name = self._col_label(col)
        if name in self.FMT3_EXACT:
            return self._fmt("num3", num_format="0.000", border=1)
        if name in self.PCT_EXACT:
            return self.fmt_pct(1)                       # 0.0% → 0.112 显示 11.2%
        if name.startswith(self.RATE_PREFIX):
            return self.fmt_pct(1)                       # count_distr_* / badprob_*
        if name.startswith(self.GENERAL_PREFIX):
            return self._fmt("general", num_format="General", border=1)
        if any(k in name for k in self.INT_KEYS):
            return self.fmt_int()
        if dtype is not None and pd.api.types.is_integer_dtype(dtype):
            return self.fmt_int()
        if dtype is not None and pd.api.types.is_object_dtype(dtype):
            return self._fmt("general", num_format="General", border=1)
        return self.fmt_num(4)

    def fmt_center(self) -> object:
        return self._fmt("center", border=1, valign="vcenter", align="center")

    # ------------------------------------------------------------------
    # 五、块写入器（支持两级表头 / 纵向合并首列 / 逐格数字格式）
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(v):
        """把缺失值与不可写类型转成 Excel 可接受的空值或标量。"""
        if v is None or v is pd.NaT:
            return ""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(v)
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return v

    def write_block(self, ws, df: pd.DataFrame, start_row: int, title: str = "",
                    index: bool = False, header_rows: int = 1, preset: str = "none",
                    merge_col: Optional[str] = None, pct_index_values: Sequence = (),
                    number_formats: Optional[Dict] = None,
                    block_formats: Optional[Dict] = None,
                    bar_blocks: Sequence[str] = (),
                    group_gap: int = 0) -> int:
        """
        手工写入一个数据块：**大标题 + 表头 + 数据 + 逐格数字格式 + 美化**。

        相比 `df.to_excel`，逐格写入才能精确控制百分比/整数格式，
        且支持两级表头与首列纵向合并（分箱表的 data_flag）。

        Parameters
        ----------
        merge_col        : 需纵向合并的列名（如 data_flag），同值只显示一次
        pct_index_values : 索引值命中这些名称时，该行用百分比格式（总览 kv 表用）
        number_formats   : {列名: "0.0%"} 覆盖自动推断
        group_gap        : **组间空行数** —— 按首列（如变量名）连续相同值切组，
                           每组之后空 N 行（最后一组后不空），便于阅读时区分
                           不同变量/分组的内容（对齐模板版式，单变量类表传 1）

        Returns
        -------
        int: 本块数据结束后的下一行行号
        """
        if df is None or df.empty:
            return start_row

        n = len(df)
        idx_levels = self._idx_levels(df, index)
        off = idx_levels
        ncols = len(df.columns) + idx_levels
        number_formats = number_formats or {}

        # 1) 大标题（跨列居中）
        if title:
            ws.merge_range(start_row, 0, start_row, max(ncols - 1, 0), title, self.fmt_title())

        # 2) 表头
        h0 = start_row + 1
        hdr = self.fmt_header()
        idx_names = list(df.index.names) if idx_levels > 1 else [df.index.name or ""]
        if header_rows >= 2 and isinstance(df.columns, pd.MultiIndex):
            # 第一层：相邻同名合并（如"用户数"跨 10 个月）
            i = 0
            cols = list(df.columns)
            while i < len(cols):
                j = i
                while j + 1 < len(cols) and cols[j + 1][0] == cols[i][0]:
                    j += 1
                l0 = self._clean(cols[i][0])
                c1, c2 = i + off, j + off
                if c2 > c1:
                    ws.merge_range(h0, c1, h0, c2, l0, hdr)
                else:
                    ws.write(h0, c1, l0, hdr)
                i = j + 1
            if index:
                for lv, nm in enumerate(idx_names):
                    ws.merge_range(h0, lv, h0 + header_rows - 1, lv, nm or "", hdr)
            # 第二层
            for k, c in enumerate(df.columns):
                ws.write(h0 + 1, k + off, self._clean(c[1]), hdr)
        else:
            if index:
                for lv, nm in enumerate(idx_names):
                    ws.write(h0, lv, nm or "", hdr)
            for k, c in enumerate(df.columns):
                ws.write(h0, k + off, self._clean(c), hdr)

        # 3) 数据（逐格套用数字格式）
        r0 = h0 + header_rows
        # 组间空行：数据行号 → 真实 Excel 行号（空行不写任何内容，含边框与色阶）
        row_map = self._build_row_map(r0, n, self._row_groups(df, index), group_gap)
        col_fmts: List[object] = [self.fmt_border()] * idx_levels
        for c in df.columns:
            nf = number_formats.get(c) or number_formats.get(self._col_label(c))
            # 两级表头：优先按「指标块」（第一层）定格式，如 用户数占比 → 0.00%
            l0 = c[0] if isinstance(c, tuple) else None
            if nf is None and l0:
                nf = (block_formats or {}).get(l0) or self.BLOCK_FMT.get(l0)
            col_fmts.append(self._fmt(f"custom_{nf}", num_format=nf, border=1)
                            if nf else self.fmt_for(c, df[c].dtype))

        merge_idx = None
        if merge_col is not None and merge_col in list(df.columns):
            merge_idx = list(df.columns).index(merge_col) + off

        values = df.to_numpy(dtype=object, na_value=None)
        idx_vals = list(df.index)
        for r in range(n):
            row = values[r]
            rr = row_map[r]
            # kv 表（如总览）：索引命中百分比项时，该行的值同样按百分比显示
            hit = self._clean(idx_vals[r]) in set(pct_index_values)
            val_fmt = self.fmt_pct(1) if hit else None
            idx_fmt = self.fmt_pct(1) if hit else self.fmt_border()
            if index:
                if idx_levels > 1:
                    tup = idx_vals[r]
                    for lv in range(idx_levels):
                        ws.write(rr, lv, self._clean(tup[lv]), idx_fmt)
                else:
                    ws.write(rr, 0, self._clean(idx_vals[r]), idx_fmt)
            for k in range(len(df.columns)):
                if k + off == merge_idx:
                    continue            # 合并列由下方 merge_range 统一写入
                ws.write(rr, k + off, self._clean(row[k]),
                         val_fmt or col_fmts[k + off])

        # 4) 首列纵向合并（分箱表 data_flag 只显示一次）
        #    组间插空行时按「组」为单位合并，不跨空行
        if merge_idx is not None and n > 0:
            col_vals = [self._clean(v) for v in df[merge_col].tolist()]
            i = 0
            while i < n:
                j = i
                # 连续同值才合并；组间空行处（行号不连续）强制断开
                while (j + 1 < n and col_vals[j + 1] == col_vals[i]
                       and row_map[j + 1] == row_map[j] + 1):
                    j += 1
                if j > i:
                    ws.merge_range(row_map[i], merge_idx, row_map[j], merge_idx,
                                   col_vals[i], self.fmt_center())
                else:
                    ws.write(row_map[i], merge_idx, col_vals[i], self.fmt_center())
                i = j + 1

        # 5) 美化（空行不参与：边框/色阶/数据条均按真实行区间分段套用）
        self.apply_header_style(ws, df, h0, index, header_rows)
        if group_gap > 0:
            # 组间空行：边框由下方 beautify 用 row_map 分段接管；两级表头路径
            # 也由 _beautify_two_level 不画边框，所以此处须先画一次
            if header_rows >= 2 and isinstance(df.columns, pd.MultiIndex):
                self.apply_border(ws, df, h0, index, header_rows, row_map=row_map)
        else:
            self.apply_border(ws, df, h0, index, header_rows)
        self.set_width(ws, df, index=index)
        if header_rows >= 2 and isinstance(df.columns, pd.MultiIndex):
            self._beautify_two_level(ws, df, h0, header_rows, index, bar_blocks,
                                     row_map=row_map if group_gap > 0 else None)
        elif preset and preset != "none":
            self.beautify(ws, df, h0, preset=preset, index=index,
                          header_rows=header_rows,
                          row_map=row_map if group_gap > 0 else None)

        return max(row_map) + 1


    def _beautify_two_level(self, ws, df: pd.DataFrame, start_row: int,
                            header_rows: int, index: bool,
                            bar_blocks: Sequence[str] = (),
                            row_map: Optional[Sequence[int]] = None) -> None:
        """
        两级表头块美化（对齐期望版）：

        * 用户数占比 / 逾期率 → 三色阶（比率类，看分布与漂移）
        * bar_blocks 指定的块（如 LIFT）→ 数据条
        * 用户数 / 逾期用户数 → **不美化**（计数类绝对量，色阶无业务含义）

        row_map : 组间插空行时的真实行映射（色阶/数据条只覆盖数据行，不跨空行）
        """
        r1 = start_row + header_rows
        cols = list(df.columns)
        off = self._idx_levels(df, index)
        # 行分组：单变量表=每个特征一段；渠道×月=每个渠道一段；趋势表=整个块一段
        groups = self._row_groups(df, index)

        def rpos(i: int) -> int:
            return row_map[i] if row_map else r1 + i

        scale = {
            "type": "3_color_scale", "min_color": C_GREEN, "mid_color": C_YELLOW,
            "max_color": C_RED, "min_type": "min", "mid_type": "percentile",
            "max_type": "max", "min_value": 0, "mid_value": 50, "max_value": 0}

        i = 0
        while i < len(cols):
            l0 = str(cols[i][0]) if isinstance(cols[i], tuple) else ""
            j = i
            while j + 1 < len(cols):
                nxt = str(cols[j + 1][0]) if isinstance(cols[j + 1], tuple) else ""
                if nxt != l0:
                    break
                j += 1
            c1, c2 = i + off, j + off
            if l0 in ("用户数占比", "逾期率"):
                # 组内归一化：每个特征单独成色阶，才能看出该特征哪个分箱更集中/更坏
                for (g1, g2) in groups:
                    ws.conditional_format(rpos(g1), c1, rpos(g2), c2, dict(scale))
            elif l0 in bar_blocks:
                for (g1, g2) in groups:
                    ws.conditional_format(rpos(g1), c1, rpos(g2), c2,
                                          {"type": "data_bar", "bar_color": C_BLUE,
                                           "bar_solid": True})
            i = j + 1

        # 单变量分析：按特征分组，组内 count_distr 色阶 / badprob 数据条
        self._beautify_by_group(ws, df, start_row, header_rows, index, row_map=row_map)

    @staticmethod
    def _split_range_by_row_map(rng: tuple, row_map: Optional[Sequence[int]]) -> List[tuple]:
        """
        把一整个 (r1,c1,r2,c2) 区间按 row_map 切分成多个不跨空行的子区间。
        当 row_map 中存在连续行号断开时（即组间空行），就在此处断开条件格式范围。
        """
        if not row_map:
            return [rng]
        r1, c1, _, c2 = rng
        out, cur_start = [], r1
        prev = r1
        for i in range(1, len(row_map)):
            r = row_map[i]
            if r != prev + 1:
                out.append((cur_start, c1, prev, c2))
                cur_start = r
            prev = r
        out.append((cur_start, c1, prev, c2))
        return out

    @staticmethod
    def _build_row_map(start: int, n: int, groups: Sequence[tuple],
                       gap: int) -> List[int]:
        """
        生成「数据行号 → 真实 Excel 行号」映射：每个分组之后空 `gap` 行
        （最后一组后不空）。gap<=0 时退化为连续行号。
        """
        out = list(range(start, start + n))
        if gap <= 0 or not groups:
            return out
        cursor = start
        for k, (i, j) in enumerate(groups):
            for r in range(i, j + 1):
                out[r] = cursor
                cursor += 1
            if k < len(groups) - 1:
                cursor += gap
        return out

    @staticmethod
    def _group_keys(df: pd.DataFrame, index: bool = False) -> List[str]:
        """
        行分组键：索引优先（渠道×月块取索引第一级 = 渠道名），否则用首列
        （单变量表取 variable）。
        """
        if index:
            idx = df.index
            if isinstance(idx, pd.MultiIndex):
                return [str(v[0]) for v in idx]
            return [str(v) for v in idx]
        return df.iloc[:, 0].astype(str).tolist()

    def _row_groups(self, df: pd.DataFrame, index: bool = False) -> List[tuple]:
        """按分组键**连续**相同值切分的行区间列表 [(起, 止), ...]（相对行号）。"""
        if df.empty:
            return []
        first = self._group_keys(df, index)
        out, i = [], 0
        while i < len(first):
            j = i
            while j + 1 < len(first) and first[j + 1] == first[i]:
                j += 1
            out.append((i, j))
            i = j + 1
        return out

    def _beautify_by_group(self, ws, df: pd.DataFrame, start_row: int,
                           header_rows: int, index: bool,
                           row_map: Optional[Sequence[int]] = None) -> None:
        """
        单变量分析专用：按首列（特征名）分组，**组内** count_distr_* 三色阶、
        badprob_* 数据条。

        分组内归一化才能看出「该特征的哪个分箱占比高 / 坏率高」；
        若整列统一刻度，会被特征之间的量级差异掩盖。
        """
        cols = list(df.columns)
        off = self._idx_levels(df, index)
        rate_cols = [k for k, c in enumerate(cols)
                     if self._col_label(c).startswith("count_distr")]
        bar_cols = [k for k, c in enumerate(cols)
                    if self._col_label(c).startswith("badprob")]
        if not rate_cols and not bar_cols:
            return

        r0 = start_row + header_rows

        def rpos(i: int) -> int:
            return row_map[i] if row_map else r0 + i

        for (i, j) in self._row_groups(df, index):
            for k in rate_cols:
                ws.conditional_format(rpos(i), k + off, rpos(j), k + off, {
                    "type": "3_color_scale", "min_color": C_GREEN, "mid_color": C_YELLOW,
                    "max_color": C_RED, "min_type": "min", "mid_type": "percentile",
                    "max_type": "max", "min_value": 0, "mid_value": 50, "max_value": 0})
            for k in bar_cols:
                ws.conditional_format(rpos(i), k + off, rpos(j), k + off,
                                      {"type": "data_bar", "bar_color": C_BLUE,
                                       "bar_solid": True})


class ReportBuilder:
    """
    Excel 报告构建器。

    根治旧版缺陷：`df.to_excel(writer, startrow=1)` 未传 `sheet_name`，
    pandas 默认写入 "Sheet1"，导致**所有表堆叠进同一个 sheet**、而手动
    `add_worksheet` 建的标题 sheet 全空、数据错位丢失。

    此处强制 `sheet_name`，并在写入后回取 worksheet 做美化。
    """

    def __init__(self, path: str, engine: str = "xlsxwriter",
                 style: Optional[StyleConfig] = None):
        self.path = path
        self.engine = engine
        # 模板生效：刷新本模块配色/阈值常量（必须在创建 styler 之前）
        self.style = apply_style(style)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # xlsxwriter 的 nan/inf 容错开关需放在 options 内（直接作为 kwargs 会被拒绝）
        try:
            self.writer = pd.ExcelWriter(
                path, engine=engine,
                engine_kwargs={"options": {"nan_inf_to_errors": True}},
            )
        except TypeError:  # 低版本 xlsxwriter 不支持该选项时降级
            self.writer = pd.ExcelWriter(path, engine=engine)
        self.styler = ExcelStyler(self.writer.book, style=self.style)
        self._sheets: List[str] = []

    # ------------------------------------------------------------------
    def add_sheet(self, name: str, df: pd.DataFrame, title: Optional[str] = None,
                  start_row: int = 1, preset: str = "none", index: bool = False,
                  freeze: bool = True, header_rows: int = 1,
                  merge_col: Optional[str] = None, pct_index_values: Sequence = (),
                  number_formats: Optional[Dict] = None,
                  block_formats: Optional[Dict] = None,
                  bar_blocks: Sequence[str] = (),
                  group_gap: int = 0) -> Optional[object]:
        """
        写入一个 sheet（含标题、数据、美化）。

        Parameters
        ----------
        name      : sheet 名称（**必传**，这是修复旧缺陷的关键）
        df        : 数据表
        title     : 大标题，默认用 sheet 名
        preset    : 美化预设 metric/psi/bin/univar/sample/none
        index     : 是否写出索引
        header_rows      : 表头行数（两级表头传 2）
        merge_col        : 需纵向合并的列（如分箱表 data_flag）
        pct_index_values : 索引值命中时用百分比格式（总览 kv 表）
        group_gap        : 按首列分组后，组之间的空行数（单变量类表传 1）
        """
        if df is None or df.empty:
            logger.warning("sheet[%s] 数据为空，跳过", name)
            return None

        safe_name, ws = self._new_worksheet(name)
        self.styler.write_block(ws, df, 0, title=title or safe_name, index=index,
                                header_rows=header_rows, preset=preset,
                                merge_col=merge_col, pct_index_values=pct_index_values,
                                number_formats=number_formats,
                                block_formats=block_formats, bar_blocks=bar_blocks,
                                group_gap=group_gap)
        if freeze:
            self.styler.freeze_header(ws, header_rows)
        self._sheets.append(safe_name)
        return ws

    def _new_worksheet(self, name: str):
        """建 sheet 并登记到 writer.sheets（供后续取用）。"""
        safe = self._safe_sheet_name(name)
        ws = self.writer.book.add_worksheet(safe)
        self.writer.sheets[safe] = ws
        return safe, ws

    def add_multi_block_sheet(self, name: str, blocks: Sequence[Dict],
                              gap: Optional[int] = None,
                              freeze: bool = False) -> Optional[object]:
        """
        一个 sheet 内自上而下排列多个数据块（块间空 `gap` 行）。

        定版版式要求：同类型指标合并进同一 sheet —— 逐月与渠道月、建模效果与渠道效果等。
        每块结构：块标题 + 表头 + 数据。

        blocks 元素
        -----------
        title / df / preset / index / header_rows / merge_col / pct_index_values /
        number_formats / group_gap（块内按分组键再空 N 行，如渠道×月按渠道分隔）
        """
        blocks = [b for b in (blocks or [])
                  if b.get("df") is not None and not b["df"].empty]
        if not blocks:
            logger.warning("sheet[%s] 无有效数据块，跳过", name)
            return None
        # 未显式指定时取模板默认（定版=2 行，compact=1 行）
        gap = self.style.gap_rows if gap is None else gap

        safe, ws = self._new_worksheet(name)
        cursor = 0
        for b in blocks:
            cursor = self.styler.write_block(
                ws, b["df"], cursor,
                title=b.get("title", ""),
                index=b.get("index", False),
                header_rows=b.get("header_rows", 1),
                preset=b.get("preset", "none"),
                merge_col=b.get("merge_col"),
                pct_index_values=b.get("pct_index_values", ()),
                number_formats=b.get("number_formats"),
                block_formats=b.get("block_formats"),
                bar_blocks=b.get("bar_blocks", ()),
                group_gap=b.get("group_gap", 0),
            )
            cursor += gap
        if freeze:
            self.styler.freeze_header(ws, 1)
        self._sheets.append(safe)
        return ws

    def add_long_sheet(self, name: str, df: pd.DataFrame, chunk_rows: int = 50000,
                       preset: str = "none", index: bool = False) -> List[str]:
        """
        大表分页写入多个 sheet（如单变量分析超出 Excel 单页限制时）。

        Returns
        -------
        实际写入的 sheet 名列表
        """
        if df is None or df.empty:
            return []
        total = len(df)
        pages = max(1, math.ceil(total / chunk_rows))
        written = []
        for i in range(pages):
            sub = df.iloc[i * chunk_rows: (i + 1) * chunk_rows]
            sname = name if pages == 1 else f"{name}_{i + 1}"
            self.add_sheet(sname, sub, title=f"{name}（第{i + 1}/{pages}页）",
                           preset=preset, index=index)
            written.append(sname)
        return written

    def add_image_sheet(self, name: str, images: Sequence[str],
                        col_width: int = 60, row_height: int = 30) -> Optional[object]:
        """插入图表的 sheet（每张图纵向排列）。"""
        images = [p for p in images if p and os.path.exists(p)]
        if not images:
            return None
        ws = self.writer.book.add_worksheet(self._safe_sheet_name(name))
        self.writer.sheets[self._safe_sheet_name(name)] = ws
        ws.set_column(0, 0, col_width)
        row = 0
        for p in images:
            try:
                ws.insert_image(row, 0, p, {"x_scale": 0.85, "y_scale": 0.85})
                row += row_height
            except Exception as exc:
                logger.warning("图片插入失败 %s：%s", p, exc)
        self._sheets.append(self._safe_sheet_name(name))
        return ws

    @staticmethod
    def _safe_sheet_name(name: str) -> str:
        """Excel sheet 名限制：≤31 字符，不含 : \\ / ? * [ ]"""
        bad = ':\\/?*[]'
        for ch in bad:
            name = str(name).replace(ch, "_")
        return name[:31]

    # ------------------------------------------------------------------
    @property
    def sheets(self) -> List[str]:
        return list(self._sheets)

    def close(self) -> str:
        self.writer.close()
        logger.info("报告已生成：%s（%d 个 sheet）", self.path, len(self._sheets))
        return self.path

    def __enter__(self) -> "ReportBuilder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["ExcelStyler", "ReportBuilder", "apply_style",
           "PSI_OK", "PSI_WARN", "KS_OK", "KS_WARN"]
