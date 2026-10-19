# 版式模板（可换模板）

配色 / 字体 / 空行 / 阈值刻度 / Sheet 编排全部外置为模板，改模板**不需要动代码**。

---

## 1. 三种用法

```bash
# A. 内置模板
python generate_report.py ... --style default     # 定版：微软雅黑 10 号，块间空 2 行，组间空 1 行
python generate_report.py ... --style compact     # 紧凑：9 号，块间空 1 行，组间不空行

# B. 自定义 JSON（企业模板）
python generate_report.py ... --style ../assets/styles/corp.json

# C. 库调用
from riskpkg import load_style
get_report(..., style=load_style("compact"))
```

导出现有模板作为改造起点：
```bash
python generate_report.py --dump-style ../assets/styles/corp.json          # 导出 default
python generate_report.py --style compact --dump-style ./my.json           # 导出 compact
```

---

## 2. 模板字段

```json
{
  "name": "corp",
  "font": "微软雅黑",
  "font_size": 10,
  "colors": {
    "green": "#63BE7B", "yellow": "#FFEB84", "red": "#F8696B",
    "blue": "#638EC6", "light_blue": "#B4C6E7", "orange": "#F4B183",
    "header_bg": "#4472C4", "header_font": "#FFFFFF", "title_bg": "#D9E1F2"
  },
  "thresholds": {"psi_ok": 0.1, "psi_warn": 0.25, "ks_ok": 0.3, "ks_warn": 0.2},
  "gap_rows": 2,
  "group_gap": 1,
  "col_width": 14,
  "freeze": true,
  "sheet_order": ["总览", "建模效果统计", "模型分10箱", "PSI热力矩阵", "校准度"],
  "sheet_include": ["总览", "建模效果统计", "模型分10箱", "PSI热力矩阵", "校准度"]
}
```

| 字段 | 作用 | 缺省 |
|---|---|---|
| `colors.*` | 色阶与数据条配色、表头/标题底色 | 定版配色 |
| `font` / `font_size` | 正文字体 | 微软雅黑 / 10 |
| `gap_rows` | 同一 sheet 内**块**之间的空行 | 2 |
| `group_gap` | 同一表内**分组**（data_flag / 变量 / 渠道）之间的空行 | 1 |
| `thresholds` | 色阶刻度（不改告警逻辑，见 thresholds.md） | 行业默认 |
| `sheet_order` | 重排 sheet 顺序；未列出的排最后并保持原序 | 内置 16 项顺序 |
| `sheet_include` | 只输出这些 sheet（**裁剪报告**） | 全部 |

省略的字段自动继承 `default`，所以企业模板可以只写"要改的那几项"。

---

## 3. 裁剪报告的两种方式

| 方式 | 适用 | 做法 |
|---|---|---|
| 按业务裁剪 | 只想给业务方看"效果 + 稳定性" | 模板里写 `sheet_include` |
| 按输入裁剪 | 没有宽表/重要性 | 自动降级到 12 项（见 data-contract.md §4），无需配置 |

示例（5 页精简版）：
```json
{"name": "biz", "sheet_include": ["总览", "建模效果统计", "模型分10箱", "PSI热力矩阵", "校准度"]}
```

---

## 4. 生效链路（排查样式不生效时看这里）

```
CLI --style  →  load_style()  →  StyleConfig
                                    ├─ ReportBuilder(style=…) → excel.apply_style() 刷新配色/阈值全局
                                    ├─ ExcelStyler(style=…)   → 字体/字号/表头底色
                                    ├─ add_multi_block_sheet(gap=None) → 用 style.gap_rows
                                    └─ _sheet_specs(group_gap=style.group_gap) → 分组空行
```

注意：`apply_style` 是**模块级刷新**，因此每次 `ReportBuilder` 构造都会重设；批量串行出报告不会串味。
