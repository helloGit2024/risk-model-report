# 排障手册

**排查顺序**：`check_env.py` → `preflight_check.py` → 出报告 → `validate_output.py`。90% 的问题在前两步就能定位。

---

## 1. 报告生成失败

| 现象 | 原因 | 处理 |
|---|---|---|
| `KeyError: 'y'` / `KeyError: 'dateback'` | 配置里的列名与实际数据不一致 | 改 `config` 的 `target`/`date`/`data_flag`；先跑 `preflight_check.py`（E2 会直接指出缺哪一列） |
| `FileNotFoundError` | 路径写错或用了相对路径但换了工作目录 | 用绝对路径；批量配置里的相对路径以**运行时 CWD** 为基准 |
| `PermissionError` | 目标 xlsx 正被 Excel 打开 | 关闭文件后重跑；或改输出路径 |
| 报 `nan_inf_to_errors` 相关 TypeError | xlsxwriter 版本过低 | `pip install -U xlsxwriter`（代码已自动降级，但建议升级） |
| 报 `Invalid value '否' for dtype 'float64'` / `LossySetitemError` | pandas ≥ 3.0 未适配（float 列写字符串） | `check_env.py` 现在会**直接阻断**并提示降级；`pip install "pandas>=1.5,<2.2"` |
| `check_env.py` 结论"环境不可用：版本不兼容" | 依赖版本超出 `requirements.txt` 的上界 | 按提示降级；`check_env` 只校验下限的历史已修，现在上界同样拦截 |
| 内存暴涨 / 极慢 | 宽表太大且开了单变量分析 | `--no-univariate`，或减小 `--top` |

---

## 2. 报告内容不对

| 现象 | 原因 | 处理 |
|---|---|---|
| 少了 4 个 sheet | 缺宽表 / 缺重要性 / `--no-univariate` | 看日志的"输入降级"行；补输入或接受降级（设计内行为） |
| lift 全部 < 1、捕获率异常低 | 分数方向反了（分高=优质客户） | preflight 会提前报 **W7**；取反或 `max - score` 后再传 `--model`（见 data-contract.md §5） |
| preflight 报 W1「缺少 oot」但数据里明明有 | 旧版只抽样前 20 万行，oot 排在文件尾部被漏掉 | 已修：W1 改为全量单列判定，看输出的 `flag_counts_source` 是否为「全量」 |
| 校准度 gap 全是 NaN / 异常 | 传入的是**分数**不是**概率** | 评分卡先用 `scorecard.score_to_p` 转概率 |
| PSI 全是 0 | 只有一个数据集，或 `data_flag` 列值不标准 | 确认列里有 train/test/oot；PSI 以 train 为基准 |
| 逐月指标全空 | 日期列解析失败，或没有 month 列 | `preflight`（W6）；确保 `date` 可解析，或直接给 `month` |
| 图表中文显示为方框 | 环境无可用的中文字体 | `check_env.py` 会提示；安装 SimHei/Microsoft YaHei，或改模板 `font` |
| 数据条/色阶没出现 | xlsxwriter 未安装或版本 <3.0；也可能用 openpyxl 打开了文件（openpyxl 能读但渲染差异） | `pip install -U xlsxwriter`；用 Excel/WPS 查看 |
| openpyxl 读报告时告警 `Unknown extension` | xlsxwriter 写了 openpyxl 不认识的扩展属性 | 无害，已在本仓库脚本内屏蔽该告警 |

---

## 3. 版式问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 换了模板但字号没变 | 旧代码路径没传 style（已修复）；或模板 JSON 里没写 `font_size` | 确认用新版 `generate_report.py`；`--dump-style` 看实际生效值 |
| 空行数量不符 | `gap_rows`（块间）与 `group_gap`（组间）是两个参数 | 见 style-templates.md §2 |
| 条件格式跨过空行 | 自研 sheet 时直接传了连续区间 | `validate_output.py` 会报 ERROR；写入时务必传 `group_gap` |
| sheet 顺序变了 | 模板 `sheet_order` 生效 | 检查模板；不想要就删掉该字段 |

---

## 4. 批量问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 某个 job 失败但整批继续 | 设计内的失败隔离 | 看该 job 的 ERROR 日志；`run_batch()` 返回的 `results` 里有 `error` 字段 |
| job 输出路径不对 | `out` 与 `out_dir` 混用 | `out` 优先；都不给才用 `{out_dir}/{name}.xlsx` |
| 多标签 job 结果一样 | 忘了给每个 job 设不同 `target` | 每个标签一个 job（见 batch.md §1） |

---

## 5. 数值回归

改过 `riskpkg/` 后，跑指标回归脚本确认口径没被改坏：

```bash
cd scripts
python run_validation.py          # 28 项断言：真实数据（若存在）+ Mock 边界
```

Mock 边界覆盖 6 类异常分支（坏率极端、缺失、常量特征、单值特征、时间断点、空分组），是改代码后的第一道防线。
