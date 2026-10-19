# 变更记录

## v3.0.0 — Skill 化（当前）

**新增能力**

| 项 | 说明 |
|---|---|
| CLI 入口 | `generate_report.py` 支持命令行；库调用 `get_report()` 保留，签名向后兼容 |
| 批量出报告 | `--batch batch.json`，支持多模型 / 多期 / 多标签，失败隔离 |
| 版式模板 | `riskpkg/style.py` + `assets/styles/*.json`；配色/字体/空行/阈值刻度/Sheet 编排全部可换 |
| Sheet 编排 | 模板 `sheet_order` 重排、`sheet_include` 裁剪 |
| 阈值可配 | 配置 JSON / CLI `--psi-ok` 等 / 模板 `thresholds` 三处覆盖 |
| 环境自检 | `check_env.py`（核心/可选依赖 + 中文字体） |
| 输入前置校验 | `preflight_check.py`（E1~E3 致命、W1~W6 降级提示，3 秒拦错） |
| 产出验收 | `validate_output.py`（sheet 齐全、条件格式不跨空行、关键列存在） |
| 脱敏样例 | `assets/sample/` 全流程可跑通（n=20000，含 6 类异常分支） |

**结构变化**

- 源码迁入 `scripts/`（唯一源码：`scripts/riskpkg/` + 6 个入口脚本）
- 文档迁入 `references/`，样例与模板迁入 `assets/`
- 新增 `img_dir` 参数，图表目录不再硬编码
- 日志改为幂等初始化，批量场景不再重复挂 handler
- 单个 sheet 写入失败不再中断整份报告（记 WARN 后继续）

**兼容性**

- `get_report()` 参数顺序与语义不变，新增参数均带默认值
- `ReportBuilder(path, engine)` 用法不变，`style` 为新增可选参数
- 默认模板 `default` 与 v2 定版版式**逐像素一致**

---

## v2.0.0 — 分层重构

- 散落的 `reportFun1.py` / `reportFun2.py` / `Utils.py` 重构为 `riskpkg/` 12 个分层模块，依赖单向、无循环引用
- 指标口径全库锁死（见 metric-definitions.md），消除重复实现与口径漂移
- 修复 9 项 P0/P1 缺陷（缺 import、`to_excel` 未指定 sheet_name、从不生成 month 列、lift 头尾方向反、KS 的 ylim 参数缺失等）
- 旧兼容层已彻底删除，入口只有 `riskpkg.*` 与 `get_report`

---

## v1.x — 定版版式

- 16 个 sheet 定稿；块间空 2 行、组间空 1 行
- 美化预设 metric / psi / bin / univar / sample / heat 统一
- LIFT 数据条、gap 三色阶、条件格式不跨空行
