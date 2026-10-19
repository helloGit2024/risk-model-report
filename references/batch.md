# 批量出报告（多模型 / 多期 / 多标签）

---

## 1. 三种"多"的落地方式

| 诉求 | 做法 |
|---|---|
| 多模型对比 | 每个模型一个 job，换 `model`（分数列） |
| 多期对比（月度/季度） | 每期一个 job，换 `rnt`（结果表） |
| 多标签（M1/M2/M3） | 每个标签一个 job，换 `target` —— **即"单标签多跑几次"**，不要试图在一份报告里混多标签 |

---

## 2. batch.json 结构

```json
{
  "defaults": {
    "config": "cfg.json",
    "model": "D05_lgb_pkl",
    "rnt": "rnt.csv",
    "wide": "wide.csv",
    "importance": "imp.csv",
    "top": 10,
    "style": "default",
    "out_dir": "out/batch",
    "eval_univariate": true
  },
  "jobs": [
    {"name": "2024Q4", "rnt": "rnt_2024Q4.csv"},
    {"name": "2025Q1", "rnt": "rnt_2025Q1.csv", "style": "compact"},
    {"name": "M1",  "target": "y_m1", "out": "out/batch/D05_M1.xlsx"},
    {"name": "M3",  "target": "y_m3"},
    {"name": "fast", "wide": null, "importance": null}
  ]
}
```

### defaults / jobs 字段

| 字段 | 说明 |
|---|---|
| `config` | 配置 JSON 路径，或内联 dict |
| `config_overrides` | 覆盖配置里的个别字段（defaults 与 job 都可写，job 优先） |
| `model` | 模型分数列名 |
| `rnt` / `wide` / `importance` | 输入路径；job 里显式写 `null` 表示**该 job 不用它**（触发降级） |
| `target` | 覆盖 `config.target`，用于多标签 |
| `top` | Top-N 特征 |
| `style` | 版式模板 |
| `out_dir` | 输出目录；`out` 缺省时落到 `{out_dir}/{name}.xlsx` |
| `out` | 显式输出路径（优先级高于 out_dir） |
| `img_dir` | 图表目录（缺省 = 报告同级 `{model}_imgs`） |
| `eval_univariate` | 是否做单变量分析 |
| `data_flag` | 数据集划分数组 |

---

## 3. 运行

```bash
cd scripts
python generate_report.py --batch ../assets/batch.example.json
```

日志形态：
```
批量开始：共 4 个任务
启动风控模型报告生成 | RUN_ID=… | 模板=default
批量进度 1/4：D05_full 成功
批量进度 3/4：D05_no_feature 成功
批量结束：成功 4 / 失败 0
```

**失败隔离**：单个 job 抛错只记 ERROR 并继续下一个，不会中断整批。退出码 `0`=全成功，`1`=至少一个失败。
要看每个 job 的明细，读日志或给 `run_batch()` 的返回值（`{"total", "ok", "fail", "results"}`）。

---

## 4. 耗时参考（20 万行 / 26 特征 / 8 核）

| 场景 | 耗时 |
|---|---|
| 完整报告（16 sheet，含单变量分析） | ~21 s |
| compact 模板 | ~16 s |
| 降级（无宽表/重要性，12 sheet） | ~4 s |

批量 N 份 ≈ 单份 × N。想加速就 `--no-univariate` 或减少 `top`。
