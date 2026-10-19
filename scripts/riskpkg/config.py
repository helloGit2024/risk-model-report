# -*- coding: utf-8 -*-
"""
配置层（ReportConfig）
================================================
* 报告配置统一建模，支持 pydantic 校验
* 支持从 JSON 文件加载，也支持 dict 直接注入（解耦对 main_config.json 的硬依赖）
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False
    BaseModel = object
    # 必须返回 default：类属性即"字段默认值"。
    # 若这里返回 None，下面的降级 __init__ 取到的默认值全是 None，
    # 构造不报错但配置静默失效（target/阈值全为 None），比直接崩溃更难查。
    Field = lambda default=None, **kw: default  # noqa: E731


class ReportConfig(BaseModel if _HAS_PYDANTIC else object):
    """
    模型报告配置。字段与 main_config.json 对齐，额外补充报告专属配置。

    兼容说明：未安装 pydantic 时退化为普通对象，行为一致。
    """

    # ---- 数据字段映射 ----
    target: str = Field(default="y", description="标签列名，1=坏样本 0=好样本")
    date: str = Field(default="dateback", description="日期列名")
    data_flag: str = Field(default="data_flag", description="数据集划分列名")
    channle_name: Optional[str] = Field(default="channle", description="渠道列名")
    keys: Optional[List[str]] = Field(default=None, description="主键列")

    # ---- 报告开关 ----
    is_channle: bool = Field(default=False, description="是否按渠道拆分报告")
    eval_univariate: bool = Field(default=True, description="是否做单变量分析")
    other_y_lst: Optional[List[str]] = Field(default=None, description="其他标签列（M1/M2/M3 多标签）")
    other_label_value: Optional[List] = Field(default=None, description="灰样本标签取值")

    # ---- 报告参数 ----
    bins: int = Field(default=10, description="模型分分箱数")
    bins_lst: Optional[List[int]] = Field(default=None, description="多套分箱数，如 [10, 20]")
    importance_top: int = Field(default=30, description="特征分析取前 N 个重要特征")
    model_name: str = Field(default="model", description="模型名称（用于报告标题）")

    # ---- 风控阈值 ----
    psi_threshold: float = Field(default=0.1, description="PSI 观察阈值")
    psi_alert_threshold: float = Field(default=0.25, description="PSI 漂移告警阈值")
    iv_threshold: float = Field(default=0.02, description="IV 入模阈值")
    ks_overfit_threshold: float = Field(default=0.05, description="KS 过拟合衰减阈值")

    if _HAS_PYDANTIC:
        model_config = {"extra": "allow", "protected_namespaces": ()}
    else:  # pragma: no cover
        class Config:
            extra = "allow"

        def __init__(self, **kwargs: Any) -> None:
            """无 pydantic 时的降级构造函数。

            旧版只有类属性默认值，没有 __init__，`cls(**data)` 会抛
            `TypeError: ReportConfig() takes no arguments`，即"降级行为一致"从未被验证。
            这里显式接收 kwargs，并把默认值与传入值合并到实例属性。
            """
            # 自动收集类属性（= 字段默认值），不硬编码字段名列表，
            # 否则以后新增字段会静默漏掉，又是同一类 bug。
            defaults = {
                k: v for k, v in vars(type(self)).items()
                if not k.startswith("_")
                and not callable(v)
                and not isinstance(v, (classmethod, staticmethod, property))
            }
            defaults.update(kwargs)
            for k, v in defaults.items():
                setattr(self, k, v)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportConfig":
        """从 dict 构建；自动忽略冲突键，避免 got multiple values 错误。"""
        data = dict(data or {})
        # 兼容旧配置的拼写差异
        alias = {
            "channle": "channle_name",
            "channel": "channle_name",
            "psiThreshold": "psi_threshold",
            "ivThreshold": "iv_threshold",
            "importance_top_n": "importance_top",
        }
        for old, new in alias.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "ReportConfig":
        """从 main_config.json 加载配置。"""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        logger.info("配置加载完成：%s", path)
        return cls.from_dict(raw)

    def as_dict(self) -> Dict[str, Any]:
        if _HAS_PYDANTIC:
            return self.model_dump()
        return dict(self.__dict__)


__all__ = ["ReportConfig"]
