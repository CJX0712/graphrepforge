"""GraphForge 训练层：下游估计器工厂 / 交叉验证 / 任务训练器。

作者：晨星
"""

from __future__ import annotations

from graphforge.training import cv as cv
from graphforge.training import estimators as estimators
from graphforge.training import trainer as trainer

from graphforge.training.cv import cross_val_scores, make_folds
from graphforge.training.estimators import (
    DEFAULT_ESTIMATOR,
    ESTIMATOR_REGISTRY,
    list_estimators,
    make_estimator,
    register_estimator,
    validate_estimator_params,
)
from graphforge.training.trainer import (
    edge_features,
    fit_predict,
    train_link_predictor,
    train_node_classifier,
)

__all__ = [
    "estimators",
    "cv",
    "trainer",
    "ESTIMATOR_REGISTRY",
    "DEFAULT_ESTIMATOR",
    "register_estimator",
    "list_estimators",
    "make_estimator",
    "validate_estimator_params",
    "cross_val_scores",
    "make_folds",
    "edge_features",
    "fit_predict",
    "train_node_classifier",
    "train_link_predictor",
]
