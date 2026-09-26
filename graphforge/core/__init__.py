"""GraphForge core 层（L0）：零反向依赖，只允许 import stdlib / typing / numpy / scipy。

对外统一导出 types / errors / config / interfaces / logging / utils 的公共符号，
上层一律从这里取用，避免深路径 import。

作者：晨星
"""

from __future__ import annotations

from graphforge.core import config as config
from graphforge.core import errors as errors
from graphforge.core import interfaces as interfaces
from graphforge.core import logging as logging
from graphforge.core import types as types
from graphforge.core import utils as utils

from graphforge.core.config import (
    GraphForgeConfig,
    config_context,
    get_config,
    reset_config,
    set_config,
    update_config,
)
from graphforge.core.errors import (
    BackendUnavailableError,
    BenchmarkAbortedError,
    CLIArgumentError,
    ConfigError,
    ConvergenceError,
    DatasetNotFoundError,
    EmbeddingFitError,
    EmptyGraphError,
    EstimatorBuildError,
    EstimatorFitError,
    GraphForgeError,
    HPOBackendUnavailableError,
    InsufficientSamplesError,
    InvalidAdjacencyError,
    InvalidEmbeddingParamError,
    InvalidGraphFormatError,
    InvalidSearchSpaceError,
    InvalidSplitRatioError,
    LabelMismatchError,
    MetricComputationError,
    MissingMetricError,
    NegativeSamplingError,
    PipelineStateError,
    SerializationError,
    UnknownMethodError,
    UnsupportedFileFormatError,
    error_class,
    error_codes,
    raise_error,
)
from graphforge.core.interfaces import (
    Embedder,
    EstimatorFactory,
    GraphSource,
    Optimizer,
    Reporter,
    Scorer,
    Splitter,
    TaskRunner,
)
from graphforge.core.logging import get_logger, setup_logging
from graphforge.core.types import (
    DEFAULT_PRIMARY_METRIC,
    BackendStatus,
    BenchmarkResult,
    Embedding,
    EvalResult,
    GraphData,
    HPOSpec,
    ParamSpec,
    SearchSpace,
    Split,
    SplitStrategy,
    TaskSpec,
    TaskType,
    TrialResult,
)
from graphforge.core.utils import (
    derive_seed,
    ensure_dir,
    make_rng,
    make_py_rng,
    stable_json,
    timer,
)

__all__ = [
    # 子模块
    "config",
    "errors",
    "interfaces",
    "logging",
    "types",
    "utils",
    # config
    "GraphForgeConfig",
    "get_config",
    "set_config",
    "reset_config",
    "update_config",
    "config_context",
    # errors
    "GraphForgeError",
    "DatasetNotFoundError",
    "InvalidGraphFormatError",
    "EmptyGraphError",
    "LabelMismatchError",
    "UnsupportedFileFormatError",
    "InvalidSplitRatioError",
    "InsufficientSamplesError",
    "InvalidAdjacencyError",
    "NegativeSamplingError",
    "BackendUnavailableError",
    "EmbeddingFitError",
    "InvalidEmbeddingParamError",
    "ConvergenceError",
    "UnknownMethodError",
    "EstimatorBuildError",
    "EstimatorFitError",
    "InvalidSearchSpaceError",
    "HPOBackendUnavailableError",
    "MetricComputationError",
    "MissingMetricError",
    "ConfigError",
    "PipelineStateError",
    "CLIArgumentError",
    "BenchmarkAbortedError",
    "SerializationError",
    "raise_error",
    "error_codes",
    "error_class",
    # interfaces
    "GraphSource",
    "Embedder",
    "Splitter",
    "EstimatorFactory",
    "Scorer",
    "TaskRunner",
    "Optimizer",
    "Reporter",
    # logging
    "get_logger",
    "setup_logging",
    # types
    "TaskType",
    "BackendStatus",
    "SplitStrategy",
    "GraphData",
    "Embedding",
    "TaskSpec",
    "Split",
    "EvalResult",
    "BenchmarkResult",
    "ParamSpec",
    "SearchSpace",
    "HPOSpec",
    "TrialResult",
    "DEFAULT_PRIMARY_METRIC",
    # utils
    "derive_seed",
    "make_rng",
    "make_py_rng",
    "timer",
    "ensure_dir",
    "stable_json",
]
