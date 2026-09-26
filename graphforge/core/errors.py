"""错误体系：统一基类 ``GraphForgeError`` + E1xx–E5xx 错误码。

约定（见 docs/architecture.md §9.1）：
    * 每个错误码对应一个异常类，类名见下表；
    * ``str(exc)`` 渲染为 ``"[E303] message (key=value; hint=...)"``；
    * 区间划分：E1xx data / E2xx preprocess / E3xx graph / E4xx training·hpo·eval
      / E5xx config·pipeline·cli。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

__all__ = [
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
]


class GraphForgeError(Exception):
    """GraphForge 全部异常的基类。

    Attributes:
        code: 形如 ``"E303"`` 的错误码。
        message: 人类可读的中文描述。
        context: 附加上下文（``hint`` 为其中最常用的键）。
    """

    code: str = "E000"
    default_message: str = "GraphForge 内部错误"

    def __init__(
        self,
        message: Optional[str] = None,
        code: Optional[str] = None,
        **ctx: Any,
    ) -> None:
        """构造异常。

        Args:
            message: 描述信息（第一个位置参数）；为 None 时取类属性 :attr:`default_message`。
            code: 错误码（仅关键字）；为 None 时取类属性 :attr:`code`。
            **ctx: 附加上下文，会以 ``key=value`` 形式追加到字符串尾部。
        """
        self.code: str = code or type(self).code
        self.message: str = message or type(self).default_message
        self.context: Dict[str, Any] = dict(ctx)
        super().__init__(self._render())

    def _render(self) -> str:
        """渲染 ``str(exc)`` 的目标文本。"""
        text = f"[{self.code}] {self.message}"
        if self.context:
            tail = "; ".join(f"{key}={value}" for key, value in self.context.items())
            text = f"{text} ({tail})"
        return text

    @property
    def hint(self) -> Optional[str]:
        """返回上下文中的 ``hint`` 字段（不存在则为 None）。"""
        value = self.context.get("hint")
        return None if value is None else str(value)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {"code": self.code, "message": self.message, "context": dict(self.context)}

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"{type(self).__name__}({self._render()})"


# ---------------------------------------------------------------- E1xx data
class DatasetNotFoundError(GraphForgeError):
    """E101 数据集不存在或未注册。"""

    code = "E101"
    default_message = "数据集不存在"


class InvalidGraphFormatError(GraphForgeError):
    """E102 图格式非法（节点数不匹配、缺列、类型错误等）。"""

    code = "E102"
    default_message = "图格式非法"


class EmptyGraphError(GraphForgeError):
    """E103 空图（0 节点 / 0 边）。"""

    code = "E103"
    default_message = "空图"


class LabelMismatchError(GraphForgeError):
    """E104 标签数量/取值与图不一致。"""

    code = "E104"
    default_message = "标签与图不匹配"


class UnsupportedFileFormatError(GraphForgeError):
    """E105 不支持的文件格式（后缀）。"""

    code = "E105"
    default_message = "不支持的文件格式"


# ---------------------------------------------------------- E2xx preprocess
class InvalidSplitRatioError(GraphForgeError):
    """E201 划分比例非法（不为正、和不为 1 等）。"""

    code = "E201"
    default_message = "划分比例非法"


class InsufficientSamplesError(GraphForgeError):
    """E202 样本量不足以完成划分 / CV / 评估。"""

    code = "E202"
    default_message = "样本量不足"


class InvalidAdjacencyError(GraphForgeError):
    """E203 邻接矩阵非法（非方阵、非稀疏、含 NaN 等）。"""

    code = "E203"
    default_message = "邻接矩阵非法"


class NegativeSamplingError(GraphForgeError):
    """E204 负样本采样失败（候选耗尽 / 与正样本重叠）。"""

    code = "E204"
    default_message = "负样本采样失败"


# --------------------------------------------------------------- E3xx graph
class BackendUnavailableError(GraphForgeError):
    """E301 可选后端不可用（karateclub / node2vec 等）。"""

    code = "E301"
    default_message = "后端不可用"


class EmbeddingFitError(GraphForgeError):
    """E302 嵌入训练失败（未 fit / 形状错误 / 数值异常）。"""

    code = "E302"
    default_message = "嵌入训练失败"


class InvalidEmbeddingParamError(GraphForgeError):
    """E303 嵌入参数非法（含保留参数泄漏）。"""

    code = "E303"
    default_message = "嵌入参数非法"


class ConvergenceError(GraphForgeError):
    """E304 迭代过程未收敛（谱分解 / SGD）。"""

    code = "E304"
    default_message = "迭代未收敛"


class UnknownMethodError(GraphForgeError):
    """E305 未知方法名（嵌入器 / 任务 / 数据集 / 指标）。"""

    code = "E305"
    default_message = "未知方法名"


# ------------------------------------------------------ E4xx training/hpo/eval
class EstimatorBuildError(GraphForgeError):
    """E401 下游估计器构造失败。"""

    code = "E401"
    default_message = "估计器构造失败"


class EstimatorFitError(GraphForgeError):
    """E402 下游估计器训练失败。"""

    code = "E402"
    default_message = "估计器训练失败"


class InvalidSearchSpaceError(GraphForgeError):
    """E403 搜索空间定义非法。"""

    code = "E403"
    default_message = "搜索空间非法"


class HPOBackendUnavailableError(GraphForgeError):
    """E404 HPO 后端不可用。"""

    code = "E404"
    default_message = "HPO 后端不可用"


class MetricComputationError(GraphForgeError):
    """E405 指标计算失败。"""

    code = "E405"
    default_message = "指标计算失败"


class MissingMetricError(GraphForgeError):
    """E406 结果中缺少指定指标。"""

    code = "E406"
    default_message = "缺少指定指标"


# ----------------------------------------------------- E5xx config/pipeline/cli
class ConfigError(GraphForgeError):
    """E501 配置解析 / 取值非法。"""

    code = "E501"
    default_message = "配置错误"


class PipelineStateError(GraphForgeError):
    """E502 管线状态非法（未初始化 / 重复执行）。"""

    code = "E502"
    default_message = "管线状态错误"


class CLIArgumentError(GraphForgeError):
    """E503 命令行参数非法。"""

    code = "E503"
    default_message = "命令行参数错误"


class BenchmarkAbortedError(GraphForgeError):
    """E504 基准评测中止（全部后端不可用）。"""

    code = "E504"
    default_message = "基准评测中止"


class SerializationError(GraphForgeError):
    """E505 序列化 / 落盘失败。"""

    code = "E505"
    default_message = "序列化失败"


def _build_registry() -> Dict[str, Type[GraphForgeError]]:
    """扫描 :class:`GraphForgeError` 的子类，构建 ``code -> class`` 映射。"""
    registry: Dict[str, Type[GraphForgeError]] = {}

    def _walk(cls: Type[GraphForgeError]) -> None:
        for sub in cls.__subclasses__():
            code = getattr(sub, "code", None)
            if isinstance(code, str) and code != "E000":
                registry[code] = sub
            _walk(sub)

    _walk(GraphForgeError)
    return registry


ERROR_CLASSES: Dict[str, Type[GraphForgeError]] = _build_registry()


def error_codes() -> List[str]:
    """返回已注册的全部错误码（升序）。"""
    return sorted(ERROR_CLASSES)


def error_class(code: str) -> Type[GraphForgeError]:
    """按错误码取异常类。

    Args:
        code: 形如 ``"E303"`` 的错误码。

    Returns:
        对应的异常类；未注册时返回 :class:`GraphForgeError` 基类。
    """
    return ERROR_CLASSES.get(code, GraphForgeError)


def raise_error(code: str, message: Optional[str] = None, **ctx: Any) -> GraphForgeError:
    """按错误码构造异常（由调用方 ``raise``）。

    Args:
        code: 错误码。
        message: 描述信息；None 时取该码的默认描述。
        **ctx: 附加上下文。

    Returns:
        已构造好的异常实例（**不会自动抛出**，便于调用方决定 raise 时机）。
    """
    cls = error_class(code)
    if message is None and cls is GraphForgeError:
        message = f"未知错误码 {code}"
    return cls(message, code=code, **ctx)
