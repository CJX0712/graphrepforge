# GraphRepForge

> 图机器学习评测基准：Node Classification / Link Prediction / Embedding Benchmark 三任务统一接口，跨模型公平比较。
> 作者：**晨星** · 版本：v0.1.0

## 1. 这是什么

GraphRepForge 把"造图 → 预处理 → 图嵌入 → 下游任务 → 评测"串成一条可复现管线，用**统一指标方向（全部"越大越好"）**让不同方法可以直接排序比较。

- **三个任务**：`node_classification`（primary = `macro_f1`）、`link_prediction`（primary = `roc_auc`）、`embedding_benchmark`（复用下游指标）
- **四个内置嵌入方法**：`spectral` / `deepwalk` / `node2vec` / `grarep`
- **可选后端**：`karateclub`（官方实现，可用时优先）、`optuna`（HPO，可用时优先）
- **确定性**：所有随机入口吃 `random_state`，种子派生走 `hashlib`，同 seed 结果可复现

## 2. 快速开始

```bash
# 1) 安装（可选依赖缺失不影响运行，系统自动降级）
pip install -r requirements.txt

# 2) 体检：查看后端可用性
python -m graphforge.cli doctor

# 3) 单条管线
python -m graphforge.cli run --dataset sbm --method node2vec --task node_classification

# 4) 基准评测
python -m graphforge.cli benchmark --datasets sbm,karate --methods spectral,node2vec

# 5) 端到端演示（落盘 artifacts/）
python examples/run_demo.py

# 6) 单测
pytest -q -W ignore::UserWarning
```

支持的 CLI 子命令：`doctor` / `list-datasets` / `list-methods` / `embed` / `run` / `benchmark`。

## 3. 配置（环境变量覆盖）

| 变量 | 默认 | 含义 |
|---|---|---|
| `GRAPHFORGE_RANDOM_STATE` | 42 | 全局随机种子 |
| `GRAPHFORGE_LOG_LEVEL` | INFO | 日志级别 |
| `GRAPHFORGE_DIM` | 128 | 嵌入维度 |
| `GRAPHFORGE_EMBED_BACKEND` | auto | auto / internal / karateclub |
| `GRAPHFORGE_HPO_BACKEND` | auto | auto / optuna / random / grid / none |
| `GRAPHFORGE_HPO_TRIALS` | 20 | 搜索次数 |
| `GRAPHFORGE_OUTPUT_DIR` | artifacts | 产物目录 |
| `GRAPHFORGE_N_JOBS` | 1 | 并行度 |

## 4. 目录结构

```
graphforge/
├── core/          L0 零反向依赖（types / errors / config / interfaces / logging / utils）
├── data/          合成图 + 文件载入 + 数据集注册表
├── preprocess/    邻接构建 / 划分 / 特征工程
├── graph/         领域层：embeddings（spectral/deepwalk/node2vec/grarep）+ tasks
├── training/      下游估计器工厂 / CV / trainer
├── hpo/           超参搜索（optuna / grid / random）
├── eval/          指标注册表（方向恒 +1）/ 排名 / 报告
├── pipeline/      编排层：GraphPipeline + Benchmark
└── cli.py         入口
```

## 5. 合规声明（重要）

**本项目内置的嵌入实现均为"经典论文复现"，不是自研 SOTA，也不做任何性能宣称。**

采用内置实现的原因：`karateclub`（PyPI 仅提供 sdist）与 `node2vec`（依赖 gensim → 要求 `numpy<2`）在 Windows / Python 3.13 环境**实测安装失败**（`metadata-generation-failed × numpy`），图嵌入领域缺失基础对照基线则 benchmark 无意义，故按原论文实现：

| 内置实现 | 对应论文 |
|---|---|
| `Node2VecEmbedder` | Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks*, KDD 2016 |
| `DeepWalkEmbedder` | Perozzi et al., *DeepWalk*, KDD 2014（训练部分改用 Mikolov NIPS 2013 负采样，替代原论文 hierarchical softmax） |
| `SpectralEmbedder` | Belkin & Niyogi, *Laplacian Eigenmaps*, Neural Computation 2003 |
| `GraRepEmbedder` | Cao et al., *GraRep*, CIKM 2015 |

约束：
1. 内置实现只做**基线对照**，benchmark 输出中 `backend` 列一律标注 `internal`；
2. 不做任何性能宣称，不引入近三年的新方法；
3. 当 `karateclub` 可用时（`GRAPHFORGE_EMBED_BACKEND=karateclub`）优先走官方实现，内置实现仅作 fallback。

## 6. Python API

```python
from graphforge.pipeline import GraphPipeline, Benchmark
from graphforge.core.types import TaskSpec, TaskType, HPOSpec

# 单条管线
result = GraphPipeline(
    dataset="sbm",              # sbm / karate / grid / lfr 或 GraphData 实例
    task=TaskType.NODE_CLASSIFICATION,
    method="node2vec",          # spectral / deepwalk / node2vec / grarep
    dim=32,
).run()
print(result.primary_metric, result.primary_value)   # macro_f1 0.999

# 网格基准
bench = Benchmark(
    datasets=("sbm", "karate"),
    methods=("spectral", "deepwalk", "node2vec", "grarep"),
    tasks=(TaskType.NODE_CLASSIFICATION, TaskType.LINK_PREDICTION),
    dim=32,
)
outcome = bench.run()
outcome.to_json("artifacts/benchmark.json")

# HPO（optuna 可用时 TPE，否则自动降级 random/grid）
result = GraphPipeline(
    dataset="karate", task=TaskType.NODE_CLASSIFICATION, method="node2vec",
    dim=16, hpo=HPOSpec(backend="auto", n_trials=20),
).run()
```

## 7. 量化基线（demo，seed=42，可复现）

`python examples/run_demo.py` 在 sbm_demo（4 社区 × 40 节点 SBM）与 karate 上的实测输出：

| dataset | method | node_classification (macro_f1↑) | link_prediction (roc_auc↑) |
|---|---|---|---|
| sbm_demo | spectral | 0.9686 | 0.9960 |
| sbm_demo | deepwalk | 1.0000 | 0.9436 |
| sbm_demo | node2vec | 1.0000 | 0.9436 |
| sbm_demo | grarep | 0.9686 | 0.9609 |
| karate | spectral | 0.6250 | 0.9911 |
| karate | deepwalk | 1.0000 | 0.8133 |
| karate | node2vec | 1.0000 | 0.8133 |
| karate | grarep | 1.0000 | 1.0000 |

总耗时约 54s（Windows / Python 3.13 / CPU）。同 seed 两次运行 JSON 逐字节一致（除 elapsed_sec）。

## 8. 部署（Docker）

```bash
docker build -t graphforge .
docker run --rm graphforge            # 默认执行 demo，产物在 /app/artifacts
docker run --rm graphforge python -m graphforge.cli doctor
```

## 9. 测试与质量

- `pytest -q -W ignore::UserWarning`：**128 个用例全绿**
- `tests/test_layering.py`：静态 AST 断言依赖单向无环（core 零反向依赖）
- `tests/test_determinism.py`：同 seed 复现、异 seed 随机性生效
- `tests/test_eval_metrics.py`：指标与 sklearn 直算逐位一致（递归防护证明）

## 10. 架构文档

完整设计（13 章，1052 行）：[docs/architecture.md](docs/architecture.md)，
含选型依据、接口清单、Mermaid 时序图 / 类图、T1–T5 任务分解与 12 条踩坑清单。
