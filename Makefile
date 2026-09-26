# GraphForge 常用快捷目标
# 说明：Windows 环境若未安装 make，可直接照抄命令执行。

PYTHON ?= python
PKG    := graphforge
ART    := artifacts

.PHONY: help install install-core test lint demo bench doctor clean

help: ## 显示帮助
	@echo "可用目标: install / install-core / test / lint / demo / bench / doctor / clean"

install: ## 安装核心依赖 + 开发依赖
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

install-core: ## 仅安装核心依赖（可选依赖失败不影响系统）
	$(PYTHON) -m pip install "numpy>=1.26" "scipy>=1.11" "scikit-learn>=1.4" "networkx>=3.1" "pandas>=2.0" "joblib>=1.3" pytest

test: ## 运行全部单测
	$(PYTHON) -m pytest -q -W ignore::UserWarning

lint: ## 编译检查（不依赖第三方 linter）
	$(PYTHON) -m compileall -q $(PKG)

doctor: ## 打印后端可用性表
	$(PYTHON) -m $(PKG).cli doctor

demo: ## 端到端演示，落盘 artifacts/
	$(PYTHON) examples/run_demo.py

bench: ## 基准评测（默认数据集 / 全部方法）
	$(PYTHON) -m $(PKG).cli benchmark --datasets sbm,karate --methods spectral,deepwalk,node2vec,grarep

clean: ## 清理缓存与产物
	@rm -rf $(ART) .pytest_cache
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name "*.py[cod]" -delete
