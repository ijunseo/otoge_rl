#
# otoge_rl_project: Makefile
#
# (V9.4) 修正:
# 1. 'install' ターゲットを README.md (V9.4) の 5.A ステップと一致させます。
# 2. CI 環境 (GPU不要) のための 'install-ci' ターゲットを維持します。
# 3. (重要) CUDA バージョンを 'make' 実行時に動的に指定できるように変更します。
#    例: make install CUDA_VERSION=cu128 (デフォルト)
#    例: make install CUDA_VERSION=cu118 (古いドライバ用)
#

# PyTorch の CUDA バージョンを変数として定義
# 5070 Ti (CUDA 12.9) 環境に基づき、'cu128' をデフォルト (default) に設定
# '?=' は、変数が指定されなかった場合のみこの値を採用することを意味します。
CUDA_VERSION ?= cu128
PYTORCH_INDEX_URL = https://download.pytorch.org/whl/$(CUDA_VERSION)

.PHONY: install install-ci lint fmt check-fmt

# [V9.4] 修正: ローカル開発環境 (Windows/macOS) 用のインストール
# (README 5.A, ステップ 5.1 + 5.2 と一致)
install:
	@echo "--- [V9.4] ステップ 5.1: GPU 版 PyTorch ($(CUDA_VERSION)) をインストールします ---"
	uv pip install torch --index-url $(PYTORCH_INDEX_URL)
	@echo "--- [V9.4] ステップ 5.2: プロジェクトの依存関係 (dev) をインストールします ---"
	uv pip install -e .[dev]

# [V9.3] CI (Linux) 環境用のインストール
# (GPUは不要なため、標準 (CPU版) PyTorch をインストールします)
install-ci:
	@echo "--- [V9.3] CI 環境 (CPU) 用の PyTorch をインストールします ---"
	uv pip install torch
	@echo "--- [V9.3] CI 環境用の依存関係 (dev) をインストールします ---"
	uv pip install -e .[dev]

# -----------------------------------------------------------------
# (V8.5 と同様) Ruff を使用した Lint と Fmt
# -----------------------------------------------------------------

# ruff を使用したリンティング (コードエラーのチェック)
lint:
	@echo "--- 'ruff check' (Lint) を実行中... ---"
	uv run ruff check src/

# ruff を使用したフォーマット (コードスタイルの自動修正)
fmt:
	@echo "--- 'ruff format' (Format) を実行中... ---"
	uv run ruff format src/

# ruff を使用したフォーマットチェック (CI用: 修正せずにチェックのみ)
check-fmt:
	@echo "--- 'ruff format --check' (Format Check) を実行中... ---"
	uv run ruff format --check src/

