#
# otoge_rl_project: Makefile
#
# このファイルは、プロジェクトのセットアップ、コード品質管理（リント、フォーマット）など、
# 頻繁に使用するコマンドを標準化するために使用されます。
# CI/CD パイプラインとローカル開発環境の両方で一貫した操作を提供します。
#

# .PHONY: ターゲットが実際のファイル名と衝突しないように設定
.PHONY: help install lint fmt check-fmt

# デフォルトターゲット (例: `make`)
help:
	@echo "利用可能なコマンド:"
	@echo "  make install    - uv を使用して、プロジェクトと開発用の依存関係をインストールします。"
	@echo "  make lint       - ruff を使用して、コードの静的解析（リント）を実行します。"
	@echo "  make fmt        - ruff を使用して、コードを自動整形します。（ローカル用）"
	@echo "  make check-fmt  - ruff を使用して、コードの整形が必要かチェックします。（CI用）"

# 依存関係のインストール (uv sync を使用して uv.lock に基づく)
install:
	@echo "--- 依存関係を 'pyproject.toml' と 'uv.lock' からインストールします ---"
	uv pip install -e .[dev]

# Linter (コード検査)
lint:
	@echo "--- Linter (ruff check) を実行します ---"
	uv run ruff check src/

# Formatter (コード整形) - ローカル用
# このコマンドはファイルを直接変更します。
fmt:
	@echo "--- Formatter (ruff format) を実行し、ファイルを修正します ---"
	uv run ruff format src/

# Formatter Check (整形検査) - CI用
# このコマンドはファイルを変更せず、整形が必要な場合はエラーを返します。
check-fmt:
	@echo "--- Formatter (ruff format --check) を実行し、整形が必要かチェックします ---"
	uv run ruff format --check src/
