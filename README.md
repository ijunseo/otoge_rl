## 1. プロジェクトの目的

このプロジェクトは、深層強化学習（PPO）を用いて、カスタムビルドされたリズムゲーム（Otoge）をプレイするエージェントを学習させることを目的としています。

## 2. 開発の動機と仮説

開発のきっかけは、松尾研究室の深層強化学習講義です。講義の中で、「ゲームAIの学習に使用するユーザーデータは、そのユーザーの傾向（例：攻撃的なプレイヤー、慎重なプレイヤー、熟練者、初心者）によってAIの振る舞いが変わる」という知見に触発されました。

これをリズムゲームに当てはめ、以下の仮説を検証したいと考えました。

1. **仮説 A:** 難しい譜面（高難易度パターン）のみで学習したエージェントは、簡単な譜面も問題なく（あるいは人間以上に）完璧にプレイできるか？
    
2. **仮説 B:** 簡単な譜面のみで学習したエージェントは、未経験の難しい譜面（人間なら対応が困難）をどの程度プレイできるのか？
    

このプロジェクトは、これらの問いを実証的に検証し考察するための実験環境です。

## 3. プロジェクトの技術的特徴

このリポジトリは、Jupyter Notebook でのプロトタイピングから、再現性と拡張性を備えた Python プロジェクト（V8計画）へ移行したものです。

- `uv` による高速な依存関係管理 (`pyproject.toml`, `uv.lock`)
    
- `JSON` (`game/config.json`, `agent/ppo_config.json`) による設定ファイル分離
    
- `GitHub Actions` と `Makefile` (`ruff`) による CI/CD とコード品質の自動化
    
- `argparse` (`main.py`) を利用した、動的なハイパーパラメータ・オーバーライド
    

## 4. プロジェクト構造 (V8 計画)

V8 計画に基づく最終的なフォルダ構造です。

```
otoge_rl_project/
├── .github/                 # (CI/CD) GitHub Actions ワークフロー
│   └── workflows/
│       └── quality.yml      #           - 'push' 時に 'uv sync', 'lint', 'check-fmt' を自動実行
│
├── src/                     # (Source) uv が認識するPythonモジュールルート
│   ├── agent/               # (モジュール 1) エージェントと学習ロジック
│   │   ├── __init__.py      #              - callbacks, utils を 'agent' パッケージレベルで公開
│   │   ├── callbacks.py     #              - TrainingRewardLogger, EvalWithAccuracy
│   │   ├── ppo_config.json  #              - (設定 1) PPO ハイパーパラメータ "基本値"
│   │   └── utils.py         #              - make_env ヘルパー関数 (環境ラッパー)
│   │
│   ├── rhythm_game/         # (モジュール 2) リズムゲーム環境ロジック
│   │   ├── __init__.py      #              - RhythmGameEnv を 'rhythm_game' パッケージレベルで公開
│   │   ├── config.json      #              - (設定 2) ゲームルール/スコア "基本値"
│   │   └── env.py           #              - RhythmGameEnv クラス実装
│   │
│   ├── main.py              # (エントリーポイント 1) メイン学習スクリプト
│   │                      #                 - 実行: 'uv run python -m main'
│   │
│   └── evaluate.py          # (エントリーポイント 2) 学習済みモデル評価スクリプト
│                          #                 - 実行: 'uv run python -m evaluate'
│
├── .gitignore               # Git 無視リスト (outputs/, .venv/, pycache/, *.mp4)
├── Makefile                 # (CI/CD) 'make install', 'make lint', 'make fmt' コマンド定義 (ruff ベース)
├── pyproject.toml           # (設定 3) uv 依存関係 (stable-baselines3, ruff) および 'src' レイアウト定義
├── uv.lock                  # (CI/CD) uv 依存関係の正確なバージョンをロック
└── README.md                # (本文書) プロジェクトの説明
```

## 5. 開発ワークフロー (必須)

プロジェクトを開始するための必須ステップです。

### ステップ 1: 環境のセットアップ

リポジトリをクローンした後、`uv` を使用して開発環境をセットアップし、`uv.lock` に基づいて依存関係をインストールします。

```
# 1. 依存関係のインストール (開発[dev]用も含む)
#    Makefile は内部的に 'uv pip install -e .[dev]' を実行します。
make install

# 2. (オプション) uv 仮想環境を手動で有効化する場合
# source .venv/bin/activate
```

## 6. 主な開発コマンド (Makefile)

`Makefile` は、CI とローカル開発のコマンドを標準化します。

- **`make install`** `pyproject.toml` と `uv.lock` に基づき、`uv` を使用して `.venv` 仮想環境のセットアップと依存関係の同期（インストール）を行います。
    
- **`make fmt`** コードフォーマッター (`ruff format`) を実行し、`src/` 以下のコードスタイルを自動で修正します。
    
- **`make lint`** リンター (`ruff check`) を実行し、`src/` 以下のコードのエラーや潜在的な問題を検出します。
    
- **`make check-fmt`** CI 用のコマンドです。コードを修正せず、フォーマットが正しいか**チェックのみ**行います。
    

## 7. 学習の実行 (メイン)

学習は `src/main.py` スクリプトを通じて実行されます (`uv run python -m main ...`)。

### 基本的な実行

`--output_path` は必須です。これが実験結果（ログ、モデル、パラメータ）を保存する一意のフォルダ名になります。

```
# 基本設定 (JSON の値) で学習を実行
# 'outputs/exp_baseline' フォルダが自動で作成されます
uv run python -m main --output_path "outputs/exp_baseline"
```

### パラメータのオーバーライド (CLI)

`main.py` は `argparse` を使用しており、`agent/ppo_config.json` や `rhythm_game/config.json` の**全ての値を CLI で上書き（オーバーライド）**できます。

```
# JSON の 'gamma' と 'learning_rate' を上書きして実験
uv run python -m main \
    --output_path "outputs/exp_lr_test" \
    --learning_rate 0.0003 \
    --gamma 0.95

# 別の設定で実験
uv run python -m main \
    --output_path "outputs/exp_fast_run" \
    --total_timesteps 1000000 \
    --n_envs 8 \
    --density "medium"
```

### 学習成果物 (Reproducibility)

`--output_path` (例: `outputs/exp_lr_test/`) には以下が保存されます:

1. **`hyperparameters.txt`**: 再現性のための最重要ファイル。この実験で使用された**全てのパラメータ（JSON基本値 + CLI上書き値）**が JSON 形式で保存されます。
    
2. **`logs/`**: `TensorBoard` ログファイル。
    
3. **`models/`**: `best_model.zip` (評価コールバックによる最良モデル) と `final_model.zip` (最終モデル)。
    
4. **`training_plot.png`**: 学習報酬と評価正解率のグラフ（`main.py` が自動生成）。
    

## 8. 評価の実行

学習済みのモデルを評価し、プレイ動画を録画します。

```
# 'outputs/exp_lr_test' の学習結果をロードして評価
uv run python -m evaluate --run_path "outputs/exp_lr_test"
```

このコマンドは `exp_lr_test/hyperparameters.txt` を読み込んで環境設定を復元し、`exp_lr_test/models/best_model.zip` をロードします。 評価動画は `exp_lr_test/evaluation_video_...mp4` として同じフォルダに保存されます。

## 9. CI/CD (コード品質)

`.github/workflows/quality.yml` は、GitHub リポジトリへの `push` イベントが発生するたびに自動で実行されます。

1. `uv` 環境をセットアップ
    
2. `make install` で依存関係を `uv sync`
    
3. `make lint` でコードエラーをチェック
    
4. `make check-fmt` でコードスタイルをチェック
    

いずれかのチェックが失敗すると CI は失敗し、品質が保証されていないコードのマージを防ぎます。