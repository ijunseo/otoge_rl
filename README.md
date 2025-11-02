# Otoge RL Agent (強化学習リズムゲームエージェント)

Jupyter Notebook でプロトタイピングされた強化学習リズムゲームエージェントを、再現性と拡張性を備えた Python プロジェクトにリファクタリングしたものです。

このプロジェクトは `uv` による高速な依存関係管理、`JSON` による設定分離、`GitHub Actions` による CI/CD、そして動的な `CLI` オーバーライドを特徴としています。

## 1. プロジェクトの目的と動機

### 目的

強化学習（PPO）を用いて、動的に生成される譜面をプレイするリズムゲームエージェントを学習させます。

### 開発動機 (仮説)

松尾研究室の深層強化学習講義で、エージェントの学習に使用するユーザーデータ（例：乱暴なユーザー、マナーの良いユーザー）の傾向によって、学習後のエージェントの振る舞いが変わるという知見に触発されました。

これをリズムゲームに応用し、以下の仮説を検証することを目的としています。

1. **仮説1:** 非常に難しいパターン（高密度）のみで学習させたエージェントは、簡単な譜面も完璧に（あるいは人間以上に）プレイできるか？
    
2. **仮説2:** 簡単な譜面のみで学習させたエージェントは、初めて見る難しいパターンにどれだけ対応できるか？（人間であれば難しいと予想されます）
    

このリポジトリは、これらの仮説を迅速に検証し、考察するためのアジャイルな開発環境です。

## 2. プロジェクト構造 (V-Final)

V8, V9 計画を経て、以下の `src-layout` 構造に最適化されています。

```
otoge_rl_project/
├── .github/                 # (CI/CD) GitHub Actions ワークフロー
│   └── workflows/
│       └── quality.yml      #           - 'push' 時に 'make lint', 'make check-fmt' を実行
│
├── src/                     # (Source) Python モジュールルート
│   ├── agent/               # (モジュール 1) PPO エージェントと学習ロジック
│   │   ├── __init__.py      #              - callbacks, utils を 'agent' パッケージレベルで公開
│   │   ├── callbacks.py     #              - 学習/評価のカスタムコールバック
│   │   ├── ppo_config.json  #              - (設定 1) PPO ハイパーパラメータの "基本値"
│   │   └── utils.py         #              - make_env ヘルパー (環境ラッパー)
│   │
│   ├── rhythm_game/         # (モジュール 2) リズムゲーム環境ロジック
│   │   ├── __init__.py      #              - RhythmGameEnv を 'rhythm_game' パッケージレベルで公開
│   │   ├── config.json      #              - (設定 2) ゲームルール/スコアの "基本値"
│   │   └── env.py           #              - RhythmGameEnv クラスの実装
│   │
│   ├── main.py              # (メイン実行 1) 学習スクリプト
│   ├── evaluate.py          # (メイン実行 2) 評価・動画保存スクリプト
│   └── play_random.py       # (メイン実行 3) ランダムテスト用スクリプト
│
├── .gitignore               # Git 無視リスト (outputs/, .venv/, *.mp4 など)
├── Makefile                 # (CI/CD) 'make lint', 'make fmt' コマンド定義 (ruff)
├── pyproject.toml           # (設定 3) uv 依存関係管理 (stable-baselines3, ruff, torch)
├── README.md                # (本文書)
└── uv.lock                  # 'uv pip install' で自動生成されるロックファイル
```

## 3. (必須) NVIDIA ドライバの確認

本プロジェクトは `torch` (PyTorch) の CUDA (GPU) 版を使用します。最新の `torch` ビルド (`cu124+`) は、最新の NVIDIA ドライバを必要とします。

1. PowerShell またはコマンドプロンプトで `nvidia-smi` を実行します。
    
2. `Driver Version: 550.xx` (またはそれ以上) および `CUDA Version: 12.4+` (例: 12.9) が表示されることを確認してください。
    
3. もしドライバが古い場合 (`53x.xx` など)、[NVIDIA 公式サイト](https://www.nvidia.com/Download/index.aspx "null") または GeForce Experience から 5000 シリーズ (5070 Ti) 用の最新ドライバをインストールし、**システムを再起動**してください。
    

## 4. 開発ワークフロー: インストールと実行

### A. 環境構築 (初回のみ)

V9.1 計画に基づき、GPU (`torch`) と `uv` の依存関係競合を避けるため、以下の **5 ステップ** を順守してください。

**1. リポジトリのクローンと移動** (重要: OneDrive や Dropbox フォルダを避け、`C:\dev\` のような短い ASCII パスにクローンしてください)

```
git clone [https://github.com/.../otoge_rl_project.git](https://github.com/.../otoge_rl_project.git)
cd otoge_rl_project
```

**2. (初回のみ) `.venv` と `uv.lock` のクリーンアップ** (リポジトリに `uv.lock` が含まれている場合、または以前のインストールに失敗した場合に実行します)

```
Remove-Item -Path .\.venv -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path .\.uv.lock -Force -ErrorAction SilentlyContinue
```

**3. (初回のみ) `uv` 仮想環境の生成**

```
uv venv
```

**4. 仮想環境の有効化** (PowerShell を開くたびに実行します)

```
.\.venv\Scripts\Activate.ps1
```

**5. (必須) V9.1 依存関係のインストール (2段階プロセス)** (GPU `torch` を先にインストールし、その後 `pyproject.toml` の残りをインストールします)

```
# (otoge_rl) PS C:\dev\otoge_rl>
# ステップ 5.1: GPU 版 PyTorch (cu124+) を先にインストール
# (NVIDIA ドライバが 550.xx+ (CUDA 12.4+) である必要があります)
uv pip install torch --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# ステップ 5.2: プロジェクトの残りの依存関係 (SB3, Ruff, Pygame など) をインストール
uv pip install -e .[dev]
```

### B. 学習の実行 (`main.py`)

`uv run` は `src` フォルダを `PYTHONPATH` として自動的に認識します。`--output_path` は必須です。

**1. 基本的な学習 (JSON のデフォルト値を使用)**

```
# (otoge_rl) PS C:\dev\otoge_rl>
# V8.2 修正: 'python src/main.py' を使用
uv run python src/main.py --output_path "outputs/exp_001_baseline" --device auto
```

**2. パラメータのオーバーライド (アジャイル実験)** (JSON ファイルを変更せず、CLI で `gamma` と `n_envs` を変更して実験)

```
# (otoge_rl) PS C:\dev\otoge_rl>
uv run python src/main.py `
    --output_path "outputs/exp_002_gamma_095" `
    --total_timesteps 2000000 `
    --n_envs 8 `
    --gamma 0.95 `
    --device cuda
```

### C. 学習済みモデルの評価 (`evaluate.py`)

`--run_path` に `main.py` で使用した `output_path` を指定します。

```
# (otoge_rl) PS C:\dev\otoge_rl>
uv run python src/evaluate.py --run_path "outputs/exp_002_gamma_095" --device auto
```

(ビデオファイル (`.mp4`) が `outputs/exp_002_gamma_095/` フォルダ内に保存されます。)

## 5. コード品質 (CI/CD)

このプロジェクトは `ruff` を使用してコード品質を強制します。

### A. (ローカル) フォーマットの自動修正

(コードを保存する前に実行してください)

```
# (otoge_rl) PS C:\dev\otoge_rl>
uv run ruff format src/
```

### B. (ローカル) エラーのチェック

(Git に push する前に実行してください)

```
# (otoge_rl) PS C:\dev\otoge_rl>
uv run ruff check src/
```