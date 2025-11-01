#
# otoge_rl_project: src/main.py
#
# このファイルは、プロジェクトのメイン学習エントリーポイントです。
# V8 計画に基づき、'uv run python -m main' で実行されます。
#
# 機能:
# 1. `agent/ppo_config.json` と `rhythm_game/config.json` から基本設定をロードします。
# 2. `argparse` を使用して、CLI からこれらの設定をオーバーライドできるようにします。
# 3. `--output_path` で指定されたフォルダに、ログ、モデル、および使用された
#    `hyperparameters.txt` を自動的に保存します。
# 4. `PPO(verbose=1)` を設定し、学習の進行状況をコンソールにリアルタイムで出力します。
#

import argparse
import json
import os
import logging
import sys
import matplotlib.pyplot as plt
import torch

# `src` が PYTHONPATH にあるため、'agent' と 'rhythm_game' を直接インポート
try:
    from agent import make_env, TrainingRewardLogger, EvalWithAccuracy, EVAL_SEED_BASE
    from rhythm_game import RhythmGameEnv
except ImportError:
    logging.error("エラー: 'agent' または 'rhythm_game' モジュールが見つかりません。")
    logging.error("'uv run' を使用して、'src' フォルダが PYTHONPATH に追加されていることを確認してください。")
    sys.exit(1)

# Stable Baselines3 と Gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.vec_env import VecFrameStack # (main() で直接インポート)
from stable_baselines3.common.callbacks import CallbackList

def setup_logging():
    """
    コンソールログの基本設定を行います。
    SB3 の情報レベルのログ (verbose=1) が表示されるようにします。
    """
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger("stable_baselines3").setLevel(logging.INFO)

def load_configs() -> tuple[dict, dict]:
    """
    設定 JSON ファイルをロードします。

    Returns:
        tuple[dict, dict]: (ppo_config, game_config)
    """
    try:
        # main.py は src/ にあるため、相対パスで config をロード
        with open("src/agent/ppo_config.json", 'r', encoding='utf-8') as f:
            ppo_config = json.load(f)
        with open("src/rhythm_game/config.json", 'r', encoding='utf-8') as f:
            game_config = json.load(f)
        return ppo_config, game_config
    except FileNotFoundError as e:
        logging.error(f"設定ファイルが見つかりません: {e}")
        logging.error("パスが 'src/agent/...' または 'src/rhythm_game/...' であることを確認してください。")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"JSON の解析に失敗しました: {e}")
        sys.exit(1)

def create_parser(ppo_cfg: dict, game_cfg: dict) -> argparse.ArgumentParser:
    """
    CLI 引数のパーサーを作成します。
    JSON の値をデフォルト値として使用します。

    Args:
        ppo_cfg (dict): PPO のデフォルト設定。
        game_cfg (dict): ゲームのデフォルト設定。

    Returns:
        argparse.ArgumentParser: 設定済みのパーサー。
    """
    parser = argparse.ArgumentParser(description="リズムゲーム RL エージェント学習スクリプト")

    # --- 実行パス ---
    parser.add_argument("--output_path", type=str, required=True, help="結果 (ログ、モデル) を保存する一意のフォルダパス。")

    # --- 学習全体設定 (Jupyter Cell 5 より) ---
    parser.add_argument("--algorithm", type=str, default="PPO", help="使用するアルゴリズム (現在は PPO のみサポート)。")
    parser.add_argument("--total_timesteps", type=int, default=3_500_000, help="総学習タイムステップ数。")
    parser.add_argument("--n_envs", type=int, default=16, help="並列実行する環境の数。")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="学習に使用するデバイス。")

    # --- 環境パラメータ (Jupyter Cell 5 / game_config.json) ---
    parser.add_argument("--density", type=str, default="high", 
                        choices=game_cfg.get("density_settings", {"low":0, "medium":0, "high":0}).keys(), 
                        help="ノーツの密度。")
    parser.add_argument("--n_stack", type=int, default=4, help="フレームスタックの数。")
    parser.add_argument("--resized_shape_h", type=int, default=100, help="観測のリサイズ後の高さ (Jupyter Cell 5 の値)。")
    parser.add_argument("--resized_shape_w", type=int, default=75, help="観測のリサイズ後の幅 (Jupyter Cell 5 の値)。")

    # --- PPO ハイパーパラメータ (ppo_config.json / Jupyter Cell 5) ---
    # .get() を使用して、JSON ファイルにキーが存在しない場合でもフォールバック値を提供
    parser.add_argument("--policy", type=str, default=ppo_cfg.get("policy", "CnnPolicy"), help="PPO ポリシー。")
    parser.add_argument("--learning_rate", type=float, default=ppo_cfg.get("learning_rate", 1e-4), help="学習率。")
    parser.add_argument("--n_steps", type=int, default=ppo_cfg.get("n_steps", 2048), help="1回の更新までに各環境で収集するステップ数。")
    parser.add_argument("--batch_size", type=int, default=ppo_cfg.get("batch_size", 4096), help="ミニバッチサイズ。")
    parser.add_argument("--n_epochs", type=int, default=ppo_cfg.get("n_epochs", 4), help="各更新でのエポック数。")
    parser.add_argument("--gamma", type=float, default=ppo_cfg.get("gamma", 0.99), help="割引率 (gamma)。")
    parser.add_argument("--gae_lambda", type=float, default=ppo_cfg.get("gae_lambda", 0.95), help="GAE のラムダ。 (Jupyter Cell 5 の値)")
    parser.add_argument("--clip_range", type=float, default=ppo_cfg.get("clip_range", 0.2), help="PPO クリップ範囲。")
    parser.add_argument("--ent_coef", type=float, default=ppo_cfg.get("ent_coef", 0.01), help="エントロピー係数。")
    parser.add_argument("--vf_coef", type=float, default=ppo_cfg.get("vf_coef", 0.5), help="価値関数係数。")
    parser.add_argument("--max_grad_norm", type=float, default=ppo_cfg.get("max_grad_norm", 0.5), help="最大勾配ノルム。")

    # --- 評価パラメータ (Jupyter Cell 5) ---
    parser.add_argument("--eval_freq_multiplier", type=int, default=1, 
                        help="評価頻度の乗数。eval_freq = multiplier * n_steps。 (Cell 5 の EVAL_FREQ = N_STEPS を再現)")
    parser.add_argument("--n_eval_episodes", type=int, default=5, help="各評価で実行するエピソード数。")

    return parser

def save_parameters(args: argparse.Namespace, ppo_cfg: dict, game_cfg: dict):
    """
    使用されたすべてのパラメータを `hyperparameters.txt` として保存します。
    (V7 リクエスト事項: CLI引数をJSON形式で保存)

    Args:
        args (argparse.Namespace): CLI からパースされた引数。
        ppo_cfg (dict): ロードされた PPO 設定。
        game_cfg (dict): ロードされたゲーム設定。
    """
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    # CLI引数を辞書に変換
    cli_args = vars(args)
    
    # 保存するすべての設定を結合
    all_params = {
        "command_line_args (final)": cli_args,
        "ppo_config_defaults (loaded)": ppo_cfg,
        "game_config_defaults (loaded)": game_cfg
    }
    
    params_path = os.path.join(output_dir, "hyperparameters.txt")
    try:
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(all_params, f, indent=4, ensure_ascii=False)
        logging.info(f"パラメータを {params_path} に保存しました。")
    except IOError as e:
        logging.error(f"パラメータの保存に失敗しました: {e}")

def plot_results(train_cb: TrainingRewardLogger, eval_cb: EvalWithAccuracy, total_timesteps: int, save_path: str):
    """
    学習結果 (報酬と正解率) のグラフを保存します。
    (Jupyter Notebook Cell 5 のプロットロジック)

    Args:
        train_cb (TrainingRewardLogger): 学習報酬コールバック。
        eval_cb (EvalWithAccuracy): 評価コールバック。
        total_timesteps (int): 総学習ステップ数 (X軸の最大値)。
        save_path (str): プロット画像の保存パス (.png)。
    """
    try:
        plt.figure(figsize=(12, 5))
        
        # 1. 学習報酬プロット
        plt.subplot(1, 2, 1)
        if train_cb.t and train_cb.rew:
            plt.plot(train_cb.t, train_cb.rew)
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward (RL Score)")
        plt.title("Training: Episode Reward")

        # 2. 評価正解率プロット
        plt.subplot(1, 2, 2)
        if eval_cb.eval_steps and eval_cb.eval_acc:
            plt.plot(eval_cb.eval_steps, eval_cb.eval_acc, marker="o")
        plt.xlabel("Timesteps")
        plt.ylabel("Accuracy (%)")
        plt.title("Evaluation: Accuracy")
        plt.xlim(0, total_timesteps) # X軸を総ステップ数に固定

        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(f"学習結果グラフを {save_path} に保存しました。")
    except Exception as e:
        logging.warning(f"グラフの保存に失敗しました: {e}")
    finally:
        plt.close()

def main():
    """
    メイン学習実行関数。
    """
    # 1. セットアップ
    setup_logging()
    ppo_cfg, game_cfg = load_configs()
    parser = create_parser(ppo_cfg, game_cfg)
    args = parser.parse_args()

    # 2. パラメータの保存 (実験の再現性のために最初に行う)
    save_parameters(args, ppo_cfg, game_cfg)

    # 3. パス設定
    output_dir = args.output_path
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    plot_path = os.path.join(output_dir, "training_plot.png")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 4. デバイス設定
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True # GPU 最適化 (Cell 5)
    
    logging.info(f"使用デバイス: {device}")

    # 5. 環境の生成 (並列化)
    # (Jupyter Cell 5 で VecFrameStack が SubprocVecEnv の後に適用されていたため、
    # VecFrameStack は main.py でインポートして適用します)
    from stable_baselines3.common.vec_env import VecFrameStack
    
    resized_shape = (args.resized_shape_h, args.resized_shape_w)
    
    # `make_env` に設定を渡す
    env_thunks = [
        make_env(
            game_config=game_cfg,
            density=args.density,
            resized_shape=resized_shape,
            rank=i,
            seed_base=42
        ) for i in range(args.n_envs)
    ]
    env = SubprocVecEnv(env_thunks)
    env = VecFrameStack(env, n_stack=args.n_stack, channels_order='last')

    # 評価用環境の生成
    eval_env_thunks = [
        make_env(
            game_config=game_cfg,
            density=args.density,
            resized_shape=resized_shape,
            rank=1000 + i, # 評価用シードオフセット
            seed_base=EVAL_SEED_BASE
        ) for i in range(max(2, args.n_envs // 4)) # 最小2, N_ENVS/4 の並列評価
    ]
    eval_env = SubprocVecEnv(eval_env_thunks)
    eval_env = VecFrameStack(eval_env, n_stack=args.n_stack, channels_order='last')

    # 6. コールバックの定義
    train_rew_cb = TrainingRewardLogger()
    
    # Cell 5 のロジック (EVAL_FREQ = N_STEPS) を再現
    eval_freq_steps = args.n_steps * args.eval_freq_multiplier
    
    eval_callback = EvalWithAccuracy(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq_steps,
        deterministic=True,
        render=False,
        n_eval_episodes=args.n_eval_episodes,
    )
    callbacks = CallbackList([train_rew_cb, eval_callback])

    # 7. PPO モデルのパラメータ設定 (CLI オーバーライド適用)
    ppo_params = {
        "policy": args.policy,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
    }

    model = PPO(
        env=env,
        device=device,
        verbose=1, # コンソールに学習進捗を出力 (リクエスト事項)
        tensorboard_log=log_dir,
        **ppo_params
    )

    # 8. 学習開始
    logging.info(f"--- {args.algorithm} モデルの学習開始 (総ステップ: {args.total_timesteps}) ---")
    logging.info(f"使用パラメータ: {json.dumps(ppo_params, indent=2)}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=f"{args.algorithm}_{args.density}",
            progress_bar=True # コンソールに進捗バーを表示
        )
    except KeyboardInterrupt:
        logging.info("学習が中断されました。現在のモデルを保存します...")
    finally:
        # 学習が完了または中断された場合でも、現在のモデルを保存
        model.save(os.path.join(model_dir, "final_model"))
        env.close()
        eval_env.close()
        logging.info("--- 学習完了 (環境クローズ済み) ---")

    # 9. 学習結果のプロット
    plot_results(train_rew_cb, eval_callback, args.total_timesteps, plot_path)

# 'uv run python -m main' で実行されるためのエントリーポイント
if __name__ == "__main__":
    main()
