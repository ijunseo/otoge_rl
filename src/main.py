#
# otoge_rl_project: src/main.py
#
# このファイルは、プロジェクトのメイン学習エントリーポイントです。
# V9.6 計画に基づき、'uv run python src/main.py' で実行されます。
#
# V9.6 の変更点:
# 1. (V9.5 修正) 'n_eval_episodes' が argparse に定義されていなかった
#    AttributeError を修正。
#
# V9.5 以前の変更点:
# 1. (V9.3) 評価ボトルネック解消のため、n_eval_envs のデフォルトを 4 に変更。
# 2. (V9.3) Monitor ラッパーを DummyVecEnv の *外側* に適用。
# 3. (V9.4) density のデフォルト値に関する KeyError を修正。
# 4. (V9.5) argparse の conflicting option string (ArgumentError) を修正。
#

import argparse
import json
import os
import logging
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# `src` が PYTHONPATH にあるため、'agent' と 'rhythm_game' を直接インポート
from agent import make_env, EvalWithAccuracy, EVAL_SEED_BASE

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback


def setup_logging():
    """
    [V8.5] INFO レベルのコンソールログを設定します。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_configs() -> tuple[dict, dict]:
    """
    [V5] agent と game の JSON 設定ファイルをロードします。

    Returns:
        tuple[dict, dict]: (ppo_cfg, game_cfg)
    """
    try:
        with open("src/agent/ppo_config.json", "r", encoding="utf-8") as f:
            ppo_cfg = json.load(f)
        with open("src/rhythm_game/config.json", "r", encoding="utf-8") as f:
            game_cfg = json.load(f)
        return ppo_cfg, game_cfg
    except FileNotFoundError as e:
        logging.error(f"設定ファイルが見つかりません: {e}")
        logging.error(
            "プロジェクトルートから 'uv run python src/main.py' を実行していることを確認してください。"
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"JSON 設定ファイルの解析に失敗しました: {e}")
        sys.exit(1)


def setup_device(device_arg: str) -> tuple[str, str]:
    """
    [V8.5] PyTorch が使用するデバイス (CUDA または CPU) を決定します。

    Args:
        device_arg (str): argparse からの '--device' 引数 ("auto", "cuda", "cpu")。

    Returns:
        tuple[str, str]: (PPO に渡すデバイス文字列, torch.cuda.available のブール文字列)
    """
    if device_arg == "auto":
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device = "cuda"
            else:
                device = "cpu"
                logging.warning("torch.cuda.is_available() が False を返しました。")
                logging.warning(
                    "インストールされた PyTorch が CUDA 12.4+ と互換性がない (CPU版？) か、NVIDIA ドライバに問題がある可能性があります。"
                )
        except Exception as e:
            logging.error(f"CUDA デバイスのチェック中にエラーが発生しました: {e}")
            device = "cpu"
            cuda_available = "False (Error)"
    else:
        device = device_arg
        cuda_available = (
            "N/A (指定)" if device == "cpu" else str(torch.cuda.is_available())
        )

    return device, f"{device} (torch.cuda.available: {cuda_available})"


def parse_arguments(ppo_cfg: dict, game_cfg: dict) -> argparse.Namespace:
    """
    [V9.6] argparse を設定し、JSON と CLI からの引数をマージします。
    AttributeError (V9.6) を修正済み。
    """
    parser = argparse.ArgumentParser(description="[V9.6] Otoge RL Agent 学習スクリプト")

    # --- 実行制御 ---
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="結果 (ログ、モデル、グラフ) を保存するフォルダパス。",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="総学習タイムステップ数。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="学習に使用するデバイス。",
    )

    # --- 環境設定 (game_cfg) ---
    parser.add_argument(
        "--n_envs", type=int, default=16, help="並列実行する学習環境の数。"
    )
    parser.add_argument(
        "--n_eval_envs",
        type=int,
        default=4,
        help="[V9.3] 並列実行する評価環境の数。(性能改善のため 1->4 に変更)",
    )

    # [V9.6] 修正: create_callbacks で使用する 'n_eval_episodes' が抜けていた (AttributeError 修正)
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=5,
        help="[V9.6] 各評価ステップで実行するエピソード数。(デフォルト: 5)",
    )

    parser.add_argument("--n_stack", type=int, default=4, help="フレームスタック数。")
    parser.add_argument(
        "--resized_shape_h", type=int, default=100, help="リサイズ後の観測の高さ。"
    )
    parser.add_argument(
        "--resized_shape_w", type=int, default=75, help="リサイズ後の観測の幅。"
    )

    # (V9.4) 修正: config.json の "default" キーへの依存を削除
    density_choices = list(game_cfg["density_settings"].keys())
    parser.add_argument(
        "--density",
        type=str,
        default="medium",  # ◀ (修正) 'default' キーの代わりに 'medium' を明記
        choices=density_choices,  # ◀ (修正) "options" サブキーを削除
        help=f"譜面の密度。 (選択肢: {density_choices}, デフォルト: 'medium')",
    )

    # --- PPO ハイパーパラメータ (ppo_cfg) ---
    # (V9.5) 修正: V9.4 適用時に発生した ArgumentError を解決 (重複定義を削除)
    parser.add_argument(
        "--learning_rate", type=float, default=ppo_cfg["learning_rate"], help="学習率。"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=ppo_cfg["n_steps"],
        help="各環境が1回の更新までに収集するステップ数。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=ppo_cfg["batch_size"],
        help="ミニバッチサイズ。",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=ppo_cfg["n_epochs"],
        help="各更新でのエポック数。",
    )
    parser.add_argument(
        "--gamma", type=float, default=ppo_cfg["gamma"], help="割引率 (Gamma)。"
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=ppo_cfg["gae_lambda"],
        help="GAE の Lambda。",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=ppo_cfg["clip_range"],
        help="PPO クリップ範囲。",
    )
    parser.add_argument(
        "--ent_coef", type=float, default=ppo_cfg["ent_coef"], help="エントロピー係数。"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=ppo_cfg["vf_coef"], help="価値関数係数。"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=ppo_cfg["max_grad_norm"],
        help="最大勾配ノルム。",
    )

    return parser.parse_args()


def save_parameters(
    args: argparse.Namespace, ppo_cfg: dict, game_cfg: dict, device_info: str
):
    """
    [V8.1] 実行に使用された全てのパラメータを hyperparameters.txt に保存します。
    """
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    params_path = os.path.join(output_path, "hyperparameters.txt")

    # CLI 引数を辞書に変換
    cli_arguments = vars(args).copy()

    # 全てのパラメータを統合
    all_params = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_info": device_info,
        "cli_arguments": cli_arguments,
        "base_ppo_config": ppo_cfg,
        "base_game_config": game_cfg,
    }

    try:
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(all_params, f, indent=4, ensure_ascii=False)
        logging.info(f"実行パラメータを {params_path} に保存しました。")
    except Exception as e:
        logging.error(f"パラメータの保存に失敗しました: {e}")


def setup_env(args: argparse.Namespace, game_cfg: dict) -> VecFrameStack:
    """
    [V9.3] 学習用の並列環境 (DummyVecEnv) を設定します。
    V8.6 の TypeError を修正済み。
    """
    logging.info(f"並列環境 (N_ENVS={args.n_envs}) を生成中...")

    # [V8.6] make_env に渡す引数を修正 (resized_shape をタプルに)
    resized_shape = (args.resized_shape_h, args.resized_shape_w)

    # [V8.6] list comprehension を使用して env_fns を生成
    env_fns = [
        make_env(
            game_config=game_cfg,
            density=args.density,
            resized_shape=resized_shape,
            seed_base=EVAL_SEED_BASE,
            rank=i,
        )
        for i in range(args.n_envs)
    ]

    # Windows での安定性を考慮し DummyVecEnv を使用
    env = DummyVecEnv(env_fns)

    # [V9.3] Monitor ラッパーは EvalCallback との互換性のため VecMonitor を使用 (setup_eval_env を参照)
    # ここでは学習の進行状況をログに出力するため、VecMonitor でラップします。
    log_path = os.path.join(args.output_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    env = VecMonitor(env, log_path)

    env = VecFrameStack(env, n_stack=args.n_stack, channels_order="last")
    return env


def setup_eval_env(args: argparse.Namespace, game_cfg: dict) -> VecFrameStack:
    """
    [V9.3] 評価用の並列環境を設定します。
    V9.3 で Monitor 警告を修正し、性能ボトルネックを改善 (n_eval_envs=4)。
    """
    logging.info(f"評価用環境 (N_EVAL_ENVS={args.n_eval_envs}) を生成中...")

    resized_shape = (args.resized_shape_h, args.resized_shape_w)

    eval_env_fns = [
        make_env(
            game_config=game_cfg,
            density=args.density,
            resized_shape=resized_shape,
            seed_base=EVAL_SEED_BASE + 1000,  # 学習環境とシードを分離
            rank=i,
        )
        for i in range(args.n_eval_envs)
    ]

    eval_env = DummyVecEnv(eval_env_fns)

    # [V9.3] 修正: EvalCallback の警告を避けるため、VecMonitor でラップします。
    # VecMonitor は VecEnv 版の Monitor ラッパーです。
    eval_env = VecMonitor(
        eval_env, os.path.join(args.output_path, "logs_eval")
    )  # 評価ログを別フォルダに

    eval_env = VecFrameStack(eval_env, n_stack=args.n_stack, channels_order="last")
    return eval_env


def create_model(
    env: VecFrameStack, args: argparse.Namespace, ppo_cfg: dict, device: str
) -> PPO:
    """
    [V8.1] PPO モデルを生成します。
    """
    # CLI 引数でオーバーライドされた PPO パラメータ
    ppo_params = {
        "policy": ppo_cfg["policy"],
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

    # ログ保存先
    tb_log_path = os.path.join(args.output_path, "logs")

    model = PPO(
        env=env,
        device=device,
        tensorboard_log=tb_log_path,
        verbose=1,  # [V8.1] コンソールに学習状況を出力
        **ppo_params,
    )
    logging.info(
        f"PPO モデル ({ppo_params['policy']}) をデバイス '{device}' 上に作成しました。"
    )
    logging.info(f"JSON 基本パラメータ: {ppo_cfg}")
    logging.info(f"最終適用パラメータ: {ppo_params}")
    return model, ppo_params


def create_callbacks(args: argparse.Namespace, eval_env: VecFrameStack) -> CallbackList:
    """
    [V9.6] 学習に使用するコールバックリストを生成します。
    AttributeError (V9.6) を修正済み。
    """
    log_path = os.path.join(args.output_path, "logs")
    model_path = os.path.join(args.output_path, "models")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # 1. 学習リワードロガー (Monitor.csv に依存するため、VecMonitor があれば不要)
    # train_rew_cb = TrainingRewardLogger() # VecMonitor が同等の機能を提供

    # 2. 評価コールバック (V9.3: n_eval_envs=4 で高速化)
    # eval_freq は n_steps を基準に計算
    eval_freq_steps = max(args.n_steps // args.n_envs, 1)

    eval_callback = EvalWithAccuracy(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq_steps,  # 1 ロールアウト (N_ENVS * N_STEPS) ごとに評価
        deterministic=True,
        render=False,
        n_eval_episodes=args.n_eval_episodes,  # [V9.6] 修正: args から正しく参照
    )

    # 3. チェックポイントコールバック (オプション)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            eval_freq_steps * 10, eval_freq_steps
        ),  # 10 ロールアウトごとにチェックポイント
        save_path=model_path,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # return CallbackList([train_rew_cb, eval_callback, checkpoint_callback])
    return CallbackList([eval_callback, checkpoint_callback])


def save_plots(args: argparse.Namespace):
    """
    [V8.1] 学習完了後、Monitor ログと評価ログからグラフを生成・保存します。
    """
    log_path = os.path.join(args.output_path, "logs")
    model_path = os.path.join(args.output_path, "models")
    plot_save_path = os.path.join(args.output_path, "training_plot.png")

    try:
        # 学習リワード (Monitor.csv)
        monitor_path = os.path.join(log_path, "monitor.csv")
        if not os.path.exists(monitor_path):
            # VecMonitor は時々 0.monitor.csv のように保存する
            monitor_path = os.path.join(log_path, "0.monitor.csv")
            if not os.path.exists(monitor_path):
                logging.warning(
                    f"Monitor.csv が見つかりません: {monitor_path}。学習リワードグラフはスキップされます。"
                )
                monitor_data = None

        if os.path.exists(monitor_path):
            # [V9.6] 修正: usecols=(0, 1, 2) -> usecols=(0, 2)
            # Monitor.csv の標準フォーマットは r (reward), l (length), t (time)
            # np.loadtxt は (0, 1) -> r, l (x軸がエピソード長になる)
            # (0, 2) -> r, t (x軸が経過時間になる)
            # SB3 Monitor は (r, l, t) ですが、V9.3 VecMonitor は (r, t) のみの場合があります。
            # usecols=(0, 2) -> reward, time
            #
            # [V9.6] 修正: (AttributeError により) model.learn() が実行されなかった場合、
            # 0.monitor.csv は "t,r" ヘッダーのみの空ファイルになります。
            # loadtxt は空ファイルで UserWarning と IndexError 를 발생시킵니다.
            try:
                monitor_data = np.loadtxt(
                    monitor_path, delimiter=",", skiprows=2, usecols=(0, 2)
                )  # reward, time
                if monitor_data.ndim == 1:  # データが1行しかない場合
                    monitor_data = monitor_data.reshape(1, -1)
                monitor_data_rewards = monitor_data[:, 0]
                monitor_data_timesteps = monitor_data[:, 1]  # time (経過秒数)
            except (IOError, IndexError, UserWarning) as e:
                logging.warning(
                    f"Monitor.csv の読み込みに失敗しました ({e})。学習リワードグラフはスキップされます。"
                )
                monitor_data = None

        # 評価精度 (evaluations.npz)
        eval_path = os.path.join(model_path, "evaluations.npz")
        if not os.path.exists(eval_path):
            logging.warning(
                f"evaluations.npz が見つかりません: {eval_path}。評価精度グラフはスキップされます。"
            )
            eval_data = None
        else:
            eval_data = np.load(eval_path)

        # グラフ生成
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"学習結果 (V9.6) - {args.output_path}")

        # サブプロット 1: 学習リワード
        if "monitor_data_rewards" in locals() and monitor_data_rewards is not None:
            # [V9.6] 修正: x軸を timesteps (経過秒数) でプロット
            axs[0].plot(
                monitor_data_timesteps,
                monitor_data_rewards,
                alpha=0.5,
                label="Reward (Raw)",
            )
            # トレンドライン (移動平均)
            if len(monitor_data_rewards) > 100:
                moving_avg = np.convolve(
                    monitor_data_rewards, np.ones(100) / 100, mode="valid"
                )
                # 이동 평균의 x축도 time 에 맞춰 조정
                avg_timesteps = monitor_data_timesteps[
                    len(monitor_data_timesteps) - len(moving_avg) :
                ]
                axs[0].plot(
                    avg_timesteps, moving_avg, color="r", label="Moving Avg (100 ep)"
                )
            axs[0].set_title("学習: エピソードリワード (Training Reward)")
            axs[0].set_xlabel("Time (seconds)")
            axs[0].set_ylabel("Reward")
            axs[0].legend()
        else:
            axs[0].set_title("学習リワード (データなし)")

        # サブプロット 2: 評価精度
        if eval_data:
            axs[1].plot(
                eval_data["timesteps"], eval_data["results"][:, 0]
            )  # Mean Reward
            axs[1].set_title("評価: 平均リワード (Evaluation Mean Reward)")
            axs[1].set_xlabel("Timesteps")
            axs[1].set_ylabel("Mean Reward")

            # (オプション) 評価精度 (accuracy_pct) があれば右 Y軸にプロット
            if "accuracy_pct" in eval_data:
                ax2 = axs[1].twinx()
                ax2.plot(eval_data["timesteps"], eval_data["accuracy_pct"], "g-")
                ax2.set_ylabel("Accuracy (%)", color="g")
        else:
            axs[1].set_title("評価リワード (データなし)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_save_path)
        logging.info(f"学習結果グラフを {plot_save_path} に保存しました。")

    except Exception as e:
        logging.error(f"グラフの保存中にエラーが発生しました: {e}")
        import traceback

        logging.error(
            traceback.format_exc()
        )  # V9.6: グラフエラーの詳細なトレースバック


def main():
    """
    [V9.6] メイン学習実行関数。
    """
    # [V9.6] 修正: try...finally ブロックの適用範囲を拡大
    # argparse や env 생성 실패 시에도 finally が呼ばれるように

    args = None
    ppo_cfg = {}
    game_cfg = {}
    env = None
    eval_env = None
    model = None

    try:
        # 1. セットアップ
        setup_logging()
        logging.info(
            "--- [V9.6] Otoge RL Agent 学習開始 ---"
        )  # バージョンを V9.6 に更新
        ppo_cfg, game_cfg = load_configs()

        # 2. 引数解析 (V9.6: AttributeError 修正済み)
        args = parse_arguments(ppo_cfg, game_cfg)

        # 3. デバイス決定 (V8.5)
        device, device_info = setup_device(args.device)
        logging.info(f"使用デバイス: {device_info}")

        # 4. パラメータ保存 (V8.1)
        save_parameters(args, ppo_cfg, game_cfg, device_info)

        # 5. 環境生成 (V9.3: TypeError, Monitor 警告修正済み)
        env = setup_env(args, game_cfg)
        eval_env = setup_eval_env(args, game_cfg)

        # 6. モデル生成 (V8.1)
        model, ppo_params = create_model(env, args, ppo_cfg, device)

        # 7. コールバック生成 (V9.6: AttributeError 修正済み)
        callbacks = create_callbacks(args, eval_env)

        # 8. 学習開始 (V8.6: progress_bar=True)
        logging.info(f"総タイムステップ {args.total_timesteps} で学習を開始します...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=f"{args.density}_{ppo_params['policy']}",  # ログファイル名を指定
            progress_bar=True,  # [V8.6] 進行状況バーを強制的に表示
        )

    except ImportError as e:
        logging.error(f"必要なパッケージがインストールされていません: {e}")
        logging.error(
            "V9.2 計画に基づき、'pyproject.toml' に 'tqdm' と 'rich' を追加し、"
        )
        logging.error("'uv pip install -e .[dev]' を実行してください。")

    except argparse.ArgumentError as e:
        logging.error(f"CLI 引数の定義に競合があります: {e}")
        logging.error("V9.5 計画: main.py の parse_arguments 関数を確認してください。")

    except AttributeError as e:
        # [V9.6] 捕捉: argparse で定義漏れがあった場合
        logging.error(f"argparse の定義漏れによる AttributeError が発生しました: {e}")
        logging.error("V9.6 計画: main.py の parse_arguments 関数を確認してください。")

    except Exception as e:
        logging.error(f"学習中に予期せぬエラーが発生しました: {e}")
        import traceback

        logging.error(traceback.format_exc())  # 完全なトレースバックを出力

    finally:
        # 9. クリーンアップとグラフ保存
        logging.info("学習完了。環境をクリーンアップします。")
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()

        # 10. 最終モデルの保存
        if model is not None:
            final_model_path = os.path.join(
                args.output_path, "models", "final_model.zip"
            )
            model.save(final_model_path)
            logging.info(f"最終モデルを {final_model_path} に保存しました。")

        # 11. グラフ保存 (V8.1)
        if args is not None:
            save_plots(args)

        logging.info("--- [V9.6] 学習プロセス終了 ---")


if __name__ == "__main__":
    main()
