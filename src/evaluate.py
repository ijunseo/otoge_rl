#
# otoge_rl_project: src/evaluate.py
#
# (V9.3)
# このファイルは、学習済みのモデル (`best_model.zip`) を評価し、
# プレイ動画 (`.mp4`) として録画するスクリプトです。
# V9.1 計画に基づき、'uv run python src/evaluate.py' で実行されます。
#
# 機能:
# 1. `--run_path` (main.py の --output_path) を必須引数として受け取ります。
# 2. `run_path` から `hyperparameters.txt` を読み込み、環境設定を復元します。
# 3. `run_path/models/best_model.zip` をロードします。
# 4. `rgb_array` モードで環境を実行し、プレイ動画を `run_path` に保存します。
# 5. (V9.3) `main.py` (V9.3) と同様に、`Monitor` を `DummyVecEnv` の外側に適用します。
#

import argparse
import json
import os
import logging
import sys
import imageio
import numpy as np

# `src` が PYTHONPATH にあるため、'agent' と 'rhythm_game' を直接インポート
from agent.utils import make_env, EVAL_SEED_BASE
from rhythm_game import RhythmGameEnv

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import ResizeObservation, ReshapeObservation
import torch

def setup_logging():
    """コンソールログ (INFO レベル) の基本設定を行います。"""
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("--- [V9.3] Otoge RL Agent 評価開始 ---")

def load_configs_from_run(run_path: str) -> tuple[dict, dict]:
    """
    指定された実行パスから設定ファイル (hyperparameters.txt と game_config.json) をロードします。

    Args:
        run_path (str): `main.py` の --output_path で指定されたフォルダパス。

    Returns:
        tuple[dict, dict]: (run_args, game_config)
    """
    params_path = os.path.join(run_path, "hyperparameters.txt")
    game_config_path = "src/rhythm_game/config.json" # ゲーム設定は常に最新をロード

    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            run_params = json.load(f)
            # `main.py` (V8.6) は 'cli_arguments' キー以下に引数を保存します
            run_args = run_params.get("cli_arguments", {})
            if not run_args:
                 logging.error(f"'{params_path}' に 'cli_arguments' キーが見つかりません。ファイルが破損しているか、古いバージョンです。")
                 sys.exit(1)
            
        with open(game_config_path, 'r', encoding='utf-8') as f:
            game_config = json.load(f)
            
        return run_args, game_config
        
    except FileNotFoundError as e:
        logging.error(f"設定ファイルが見つかりません: {e}")
        logging.error(f"指定されたパス '{run_path}' が正しいか確認してください。")
        sys.exit(1)
    except Exception as e:
        logging.error(f"設定ファイルの読み込み中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)

def setup_evaluation_vec_env(run_args: dict, game_cfg: dict) -> VecFrameStack:
    """
    (V9.3) モデル評価（動画録画）用のベクトル環境をセットアップします。
    `main.py` (V9.3) の `setup_eval_env` と同じラッパー構造を使用します。
    """
    logging.info("評価用（録画用）環境を生成中...")
    
    resized_h = run_args.get("resized_shape_h", 100)
    resized_w = run_args.get("resized_shape_w", 75)
    n_stack = run_args.get("n_stack", 4)
    density = run_args.get("density", "medium")

    env_kwargs = dict(
        game_config=game_cfg,
        density=density,
        resized_shape=(resized_h, resized_w)
    )
    
    # (V9.3) make_env を使用して単一の thunk を作成
    eval_env_fn = make_env(rank=9999, seed_base=EVAL_SEED_BASE, **env_kwargs)
    
    # (V9.3) DummyVecEnv でラップ
    env = DummyVecEnv([eval_env_fn])
    
    # (V9.3) Monitor を *外側* にラップ (必須ではないが、一貫性のために)
    # (Monitor は .csv ログを必要としないため、filename=None)
    env = Monitor(env) 
    
    # (V9.3) VecFrameStack を適用
    env = VecFrameStack(env, n_stack=n_stack, channels_order='last')
    
    return env

def main():
    """
    メイン評価実行関数。
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="[V9.3] 学習済みモデル評価＆録画スクリプト")
    parser.add_argument("--run_path", type=str, required=True, help="評価するモデルと設定が含まれる実行フォルダパス (main.py の output_path)。")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="推論に使用するデバイス。")
    args = parser.parse_args()

    # 1. 設定とパスのロード
    try:
        run_args, game_cfg = load_configs_from_run(args.run_path)
    except Exception:
        # load_configs_from_run が既にエラーをロギングしています
        return

    model_path = os.path.join(args.run_path, "models", "best_model.zip")
    video_path = os.path.join(args.run_path, f"evaluation_video_{run_args.get('density', 'default')}.mp4")

    if not os.path.exists(model_path):
        logging.error(f"モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    # 2. デバイス設定
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logging.info(f"使用デバイス: {device}")

    # 3. (V9.3) 評価用環境の生成
    try:
        vec_env = setup_evaluation_vec_env(run_args, game_cfg)
        # `rgb_array` モードでレンダリングするための「素」の環境も作成
        # (V9.3) setup_evaluation_vec_env が VecEnv を返すため、
        # `render()` のためには別途 'rgb_array' モードの env が必要
        
        render_env = RhythmGameEnv(
            game_config=game_cfg,
            density=run_args.get("density", "medium"),
            render_mode='rgb_array' # 録画用に 'rgb_array' モードを使用
        )
        
    except Exception as e:
        logging.error(f"評価環境の生成に失敗しました: {e}", exc_info=True)
        sys.exit(1)

    # 4. モデルのロード
    try:
        model = PPO.load(model_path, device=device)
        logging.info(f"モデルを {model_path} から正常にロードしました。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}", exc_info=True)
        sys.exit(1)

    # 5. 評価（録画）実行
    frames = []
    try:
        logging.info("評価（録画）を開始します...")
        
        # (V9.3) render_env と vec_env の状態を同期させる
        # 1. vec_env (モデル入力用) をリセット
        obs = vec_env.reset() 
        # 2. render_env (録画用) も同じシードでリセット
        #    (注: seed_base+rank (9999) は vec_env の _init で既に設定されています)
        #    (render_env も同じ状態を共有するために reset が必要)
        #    (V9.3) setup_evaluation_vec_env が内部でシード (EVAL_SEED_BASE + 9999) を設定します。
        #    render_env も同じシードでリセットする必要があります。
        render_env.reset(seed=(EVAL_SEED_BASE + 9999))
        
        
        done = False
        total_game_score = 0
        max_score = 0

        while not done:
            # 1. モデルが VecEnv (スタック/リサイズ済み) の観測を見て行動を決定
            action, _ = model.predict(obs, deterministic=True)
            
            # 2. VecEnv (モデル入力用) を 1 ステップ進める
            obs, reward, done, info = vec_env.step(action)
            
            # 3. render_env (録画用) も *同じ行動* で 1 ステップ進める
            #    (V9.3) render_env はベクトル化されていないため、action[0] を使用
            render_obs, render_reward, render_done, _, render_info = render_env.step(action[0])
            
            # 4. render_env (素の環境) から 'rgb_array' フレームを取得
            frame = render_env.render()
            frames.append(frame)

            # (V9.3) render_env の情報を使用 (VecEnv/Monitor は done=True 時にしか info を更新しないため)
            if "game_score" in render_info:
                total_game_score = render_info["game_score"]
            if "max_score" in render_info:
                max_score = render_info["max_score"]

            # done は VecEnv (done[0]) を基準にする
            done = done[0]

        score_percentage = (total_game_score / max_score) * 100 if max_score > 0 else 0.0
        logging.info("評価完了。")
        logging.info(f"最終ゲームスコア: {total_game_score:.0f} / {max_score:.0f} ({score_percentage:.2f}%)")

    except Exception as e:
        logging.error(f"評価実行中にエラーが発生しました: {e}", exc_info=True)
    finally:
        vec_env.close()
        render_env.close()

    # 6. 動画保存
    if frames:
        try:
            logging.info(f"'{video_path}' に動画を保存中 (フレーム数: {len(frames)})...")
            imageio.mimsave(video_path, frames, fps=RhythmGameEnv.metadata["render_fps"])
            logging.info("動画の保存が完了しました。")
        except Exception as e:
            logging.error(f"動画の保存に失敗しました: {e}", exc_info=True)
    else:
        logging.warning("録画されたフレームがありません。動画は保存されません。")

if __name__ == "__main__":
    main()

