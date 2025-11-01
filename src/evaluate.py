#
# otoge_rl_project: src/evaluate.py
#
# このファイルは、学習済みのモデル (`best_model.zip`) を評価し、
# プレイ動画 (`.mp4`) として録画するスクリプトです。
# V8 計画に基づき、'uv run python -m evaluate' で実行されます。
#
# 機能:
# 1. `--run_path` (main.py の --output_path) を必須引数として受け取ります。
# 2. `run_path` から `hyperparameters.txt` を読み込み、環境設定を復元します。
# 3. `run_path/models/best_model.zip` をロードします。
# 4. `rgb_array` モードで環境を実行し、プレイ動画を `run_path` に保存します。
#

import argparse
import json
import os
import logging
import sys
import imageio
import numpy as np

# `src` が PYTHONPATH にあるため、'agent' と 'rhythm_game' を直接インポート
try:
    from agent import EVAL_SEED_BASE
    from rhythm_game import RhythmGameEnv
except ImportError:
    logging.error("エラー: 'agent' または 'rhythm_game' モジュールが見つかりません。")
    sys.exit(1)

# Stable Baselines3 と Gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import ResizeObservation, ReshapeObservation
import torch

def setup_logging():
    """コンソールログの基本設定を行います。"""
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
            # hyperparameters.txt には CLI 引数が 'cli_arguments' キーで保存されている
            run_args = run_params.get("cli_arguments", {})
            if not run_args:
                logging.warning(f"{params_path} に 'cli_arguments' が見つかりません。")
                
        with open(game_config_path, 'r', encoding='utf-8') as f:
            game_config = json.load(f)
            
        return run_args, game_config
        
    except FileNotFoundError as e:
        logging.error(f"設定ファイルが見つかりません: {e}")
        logging.error(f"指定されたパス '{run_path}' が正しいか確認してください。")
        sys.exit(1)

def create_viz_env(game_cfg: dict, run_args: dict) -> VecFrameStack:
    """
    動画録画 (`rgb_array` モード) 用の環境を生成・ラップします。

    Args:
        game_cfg (dict): ゲーム設定。
        run_args (dict): 実行時の CLI 引数。

    Returns:
        VecFrameStack: ラップされた単一の VecEnv。
    """
    from stable_baselines3.common.vec_env import VecFrameStack
    
    # CLI 引数から環境パラメータを復元 (main.py のデフォルト値と一致させる)
    density = run_args.get("density", "high")
    resized_h = run_args.get("resized_shape_h", 100)
    resized_w = run_args.get("resized_shape_w", 75)
    n_stack = run_args.get("n_stack", 4)
    resized_shape = (resized_h, resized_w)

    # 録画用に 'rgb_array' モードで単一の環境を生成
    def make_single_env():
        env = RhythmGameEnv(
            game_config=game_cfg,
            density=density,
            render_mode='rgb_array' # 録画モード
        )
        # main.py と同じラッパーを適用
        env = ResizeObservation(env, shape=resized_shape)
        env = ReshapeObservation(env, shape=(resized_shape[0], resized_shape[1], 1))
        env.reset(seed=EVAL_SEED_BASE + 100) # 評価用シード
        return env

    # DummyVecEnv を使用 (SubprocVecEnv は不要)
    viz_env_wrapped = DummyVecEnv([make_single_env])
    viz_vec_env = VecFrameStack(viz_env_wrapped, n_stack=n_stack, channels_order='last')
    return viz_vec_env

def main():
    """
    メイン評価実行関数。
    """
    # 1. セットアップ
    setup_logging()
    parser = argparse.ArgumentParser(description="学習済みモデル評価 (動画録画) スクリプト")
    parser.add_argument("--run_path", type=str, required=True, 
                        help="評価するモデルと設定が含まれる実行フォルダパス (main.py の --output_path)。")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], 
                        help="推論に使用するデバイス。")
    parser.add_argument("--fps", type=int, default=30, help="録画する動画の FPS。")
    args = parser.parse_args()

    # 2. 設定とパスのロード
    logging.info(f"'{args.run_path}' から設定をロードしています...")
    run_args, game_cfg = load_configs_from_run(args.run_path)

    model_path = os.path.join(args.run_path, "models", "best_model.zip")
    video_path = os.path.join(args.run_path, f"evaluation_video_{run_args.get('density', 'default')}.mp4")

    if not os.path.exists(model_path):
        logging.error(f"モデルファイルが見つかりません: {model_path}")
        sys.exit(1)

    # 3. デバイス設定
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"使用デバイス: {device}")

    # 4. 録画用環境の生成 (Jupyter Cell 5 のロジック)
    viz_vec_env = create_viz_env(game_cfg, run_args)

    # 5. モデルのロード
    try:
        loaded_model = PPO.load(model_path, device=device)
        logging.info(f"最高性能モデルの読み込み成功: {model_path}")
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}")
        viz_vec_env.close()
        sys.exit(1)

    # 6. 動画録画の実行
    frames = []
    total_game_score = 0
    max_score = 0
    
    try:
        obs = viz_vec_env.reset()
        dones = np.array([False])
        
        logging.info("--- 動画録画開始 ---")
        while not np.all(dones):
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = viz_vec_env.step(action)
            
            # (H, W, 3) 形式のフレームを取得
            frame = viz_vec_env.envs[0].render() 
            if frame is not None:
                frames.append(frame)

            if np.all(dones):
                final_info = infos[0] if isinstance(infos, (list, tuple)) else infos
                total_game_score = final_info.get('game_score', 0)
                max_score = final_info.get('max_score', 0)

    except KeyboardInterrupt:
        logging.warning("録画が中断されました。")
    finally:
        viz_vec_env.close()
        logging.info("--- 動画録画終了 (環境クローズ済み) ---")

    # 7. 動画ファイルとして保存
    if frames:
        score_percentage = (total_game_score / max_score) * 100 if max_score > 0 else 0.0
        logging.info(f"最高性能エージェントの最終スコア: {total_game_score:.0f} / {max_score:.0f} ({score_percentage:.2f}%)")
        
        logging.info(f"プレイ動画 ({len(frames)} フレーム) を '{video_path}' (FPS={args.fps}) として保存します...")
        try:
            imageio.mimsave(video_path, [np.array(f) for f in frames], fps=args.fps)
            logging.info("保存完了。")
        except Exception as e:
            logging.error(f"動画の保存に失敗しました: {e}")
            logging.error("imageio-ffmpeg がインストールされているか確認してください (`uv pip install imageio-ffmpeg`)")
    else:
        logging.warning("録画されたフレームがありません。動画は保存されません。")

# 'uv run python -m evaluate' で実行されるためのエントリーポイント
if __name__ == "__main__":
    main()
