#
# otoge_rl_project: src/play_random.py
#
# (V9.3)
# このファイルは、Jupyter Notebook (Cell 4) のランダムエージェントテストを
# Python スクリプトとして実行します。
# 'uv run python src/play_random.py' で実行されます。
#
# 機能:
# 1. `rhythm_game/config.json` をロードします。
# 2. `RhythmGameEnv` を 'human' モードで直接インスタンス化します。
# 3. 環境が終了するまでランダムなアクションを実行し、Pygame ウィンドウに表示します。
# 4. (V9.3) `make_env` を使用せず、`RhythmGameEnv` を直接呼び出します。
#

import gymnasium as gym
import logging
import json
import sys
import time

# `src` が PYTHONPATH にあるため、'rhythm_game' を直接インポート
from rhythm_game import RhythmGameEnv

# (V9.3) utils.py (make_env) は RL 学習/評価専用のため、ここでは使用しない

def setup_logging():
    """コンソールログ (INFO レベル) の基本設定を行います。"""
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("--- [V9.3] Otoge RL Agent ランダムテスト開始 ---")

def load_game_config() -> dict:
    """
    rhythm_game の JSON 設定ファイルをロードします。
    """
    try:
        with open("src/rhythm_game/config.json", 'r', encoding='utf-8') as f:
            game_config = json.load(f)
        return game_config
    except FileNotFoundError as e:
        logging.error(f"設定ファイル (src/rhythm_game/config.json) が見つかりません: {e}")
        sys.exit(1)

def main():
    """
    メイン実行関数。
    """
    setup_logging()
    game_cfg = load_game_config()

    # (V9.3) render_mode='human' で環境を直接生成
    try:
        env = RhythmGameEnv(
            game_config=game_cfg,
            density=game_cfg.get("density_settings", {}).get("default", "medium"),
            render_mode='human'
        )
        logging.info("'human' モードで環境を正常に作成しました。Pygame ウィンドウを確認してください。")
    except Exception as e:
        logging.error(f"環境の作成に失敗しました: {e}", exc_info=True)
        return

    try:
        # (V8.6) Jupyter Notebook (Cell 4) と同様にシード 42 でリセット
        observation, info = env.reset(seed=42)
        
        done = False
        truncated = False
        frame_count = 0

        while not done and not truncated:
            # ランダムな行動を選択 (0..4)
            action = env.action_space.sample() 
            
            observation, reward, done, truncated, info = env.step(action)
            frame_count += 1
            
            score = info.get('game_score', 0.0)

            # (V8.6) 報酬が発生した時（またはミス時）のみログを出力
            if reward != 0:
                logging.info(f"フレーム: {frame_count}, 行動: {action}, 報酬: {reward:.1f}, 現在の合計スコア: {score:.1f}")
            
            # (V8.6) render_mode='human' の場合、env.step() が描画を処理します

    except KeyboardInterrupt:
        logging.warning("ユーザーによりテストが中断されました。")
    except Exception as e:
        logging.error(f"ランダムエージェントの実行中にエラーが発生しました: {e}", exc_info=True)
    finally:
        if 'env' in locals():
            final_score = env.get_info().get('game_score', 0.0)
            logging.info("--- エピソード終了 ---")
            logging.info(f"合計フレーム: {frame_count}")
            logging.info(f"最終スコア: {final_score:.1f}")
            env.close()
            logging.info("Pygame ウィンドウを閉じました。")
            logging.info("--- [V9.3] ランダムテスト終了 ---")

if __name__ == "__main__":
    main()

