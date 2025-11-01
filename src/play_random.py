#
# otoge_rl_project: src/play_random.py
#
# このファイルは、Jupyter Notebook (Cell 4) のランダムエージェントテストを
# 実行可能スクリプトとして移植したものです。
# 'uv run python -m play_random' で実行されます。
#
# 機能:
# 1. 'human' モードで Pygame ウィンドウを開き、環境が視覚的に正しく
#    動作するかをテストします。
# 2. Gymnasium 標準の env_checker を実行し、環境が標準仕様に
#    準拠しているかを確認します。
#

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import json
import logging
import sys

# `src` が PYTHONPATH にあるため、'rhythm_game' を直接インポート
try:
    from rhythm_game import RhythmGameEnv
except ImportError:
    logging.error("エラー: 'rhythm_game' モジュールが見つかりません。")
    sys.exit(1)

def setup_logging():
    """コンソールログの基本設定を行います。"""
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def load_game_config() -> dict:
    """
    ゲーム設定 JSON ファイルをロードします。

    Returns:
        dict: game_config
    """
    try:
        with open("src/rhythm_game/config.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as e:
        logging.error(f"設定ファイル (src/rhythm_game/config.json) が見つかりません: {e}")
        sys.exit(1)

def main():
    """
    メイン実行関数。
    """
    setup_logging()
    game_cfg = load_game_config()
    
    density = 'high' # テスト用の密度 (ハードコード)
    seed = 42        # テスト用シード (ハードコード)

    # --- 1. Gymnasium 標準チェック (Cell 4) ---
    logging.info("--- Gymnasium 環境標準チェック開始 (レンダリングなし) ---")
    try:
        env_check = RhythmGameEnv(game_config=game_cfg, density=density, render_mode=None)
        check_env(env_check)
        logging.info("✅ Gymnasium 環境標準チェックに合格しました！")
    except Exception as e:
        logging.error(f"❌ 環境標準チェックに失敗しました: {e}")
        # チェックに失敗しても、視覚テストのために続行する場合があります
        # sys.exit(1) 
    finally:
        env_check.close()


    # --- 2. ランダムエージェントによる視覚テスト (Cell 4) ---
    logging.info(f"\n--- ランダムエージェントのテスト開始 (Density: {density}, Seed: {seed}) ---")
    logging.info("Pygame ウィンドウが開き、1エピソードが終了するまで実行されます...")
    
    env = None
    try:
        # 'human' モードで環境を初期化
        env = RhythmGameEnv(game_config=game_cfg, density=density, render_mode='human')
        observation, info = env.reset(seed=seed)

        done = False
        truncated = False
        total_rl_score = 0
        total_game_score = 0
        frame_count = 0

        while not done and not truncated:
            action = env.action_space.sample()  # ランダムな行動を選択
            observation, reward, done, truncated, info = env.step(action)
            
            total_rl_score += reward
            frame_count += 1
            
            # 報酬が発生した時のみログを出力 (Cell 4 のロジック)
            if reward != 0:
                print(f"フレーム: {frame_count}, 行動: {action}, RL報酬: {reward:.1f}, "
                      f"現在のRL合計: {info.get('rl_score', 0.0):.1f}, "
                      f"現在のGameスコア: {info.get('game_score', 0.0):.1f}")
        
        logging.info("\n--- エピソード終了 ---")
        logging.info(f"合計フレーム: {frame_count}")
        logging.info(f"最終 RL スコア: {info.get('rl_score', 0.0):.1f}")
        logging.info(f"最終 Game スコア: {info.get('game_score', 0.0):.1f} / {info.get('max_score', 0.0):.1f}")

    except Exception as e:
        logging.error(f"ランダムエージェントの実行中にエラーが発生しました: {e}")
    finally:
        if env:
            env.close() # Pygame ウィンドウを閉じる
        logging.info("--- テスト終了 (環境クローズ済み) ---")

# 'uv run python -m play_random' で実行されるためのエントリーポイント
if __name__ == "__main__":
    main()
