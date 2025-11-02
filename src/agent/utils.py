#
# otoge_rl_project: src/agent/utils.py
#
# (V9.3)
# このファイルは、Gymnasium 環境を Stable Baselines3 用に
# ラップするためのヘルパー関数 `make_env` を提供します。
#
# V9.3 変更点:
# 1. (V8.6) `make_env` のシグネチャ (引数) を V8.6 計画に基づき更新。
# 2. (V9.3) `Monitor` ラッパーを削除。
#    `Monitor` は `main.py` の `setup_eval_env` 関数 (V9.3) と、
#    `evaluate.py` (V9.3) で `DummyVecEnv` の *外側* に適用されます。
#

import gymnasium as gym
import os

# Stable Baselines3 Wrappers
# (V9.3) Monitor は main.py/evaluate.py で管理するため、ここではインポートしない
from gymnasium.wrappers import ResizeObservation, ReshapeObservation

# `src` が PYTHONPATH にあるため、'rhythm_game' を直接インポート
from rhythm_game import RhythmGameEnv
from stable_baselines3.common.utils import set_random_seed

# --- 1. グローバル設定 ---

# 評価用のシードベース (学習と評価のエピソードが重複しないように)
EVAL_SEED_BASE = 42

# --- 2. (V9.3) make_env ヘルパー関数 ---

def make_env(
    game_config: dict,
    density: str,
    resized_shape: tuple[int, int],
    rank: int,
    seed_base: int = 0
) -> callable:
    """
    (V9.3) Stable Baselines3 の VecEnv (DummyVecEnv) 用の
    単一環境生成関数 (thunk) を作成します。

    Args:
        game_config (dict): rhythm_game/config.json の内容。
        density (str): 譜面の密度 ('low', 'medium', 'high')。
        resized_shape (tuple[int, int]): (H, W) のリサイズ後の形状。
        rank (int): 並列環境のインデックス (0 から N_ENVS-1)。
        seed_base (int): 乱数シードのベース値。

    Returns:
        callable: 引数を取らない環境初期化関数 (_init)。
    """
    def _init() -> gym.Env:
        """ VecEnv が呼び出す内部初期化関数 """
        # (V8.6) 決定論的な動作のため、ランクに基づきシードを設定
        seed = seed_base + rank
        set_random_seed(seed)
        
        # (V9.3) game_config を RhythmGameEnv に渡す
        env = RhythmGameEnv(
            game_config=game_config,
            density=density,
            render_mode=None # 学習/評価中は 'rgb_array' または 'human' を使用しない
        )
        
        # (V8.6) CnnPolicy のために観測をリサイズ (H, W, 1)
        # (V9.3) Monitor は VecEnv の外側でラップするため、ここでは適用しない
        env = ResizeObservation(env, shape=resized_shape)
        env = ReshapeObservation(env, shape=(resized_shape[0], resized_shape[1], 1))
        
        return env
    
    return _init

