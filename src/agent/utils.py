#
# otoge_rl_project: src/agent/utils.py
#
# Jupyter Notebook (Cell 5) から移植された `make_env` ヘルパー関数を定義します。
# この関数は、環境を生成し、Monitor や Observation Wrapper でラップする役割を持ちます。
#

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import ResizeObservation, ReshapeObservation

# `rhythm_game` モジュールから `RhythmGameEnv` をインポート
from rhythm_game import RhythmGameEnv

def make_env(game_config: dict, density: str, resized_shape: tuple, rank: int, seed_base: int = 42):
    """
    SubprocVecEnv 用の環境生成関数 (Thunk) を作成します。

    Args:
        game_config (dict): `game/config.json` の内容。
        density (str): ノーツの密度。
        resized_shape (tuple): 観測をリサイズする先の形状 (H, W)。
        rank (int): 並列環境のインデックス。
        seed_base (int, optional): 環境のシードベース。

    Returns:
        callable: 環境を返す関数 (thunk)。
    """
    def _thunk():
        # 1. ベース環境を生成
        env = RhythmGameEnv(
            game_config=game_config,
            density=density,
            render_mode=None # SubprocVecEnv は 'rgb_array' や 'human' をサポートしない
        )
        # 2. Monitor ラッパー (報酬やエピソード長を info に記録するため)
        env = Monitor(env)
        # 3. 観測のリサイズ (H, W, 1) -> (resized_shape[0], resized_shape[1], 1)
        env = ResizeObservation(env, shape=resized_shape)
        # 4. 形状の再整形 (観測空間の形状を明示)
        env = ReshapeObservation(env, shape=(resized_shape[0], resized_shape[1], 1))
        
        # 5. 環境にシードを設定
        env.reset(seed=seed_base + rank)
        return env
    
    return _thunk
