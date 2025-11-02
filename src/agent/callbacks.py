#
# otoge_rl_project: src/agent/callbacks.py
#
# Jupyter Notebook (Cell 2, 5) から移植されたカスタムコールバックを定義します。
# - TrainingRewardLogger: 学習中のエピソード報酬を記録します。
# - EvalWithAccuracy: 評価中に 'game_score' と 'max_score' を使用して
#                     正解率 (accuracy) を計算し、TensorBoard に記録します。
#

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


# --- 学習報酬ロガー (Cell 2/5) ---
class TrainingRewardLogger(BaseCallback):
    """
    エピソードが終了するたびに（現在のtimesteps、episode_reward）を記録します。
    Monitor ラッパーが info['episode'] を挿入してくれることを活用します。

    Args:
        verbose (int): 冗長レベル。
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.t = []
        self.rew = []

    def _on_step(self) -> bool:
        """
        このメソッドは、エージェントが 1 ステップ進むたびに呼び出されます。
        """
        # VecEnv 基準で infos はリスト。done 時に 'episode' キーが含まれる。
        infos = self.locals.get("infos", None)
        if infos:
            for info in infos:
                ep = info.get("episode")
                if ep is not None:  # エピソード終了時点
                    self.t.append(self.num_timesteps)
                    self.rew.append(ep["r"])
        return True


# --- 評価用シード (Cell 5) ---
EVAL_SEED_BASE = 2025


# --- 正解率付き評価コールバック (Cell 2/5) ---
class EvalWithAccuracy(EvalCallback):
    """
    基本的な EvalCallback を拡張し、「eval/accuracy_pct」と「eval/mean_game_score」を
    TensorBoard に記録し、(timesteps, accuracy_pct) を history として保管します。

    正解率 = 各エピソードの (game_score / max_score) * 100 の平均。
    """

    def __init__(self, eval_env, *args, n_eval_episodes=5, **kwargs):
        super().__init__(eval_env, *args, n_eval_episodes=n_eval_episodes, **kwargs)
        self.eval_steps = []
        self.eval_acc = []

    def _on_step(self) -> bool:
        """
        評価タイミングで呼び出され、カスタムメトリクスを計算します。
        """
        # 親クラスの _on_step を呼び出し (モデルの保存など)
        result = super()._on_step()

        # eval_freq ごとに評価が実行されたかチェック
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            acc_list = []
            gs_list = []  # Game Score リスト

            for ep in range(self.n_eval_episodes):
                # Gymnasium/VecEnv の reset と互換
                try:
                    # 評価用シードを使用して一貫性を保つ (Cell 5 のロジック)
                    obs, _ = self.eval_env.reset(seed=EVAL_SEED_BASE + ep)
                except Exception:
                    # シードをサポートしない古い VecEnv のためのフォールバック
                    obs = self.eval_env.reset()

                dones = np.array([False])
                while not dones[0]:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = self.eval_env.step(action)

                # VecEnv の場合、infos はリスト
                info = infos[0] if isinstance(infos, (list, tuple)) else infos

                gs = info.get("game_score", None)
                ms = info.get("max_score", None)

                if gs is not None:
                    gs_list.append(float(gs))

                if gs is not None and ms and ms > 0:
                    acc_list.append(100.0 * float(gs) / float(ms))

            if gs_list:
                self.logger.record("eval/mean_game_score", float(np.mean(gs_list)))

            if acc_list:
                mean_acc = float(np.mean(acc_list))
                # TensorBoard に記録
                self.logger.record("eval/accuracy_pct", mean_acc)
                # 内部記録 (グラフ用)
                self.eval_steps.append(self.num_timesteps)
                self.eval_acc.append(mean_acc)
        return result
