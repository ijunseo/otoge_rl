#
# otoge_rl_project: src/rhythm_game/env.py
#
# このファイルは、Jupyter Notebook (Cell 3, 5) から移植された
# メインの `RhythmGameEnv` クラスを定義します。
# ハードコードされていた設定値は `config.json` から受け取るように変更されています。
#

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class RhythmGameEnv(gym.Env):
    """
    強化学習エージェントのためのリズムゲーム環境 (Gymnasium 標準)。

    Attributes:
        metadata (dict): レンダリングモードとFPSを定義します。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, game_config: dict, density: str = 'medium', render_mode: str | None = None):
        """
        環境を初期化します。

        Args:
            game_config (dict): `config.json` からロードされたゲーム設定。
            density (str, optional): ノーツの密度 ('low', 'medium', 'high')。
            render_mode (str | None, optional): レンダリングモード。
        """
        super().__init__()
        self.render_mode = render_mode
        self.density = density

        # --- game_config から設定をロード ---
        gc_params = game_config["game_parameters"]
        self.screen_width = gc_params["screen_width"]
        self.screen_height = gc_params["screen_height"]
        self.num_lanes = gc_params["num_lanes"]
        self.note_width, self.note_height = 50, 20 # ノートの視覚的なサイズ (固定値)
        self.note_speed = game_config["game_parameters"]["note_speed"]
        self.judgment_line_y = gc_params["judgment_line_y"]

        # 判定範囲と報酬
        self.judgments = game_config["judgments"]
        self.miss_penalty = game_config["penalties"]["miss"]
        self.wrong_input_penalty = game_config["penalties"]["wrong_input"]

        # 密度設定
        self.density_settings = game_config["density_settings"]
        self.min_spacing_frames = self.density_settings[self.density]
        
        self.max_frames = 500  # エピソードの最大長 (固定値)

        # --- Gym空間の定義 ---
        self.action_space = spaces.Discrete(self.num_lanes + 1)  # 0..3: 打鍵, 4: 無入力
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8
        )

        # --- レンダリング初期化 ---
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Rhythm Game RL Environment")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        # --- ランタイム状態 (resetで初期化) ---
        self.notes = []
        self.current_frame = 0
        self.total_rl_score = 0  # RLエージェント用の報酬
        self.total_game_score = 0 # 実際のゲームスコア
        self.total_notes = 0
        self.hits = 0
        self.misses = 0
        self.max_score = 0

    def _generate_notes(self):
        """シードの一貫性のためにself.np_randomを使用。"""
        notes = []
        # (Cell 5 のロジック: 500フレーム、min_spacing_frames間隔で生成)
        for frame in range(0, self.max_frames, self.min_spacing_frames):
            if self.np_random.random() < 0.7:  # 70%の確率でノーツを生成
                lane = self.np_random.integers(0, self.num_lanes)
                # 開始位置は画面の上部外側
                y0 = -self.note_height / 2 - frame * self.note_speed
                notes.append([lane, y0])
        return notes

    def _calculate_max_score(self):
        """全てのノーツをperfectで叩いた場合の理論上の最高スコア。"""
        return len(self.notes) * self.judgments["perfect"]["reward"]

    def _get_obs(self):
        """現在のゲーム状態から観測 (グレースケール画像) を生成します。"""
        surface = pygame.Surface((self.screen_width, self.screen_height))
        surface.fill((255, 255, 255))
        lane_w = self.screen_width / self.num_lanes

        # レーンの境界線
        for i in range(1, self.num_lanes):
            pygame.draw.line(surface, (200, 200, 200),
                             (i * lane_w, 0), (i * lane_w, self.screen_height), 1)
        
        # ノーツの描画
        for lane, y in self.notes:
            if 0 <= y <= self.screen_height:
                x = (lane + 0.5) * lane_w
                rect = pygame.Rect(x - self.note_width / 2, y - self.note_height / 2,
                                  self.note_width, self.note_height)
                pygame.draw.rect(surface, (0, 0, 0), rect)

        # Pygame Surface を numpy 配列に変換 (W, H, 3)
        rgb = pygame.surfarray.array3d(surface)
        # (H, W, 3) に転置
        rgb = np.transpose(rgb, (1, 0, 2))
        # グレースケールに変換し、次元を維持 (float)
        gray = rgb.mean(axis=2, keepdims=True)
        # uint8 にキャスト (H, W, 1)
        return gray.astype(np.uint8)

    def _get_info(self):
        """デバッグ用の追加情報を返します。"""
        hit_rate = (self.hits / self.total_notes) if self.total_notes > 0 else 0.0
        return {
            "rl_score": self.total_rl_score,     # RLエージェントの総報酬
            "game_score": self.total_game_score, # 実際のゲームスコア
            "max_score": self.max_score,         # 理論上の最大スコア
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "notes_left": len(self.notes),
        }

    def reset(self, seed: int | None = None, options=None):
        """環境をリセットします。"""
        super().reset(seed=seed)
        self.current_frame = 0
        self.notes = self._generate_notes()
        self.total_notes = len(self.notes)
        self.hits = 0
        self.misses = 0
        self.total_rl_score = 0
        self.total_game_score = 0
        self.max_score = self._calculate_max_score()
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        """環境を1ステップ進めます。"""
        # 1) ノーツの移動
        for n in self.notes:
            n[1] += self.note_speed

        # 2) 入力処理
        rl_reward = 0.0
        game_score_delta = 0.0
        processed = False
        action_taken = (action < self.num_lanes)

        for note in self.notes[:]:
            lane, y = note
            diff = abs(y - self.judgment_line_y)

            # 判定範囲内で最も近いノートをチェック
            if diff <= self.judgments["ok"]["range"]:
                if action_taken and action == lane:
                    # 判定範囲内での打鍵
                    if diff <= self.judgments["perfect"]["range"]:
                        score = self.judgments["perfect"]["reward"]
                    elif diff <= self.judgments["good"]["range"]:
                        score = self.judgments["good"]["reward"]
                    else:
                        score = self.judgments["ok"]["reward"]
                    
                    rl_reward += score  # RL報酬 (Cell 5 のロジック)
                    game_score_delta += score # 実際のゲームスコア
                    self.hits += 1
                    self.notes.remove(note)
                    processed = True
                    break  # 1フレームあたり最大1ヒット
            elif y > self.judgment_line_y + self.judgments["ok"]["range"]:
                # 3) 見逃しノーツ (判定ラインを通過)
                self.misses += 1
                rl_reward += self.miss_penalty
                self.notes.remove(note)
                # 見逃しは break しない (複数のレーンで同時に見逃す可能性があるため)

        # 4) 誤入力ペナルティ
        if action_taken and not processed:
            # 判定範囲内に処理できるノートがなかった場合
            rl_reward += self.wrong_input_penalty

        self.total_rl_score += rl_reward
        self.total_game_score += game_score_delta
        self.current_frame += 1

        # 5) 終了判定
        # (Cell 5 のロジック: ノーツが0、かつフレームが500以上)
        terminated = (len(self.notes) == 0) and (self.current_frame >= self.max_frames)
        truncated = False  # 時間制限による打ち切りはここでは使用しない

        obs = self._get_obs()
        info = self._get_info()

        # 6) レンダリング
        if self.render_mode == "human":
            self.render()

        return obs, rl_reward, terminated, truncated, info

    def render(self):
        """人間または 'rgb_array' のために画面を描画します。"""
        # 観測用の描画 (判定ラインなし)
        obs_surface = pygame.surfarray.make_surface(np.transpose(self._get_obs().repeat(3, axis=2), (1, 0, 2)))
        
        # 判定ラインを観測用 Surface に追加描画
        pygame.draw.line(obs_surface, (255, 0, 0), (0, self.judgment_line_y),
                         (self.screen_width, self.judgment_line_y), 2)

        if self.render_mode == "human":
            self.screen.blit(obs_surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None
        
        elif self.render_mode == "rgb_array":
            # (H, W, 3) 形式で返す
            return np.transpose(pygame.surfarray.array3d(obs_surface), (1, 0, 2))

    def close(self):
        """環境をクローズし、Pygame を終了します。"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
